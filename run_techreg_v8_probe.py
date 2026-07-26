from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tgfm.data import TechRegDataset
from tgfm.models import build_teacher
from tgfm.utils import load_config, make_run_dir, save_config, seed_everything, write_csv, write_json
from train_raw_low_snr_expert import RawLowSNRExpert, ReliabilityRouter


ROUTE_NAMES = ["very_low", "low", "edge", "reliable"]
CLASS_NAMES = ["wf", "lte", "dvbt"]


def expert_feature(expert: RawLowSNRExpert, x: torch.Tensor) -> torch.Tensor:
    if expert.branch_mode == "time_only":
        return expert.time_branch(x)
    if expert.branch_mode == "freq_only":
        return expert.freq_branch(expert.fft_iq(x))
    return torch.cat([expert.time_branch(x), expert.freq_branch(expert.fft_iq(x))], dim=-1)


class TechRegProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 3) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class FeatureExtractor(nn.Module):
    def __init__(self, teacher, router, expert, mode: str) -> None:
        super().__init__()
        self.teacher = teacher
        self.router = router
        self.expert = expert
        self.mode = mode

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.teacher.encode(x)
        h = expert_feature(self.expert, x)
        route_logits = self.router(z)
        route = route_logits.argmax(dim=-1)
        route_prob = F.softmax(route_logits, dim=-1)
        if self.mode == "teacher":
            feat = z
        elif self.mode == "raw_time":
            feat = h
        elif self.mode == "concat":
            feat = torch.cat([z, h], dim=-1)
        elif self.mode == "routed":
            # Same dimensionality as concat; reliable samples keep teacher-only information,
            # unreliable samples expose raw expert information.
            zero_z = torch.zeros_like(z)
            zero_h = torch.zeros_like(h)
            feat_teacher = torch.cat([z, zero_h], dim=-1)
            feat_raw = torch.cat([zero_z, h], dim=-1)
            feat = torch.where(route.eq(3).view(-1, 1), feat_teacher, feat_raw)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return feat, route, route_prob


def feature_dim(mode: str) -> int:
    if mode == "teacher":
        return 512
    if mode == "raw_time":
        return 384
    if mode in {"concat", "routed"}:
        return 896
    raise ValueError(mode)


def balanced_subset(ds: TechRegDataset, max_per_class: int | None, seed: int) -> Subset | TechRegDataset:
    if max_per_class is None:
        return ds
    g = torch.Generator().manual_seed(seed)
    labels = torch.tensor([row[1] for row in ds.rows])
    keep = []
    for c in sorted(labels.unique().tolist()):
        idx = torch.where(labels == c)[0]
        idx = idx[torch.randperm(idx.numel(), generator=g)]
        keep.extend(idx[: min(max_per_class, idx.numel())].tolist())
    return Subset(ds, keep)


def fewshot_subset(ds: TechRegDataset, shots: int, seed: int) -> Subset:
    return balanced_subset(ds, shots, seed)


@torch.no_grad()
def extract_dataset(extractor: FeatureExtractor, loader, device: torch.device) -> dict:
    feats, labels, routes, route_probs, paths = [], [], [], [], []
    extractor.eval()
    for batch_idx, batch in enumerate(loader):
        x = batch["x"].to(device, non_blocking=True)
        feat, route, prob = extractor(x)
        feats.append(feat.cpu())
        labels.append(batch["label"].long())
        routes.append(route.cpu())
        route_probs.append(prob.cpu())
        paths.extend(batch.get("path", [""] * x.shape[0]))
        if batch_idx % 50 == 0:
            print(f"extract batch={batch_idx}/{len(loader)}", flush=True)
    return {
        "feat": torch.cat(feats),
        "label": torch.cat(labels),
        "route": torch.cat(routes),
        "route_prob": torch.cat(route_probs),
        "path": paths,
    }


def train_probe(train_pack: dict, val_pack: dict, in_dim: int, device: torch.device, epochs: int, lr: float) -> tuple[TechRegProbe, dict]:
    model = TechRegProbe(in_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    x_train = train_pack["feat"].to(device)
    y_train = train_pack["label"].to(device)
    x_val = val_pack["feat"].to(device)
    y_val = val_pack["label"].to(device)
    best = {"acc": -1.0, "epoch": 0, "state": None}
    log = []
    bs = 2048
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        order = torch.randperm(y_train.numel(), device=device)
        loss_sum = 0.0
        total = 0
        for start in range(0, y_train.numel(), bs):
            idx = order[start : start + bs]
            opt.zero_grad(set_to_none=True)
            logits = model(x_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach().cpu()) * idx.numel()
            total += idx.numel()
        model.eval()
        pred = model(x_val).argmax(dim=-1)
        acc = float(pred.eq(y_val).float().mean().cpu())
        if acc > best["acc"]:
            best = {"acc": acc, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        log.append({"epoch": epoch, "loss": loss_sum / max(1, total), "val_acc": acc, "epoch_time_sec": time.time() - t0})
        print(f"probe epoch={epoch} val_acc={acc:.4f} best={best['acc']:.4f}", flush=True)
    model.load_state_dict(best["state"])
    return model, {"best_epoch": best["epoch"], "best_val_acc": best["acc"], "log": log}


@torch.no_grad()
def evaluate_probe(model: TechRegProbe, pack: dict, device: torch.device) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    model.eval()
    x = pack["feat"].to(device)
    y = pack["label"].to(device)
    route = pack["route"]
    prob = pack["route_prob"]
    logits = model(x)
    pred = logits.argmax(dim=-1)
    ok = pred.eq(y)
    conf = torch.zeros(3, 3, dtype=torch.long)
    for yi, pi in zip(y.cpu(), pred.cpu()):
        conf[int(yi), int(pi)] += 1
    overall = {"acc": float(ok.float().mean().cpu()), "confusion_matrix": conf.tolist()}

    per_class_rows = []
    ok_cpu = ok.cpu()
    y_cpu = y.cpu()
    pred_cpu = pred.cpu()
    for c in range(3):
        mask = y_cpu.eq(c)
        n = int(mask.sum())
        per_class_rows.append({
            "class": CLASS_NAMES[c],
            "n": n,
            "acc": float(ok_cpu[mask].float().mean()) if n else 0.0,
            "pred_wf": int(pred_cpu[mask].eq(0).sum()) if n else 0,
            "pred_lte": int(pred_cpu[mask].eq(1).sum()) if n else 0,
            "pred_dvbt": int(pred_cpu[mask].eq(2).sum()) if n else 0,
        })

    route_rows = []
    for r in range(4):
        mask = route.eq(r)
        n = int(mask.sum())
        route_rows.append({"route": ROUTE_NAMES[r], "n": n, "acc": float(ok.cpu()[mask].float().mean()) if n else 0.0})

    dist_rows = []
    for c in range(3):
        cmask = y.cpu().eq(c)
        total = int(cmask.sum())
        row = {"class": CLASS_NAMES[c], "total": total}
        for r in range(4):
            row[f"{ROUTE_NAMES[r]}_count"] = int((cmask & route.eq(r)).sum())
            row[f"{ROUTE_NAMES[r]}_ratio"] = row[f"{ROUTE_NAMES[r]}_count"] / max(1, total)
        dist_rows.append(row)

    calib_rows = []
    reliable = prob[:, 3]
    bins = torch.linspace(0, 1, 11)
    for b in range(10):
        mask = (reliable >= bins[b]) & (reliable < bins[b + 1] if b < 9 else reliable <= bins[b + 1])
        n = int(mask.sum())
        calib_rows.append({
            "bin": b,
            "reliability_min": float(bins[b]),
            "reliability_max": float(bins[b + 1]),
            "n": n,
            "acc": float(ok.cpu()[mask].float().mean()) if n else 0.0,
        })
    return overall, route_rows, dist_rows, calib_rows, per_class_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/rml2018_tgfm_v7_curriculum.yaml"))
    ap.add_argument("--teacher-run", required=True)
    ap.add_argument("--router-run", required=True)
    ap.add_argument("--expert-run", required=True)
    ap.add_argument("--mode", choices=["teacher", "raw_time", "concat", "routed"], required=True)
    ap.add_argument("--shots", type=int)
    ap.add_argument("--max-windows-per-file", type=int, default=200, help="Use <=0 for all non-overlapping windows in each file.")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    window_tag = "full" if int(args.max_windows_per_file) <= 0 else str(args.max_windows_per_file)
    tag = f"techreg_v8_{args.mode}" + (f"_{args.shots}shot" if args.shots else "_linear") + f"_{window_tag}win"
    run_dir = make_run_dir(cfg["paths"]["output_root"], tag)
    save_config({**cfg, "args": vars(args)}, run_dir / "config.yaml")

    teacher = build_teacher(cfg).to(device)
    teacher.load_state_dict(torch.load(Path(args.teacher_run) / "best.pt", map_location="cpu")["model"])
    teacher.eval()
    router = ReliabilityRouter(int(cfg["teacher"]["z_dim"])).to(device)
    router.load_state_dict(torch.load(Path(args.router_run) / "best.pt", map_location="cpu")["router"])
    router.eval()
    ckpt = torch.load(Path(args.expert_run) / "best.pt", map_location="cpu")
    expert_args = ckpt.get("args", {})
    expert = RawLowSNRExpert(
        int(cfg["data"]["num_classes"]),
        width=int(expert_args.get("width", 384)),
        hidden=int(expert_args.get("hidden", 768)),
        dropout=float(expert_args.get("dropout", 0.20)),
        branch_mode=str(expert_args.get("branch_mode", "time_only")),
    ).to(device)
    expert.load_state_dict(ckpt["expert"])
    expert.eval()
    extractor = FeatureExtractor(teacher, router, expert, args.mode).to(device)

    root = Path(cfg["paths"]["data_root"]) / "TechReg"
    max_windows = None if int(args.max_windows_per_file) <= 0 else int(args.max_windows_per_file)
    train_ds = TechRegDataset(root, ["gentbrugge", "igent", "merelbeke", "rabot"], cfg["data"]["signal_length"], max_windows_per_file=max_windows)
    val_ds = TechRegDataset(root, ["reep"], cfg["data"]["signal_length"], max_windows_per_file=max_windows)
    test_ds = TechRegDataset(root, ["uz"], cfg["data"]["signal_length"], max_windows_per_file=max_windows)
    if args.shots is not None:
        train_ds = fewshot_subset(train_ds, int(args.shots), int(cfg["seed"]))
    else:
        train_ds = balanced_subset(train_ds, None, int(cfg["seed"]))
    bs = 512
    workers = 4
    kwargs = {"num_workers": workers, "pin_memory": device.type == "cuda", "persistent_workers": workers > 0}
    train_pack = extract_dataset(extractor, DataLoader(train_ds, batch_size=bs, shuffle=False, **kwargs), device)
    val_pack = extract_dataset(extractor, DataLoader(val_ds, batch_size=bs, shuffle=False, **kwargs), device)
    test_pack = extract_dataset(extractor, DataLoader(test_ds, batch_size=bs, shuffle=False, **kwargs), device)

    model, train_info = train_probe(train_pack, val_pack, feature_dim(args.mode), device, int(args.epochs), lr=5e-4)
    test, route_rows, dist_rows, calib_rows, per_class_rows = evaluate_probe(model, test_pack, device)
    torch.save({"model": model.state_dict(), "train_info": train_info}, run_dir / "best.pt")
    write_csv(run_dir / "train_log.csv", train_info["log"])
    write_json(run_dir / "metrics_overall.json", {"best_epoch": train_info["best_epoch"], "best_val_acc": train_info["best_val_acc"], "test": test})
    write_csv(run_dir / "metrics_per_class.csv", per_class_rows)
    write_csv(run_dir / "metrics_by_route.csv", route_rows)
    write_csv(run_dir / "router_distribution_by_class.csv", dist_rows)
    write_csv(run_dir / "reliability_calibration.csv", calib_rows)
    print(f"TECHREG_V8_OK run_dir={run_dir} test_acc={test['acc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
