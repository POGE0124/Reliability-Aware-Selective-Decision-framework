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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tgfm.data import RML2018Dataset, create_rml2018_split
from tgfm.models import build_teacher
from tgfm.utils import load_config, make_run_dir, save_config, seed_everything, summarize_snr_acc, write_csv, write_json


LOW_SNRS = [-10, -8, -6, -4, -2, 0]


class ConvBranch(nn.Module):
    def __init__(self, width: int = 384, dropout: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 96, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Conv1d(96, 192, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Conv1d(192, width, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).mean(dim=-1)


class RawLowSNRExpert(nn.Module):
    def __init__(self, num_classes: int, width: int = 384, hidden: int = 768, dropout: float = 0.20, branch_mode: str = "time_freq") -> None:
        super().__init__()
        if branch_mode not in {"time_freq", "time_only", "freq_only"}:
            raise ValueError(f"Unknown branch_mode: {branch_mode}")
        self.branch_mode = branch_mode
        self.time_branch = ConvBranch(width=width, dropout=dropout * 0.5)
        self.freq_branch = ConvBranch(width=width, dropout=dropout * 0.5)
        in_dim = width * 2 if branch_mode == "time_freq" else width
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden, num_classes),
        )

    @staticmethod
    def fft_iq(x: torch.Tensor) -> torch.Tensor:
        s = torch.complex(x[:, 0], x[:, 1])
        f = torch.fft.fft(s, dim=-1, norm="ortho")
        return torch.stack([f.real, f.imag], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.branch_mode == "time_only":
            h = self.time_branch(x)
        elif self.branch_mode == "freq_only":
            h = self.freq_branch(self.fft_iq(x))
        else:
            h = torch.cat([self.time_branch(x), self.freq_branch(self.fft_iq(x))], dim=-1)
        return self.head(h)


class ReliabilityRouter(nn.Module):
    def __init__(self, z_dim: int = 512, hidden: int = 512, num_routes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_routes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def route_from_snr(snr: torch.Tensor) -> torch.Tensor:
    s = snr.round().long()
    route = torch.full_like(s, 3)
    route = torch.where(s.le(-8), torch.zeros_like(route), route)
    route = torch.where((s.ge(-6) & s.le(-2)), torch.ones_like(route), route)
    route = torch.where(s.eq(0), torch.full_like(route, 2), route)
    return route


def update(stats, logits, labels, snrs) -> None:
    pred = logits.argmax(-1)
    ok = pred.eq(labels)
    for i in range(labels.numel()):
        s = int(float(snrs[i]))
        stats[s][0] += int(ok[i].item())
        stats[s][1] += 1


@torch.no_grad()
def eval_low(model, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits = model(x)
        correct += int(logits.argmax(dim=-1).eq(y).sum())
        total += int(y.numel())
    return correct / max(1, total)


@torch.no_grad()
def eval_native(cfg, teacher, router, expert, run_dir: Path, device: torch.device) -> None:
    ds = RML2018Dataset(cfg["paths"]["rml2018_h5"], cfg["paths"]["split_path"], "test", cfg["data"]["signal_length"])
    loader = DataLoader(
        ds,
        batch_size=int(cfg["eval"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg["eval"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg["eval"]["num_workers"]) > 0,
    )
    stats = {m: defaultdict(lambda: [0, 0]) for m in ["direct", "predicted_raw_expert", "oracle_raw_expert"]}
    for batch_idx, batch in enumerate(loader):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        snr = batch["snr"].to(device, non_blocking=True)
        z = teacher.encode(x)
        direct_logits = teacher.classifier(z)
        expert_logits = expert(x)
        pred_route = router(z).argmax(dim=-1)
        oracle_route = route_from_snr(snr)
        pred_logits = direct_logits.clone()
        oracle_logits = direct_logits.clone()
        pred_mask = pred_route.le(2)
        oracle_mask = oracle_route.le(2)
        if bool(pred_mask.any()):
            pred_logits[pred_mask] = expert_logits[pred_mask]
        if bool(oracle_mask.any()):
            oracle_logits[oracle_mask] = expert_logits[oracle_mask]
        update(stats["direct"], direct_logits, y, snr)
        update(stats["predicted_raw_expert"], pred_logits, y, snr)
        update(stats["oracle_raw_expert"], oracle_logits, y, snr)
        if batch_idx % 30 == 0:
            print(f"raw expert eval native batch={batch_idx}/{len(loader)}", flush=True)

    rows = []
    summary = {}
    for method in ["direct", "predicted_raw_expert", "oracle_raw_expert"]:
        by_snr = {int(k): v[0] / max(1, v[1]) for k, v in sorted(stats[method].items())}
        summary[method] = summarize_snr_acc(by_snr)
        for snr, acc in by_snr.items():
            rows.append({"domain": "native", "method": method, "snr": snr, "acc": acc, "correct": stats[method][snr][0], "total": stats[method][snr][1]})
    write_json(run_dir / "native_metrics_overall.json", summary)
    write_csv(run_dir / "native_metrics_by_snr.csv", rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--teacher-run", required=True)
    ap.add_argument("--router-run", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--branch-mode", choices=["time_freq", "time_only", "freq_only"], default="time_freq")
    ap.add_argument("--seed-override", type=int)
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_seed = int(cfg["seed"] if args.seed_override is None else args.seed_override)
    seed_everything(run_seed)
    create_rml2018_split(cfg["paths"]["rml2018_h5"], cfg["paths"]["split_path"], int(cfg["seed"]), float(cfg["data"]["train_ratio"]), float(cfg["data"]["val_ratio"]))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    tag = f"_{args.run_tag}" if args.run_tag else ""
    run_dir = make_run_dir(cfg["paths"]["output_root"], f"v8_raw_low_snr_expert{tag}")
    save_config({**cfg, "args": vars(args), "low_snrs": LOW_SNRS}, run_dir / "config.yaml")

    train_ds = RML2018Dataset(cfg["paths"]["rml2018_h5"], cfg["paths"]["split_path"], "train", cfg["data"]["signal_length"], snr_values=LOW_SNRS)
    val_ds = RML2018Dataset(cfg["paths"]["rml2018_h5"], cfg["paths"]["split_path"], "val", cfg["data"]["signal_length"], snr_values=LOW_SNRS)
    train_workers = int(cfg.get("train", {}).get("num_workers", cfg.get("eval", {}).get("num_workers", 4)))
    eval_workers = int(cfg.get("eval", {}).get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=True, num_workers=train_workers, pin_memory=device.type == "cuda", persistent_workers=train_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, drop_last=False, num_workers=eval_workers, pin_memory=device.type == "cuda", persistent_workers=eval_workers > 0)

    teacher = build_teacher(cfg).to(device)
    teacher.load_state_dict(torch.load(Path(args.teacher_run) / "best.pt", map_location="cpu")["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    router = ReliabilityRouter(int(cfg["teacher"]["z_dim"])).to(device)
    router.load_state_dict(torch.load(Path(args.router_run) / "best.pt", map_location="cpu")["router"])
    router.eval()
    for p in router.parameters():
        p.requires_grad = False

    expert = RawLowSNRExpert(int(cfg["data"]["num_classes"]), width=int(args.width), hidden=int(args.hidden), dropout=float(args.dropout), branch_mode=str(args.branch_mode)).to(device)
    opt = torch.optim.AdamW(expert.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(args.epochs), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.0
    best_epoch = 0
    stale = 0
    with (run_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "val_acc", "epoch_time_sec"])
        writer.writeheader()
        for epoch in range(1, int(args.epochs) + 1):
            t0 = time.time()
            expert.train()
            loss_sum = 0.0
            total = 0
            for batch in train_loader:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = expert(x)
                    loss = F.cross_entropy(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                b = int(y.numel())
                loss_sum += float(loss.detach().cpu()) * b
                total += b
            scheduler.step()
            val_acc = eval_low(expert, val_loader, device)
            if val_acc > best:
                best = val_acc
                best_epoch = epoch
                stale = 0
                torch.save({"expert": expert.state_dict(), "cfg": cfg, "args": vars(args), "best_epoch": best_epoch, "val_acc": best}, run_dir / "best.pt")
            else:
                stale += 1
            row = {"epoch": epoch, "loss": loss_sum / max(1, total), "val_acc": val_acc, "epoch_time_sec": time.time() - t0}
            writer.writerow(row)
            f.flush()
            print(f"raw expert epoch={epoch} loss={row['loss']:.4f} val_acc={val_acc:.4f} best={best:.4f}", flush=True)
            if stale >= 8:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch} best={best:.4f}", flush=True)
                break

    expert.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["expert"])
    eval_native(cfg, teacher, router, expert, run_dir, device)
    write_json(run_dir / "metrics_overall.json", {"best_epoch": best_epoch, "best_val_acc": best, "low_snrs": LOW_SNRS, "run_seed": run_seed, "branch_mode": args.branch_mode})
    print(f"RAW_LOW_SNR_EXPERT_OK run_dir={run_dir} best_val={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
