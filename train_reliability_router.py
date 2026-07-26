from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tgfm.data import RML2018Dataset, create_rml2018_split
from tgfm.models import build_teacher
from tgfm.utils import load_config, make_run_dir, save_config, seed_everything, write_json


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
    # 0: very_low -> MLP, 1: low -> DAE, 2: edge -> Flow, 3: reliable -> Direct
    s = snr.round().long()
    route = torch.full_like(s, 3)
    route = torch.where(s.le(-8), torch.zeros_like(route), route)
    route = torch.where((s.ge(-6) & s.le(-2)), torch.ones_like(route), route)
    route = torch.where(s.eq(0), torch.full_like(route, 2), route)
    return route


def balanced_indices(cfg: dict, split: str, max_per_route: int, seed: int) -> np.ndarray:
    split_path = Path(cfg["paths"]["split_path"])
    with np.load(split_path) as data:
        indices = data[split].astype(np.int64)
    import h5py

    with h5py.File(cfg["paths"]["rml2018_h5"], "r") as f:
        all_snrs = np.asarray(f["Z"][:, 0]).astype(np.int64)
        snrs = all_snrs[indices]
    route = np.full_like(snrs, 3)
    route[snrs <= -8] = 0
    route[(snrs >= -6) & (snrs <= -2)] = 1
    route[snrs == 0] = 2
    rng = np.random.default_rng(seed)
    parts = []
    for r in range(4):
        idx = indices[route == r].copy()
        rng.shuffle(idx)
        parts.append(idx[: min(max_per_route, idx.size)])
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out.astype(np.int64)


@torch.no_grad()
def evaluate(teacher, router, loader, device: torch.device) -> tuple[float, list[list[int]]]:
    router.eval()
    total = 0
    correct = 0
    conf = [[0 for _ in range(4)] for _ in range(4)]
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        snr = batch["snr"].to(device, non_blocking=True)
        target = route_from_snr(snr)
        logits = router(teacher.encode(x))
        pred = logits.argmax(dim=-1)
        correct += int(pred.eq(target).sum())
        total += int(target.numel())
        for t, p in zip(target.detach().cpu().tolist(), pred.detach().cpu().tolist(), strict=False):
            conf[int(t)][int(p)] += 1
    return correct / max(1, total), conf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--teacher-run", required=True)
    ap.add_argument("--max-per-route", type=int, default=120000)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    create_rml2018_split(cfg["paths"]["rml2018_h5"], cfg["paths"]["split_path"], int(cfg["seed"]), float(cfg["data"]["train_ratio"]), float(cfg["data"]["val_ratio"]))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(cfg["paths"]["output_root"], "v7_reliability_router")
    save_config({**cfg, "teacher_run": args.teacher_run, "max_per_route": args.max_per_route, "epochs": args.epochs}, run_dir / "config.yaml")

    teacher = build_teacher(cfg).to(device)
    ckpt = torch.load(Path(args.teacher_run) / "best.pt", map_location="cpu")
    teacher.load_state_dict(ckpt["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    train_ds = RML2018Dataset(cfg["paths"]["rml2018_h5"], cfg["paths"]["split_path"], "train", cfg["data"]["signal_length"])
    train_ds.indices = balanced_indices(cfg, "train", int(args.max_per_route), int(cfg["seed"]) + 91)
    val_ds = RML2018Dataset(cfg["paths"]["rml2018_h5"], cfg["paths"]["split_path"], "val", cfg["data"]["signal_length"])
    val_ds.indices = balanced_indices(cfg, "val", max(20000, int(args.max_per_route) // 5), int(cfg["seed"]) + 92)
    loader_kwargs = dict(batch_size=int(cfg["eval"]["batch_size"]), num_workers=int(cfg["eval"]["num_workers"]), pin_memory=device.type == "cuda", persistent_workers=int(cfg["eval"]["num_workers"]) > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)

    router = ReliabilityRouter(int(cfg["teacher"]["z_dim"])).to(device)
    opt = torch.optim.AdamW(router.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(args.epochs), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.0
    best_epoch = 0
    with (run_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "train_acc", "val_acc", "epoch_time_sec"])
        writer.writeheader()
        for epoch in range(1, int(args.epochs) + 1):
            t0 = time.time()
            router.train()
            loss_sum = 0.0
            correct = 0
            total = 0
            for batch in train_loader:
                x = batch["x"].to(device, non_blocking=True)
                snr = batch["snr"].to(device, non_blocking=True)
                target = route_from_snr(snr)
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    z = teacher.encode(x)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = router(z)
                    loss = F.cross_entropy(logits, target)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                b = int(target.numel())
                loss_sum += float(loss.detach().cpu()) * b
                correct += int(logits.argmax(dim=-1).eq(target).sum())
                total += b
            scheduler.step()
            val_acc, conf = evaluate(teacher, router, val_loader, device)
            train_acc = correct / max(1, total)
            if val_acc > best:
                best = val_acc
                best_epoch = epoch
                torch.save({"router": router.state_dict(), "cfg": cfg, "best_epoch": best_epoch, "val_acc": best, "confusion": conf}, run_dir / "best.pt")
            row = {"epoch": epoch, "loss": loss_sum / max(1, total), "train_acc": train_acc, "val_acc": val_acc, "epoch_time_sec": time.time() - t0}
            writer.writerow(row)
            f.flush()
            print(f"router epoch={epoch} loss={row['loss']:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} best={best:.4f}", flush=True)
    write_json(run_dir / "metrics_overall.json", {"best_epoch": best_epoch, "best_val_acc": best})
    print(f"ROUTER_OK run_dir={run_dir} best_val={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
