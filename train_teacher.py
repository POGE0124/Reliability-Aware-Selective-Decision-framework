from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tgfm.data import RML2018Dataset, create_rml2018_split
from tgfm.models import build_teacher
from tgfm.utils import load_config, make_run_dir, save_config, seed_everything, write_json


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    by_snr = {}
    conf = torch.zeros(24, 24, dtype=torch.long)
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        snr = batch["snr"]
        logits = model(x)["logits"]
        pred = logits.argmax(dim=-1)
        ok = pred.eq(y)
        total += y.numel()
        correct += int(ok.sum())
        for i in range(y.numel()):
            s = int(float(snr[i]))
            by_snr.setdefault(s, [0, 0])
            by_snr[s][0] += int(ok[i].item())
            by_snr[s][1] += 1
            conf[int(y[i]), int(pred[i])] += 1
    return {
        "acc": correct / max(1, total),
        "by_snr": {str(k): v[0] / max(1, v[1]) for k, v in sorted(by_snr.items())},
        "confusion_matrix": conf.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "rml2018_tgfm.yaml"))
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-eval", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    create_rml2018_split(cfg["paths"]["rml2018_h5"], cfg["paths"]["split_path"], int(cfg["seed"]), float(cfg["data"]["train_ratio"]), float(cfg["data"]["val_ratio"]))
    run_dir = make_run_dir(cfg["paths"]["output_root"], "teacher")
    save_config(cfg, run_dir / "config.yaml")
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    tc = cfg["teacher"]
    train_ds = RML2018Dataset(cfg["paths"]["rml2018_h5"], cfg["paths"]["split_path"], "train", cfg["data"]["signal_length"], snr_min=cfg["data"]["teacher_snr_min"], max_samples=args.max_train)
    val_ds = RML2018Dataset(cfg["paths"]["rml2018_h5"], cfg["paths"]["split_path"], "val", cfg["data"]["signal_length"], snr_min=cfg["data"]["teacher_snr_min"], max_samples=args.max_eval)
    test_ds = RML2018Dataset(cfg["paths"]["rml2018_h5"], cfg["paths"]["split_path"], "test", cfg["data"]["signal_length"], max_samples=args.max_eval)
    kwargs = {"num_workers": int(tc["num_workers"]), "pin_memory": device.type == "cuda", "persistent_workers": int(tc["num_workers"]) > 0}
    if int(tc["num_workers"]) > 0:
        kwargs["prefetch_factor"] = int(tc["prefetch_factor"])
    train_loader = DataLoader(train_ds, batch_size=int(tc["batch_size"]), shuffle=True, drop_last=False, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=int(tc["batch_size"]), shuffle=False, drop_last=False, **kwargs)
    test_loader = DataLoader(test_ds, batch_size=int(tc["batch_size"]), shuffle=False, drop_last=False, **kwargs)

    model = build_teacher(cfg).to(device)
    epochs = int(args.epochs or tc["epochs"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]))
    warmup = int(tc.get("warmup_epochs", 0))
    min_lr = float(tc["min_lr"])
    base_lr = float(tc["lr"])

    def lr_factor(epoch_idx: int) -> float:
        step = epoch_idx + 1
        if warmup > 0 and step <= warmup:
            return step / float(warmup)
        denom = max(1, epochs - warmup)
        progress = min(1.0, max(0.0, (step - warmup) / float(denom)))
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
        return (min_lr / base_lr) + (1.0 - min_lr / base_lr) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(tc["amp"]) and device.type == "cuda")
    best = -1.0
    best_epoch = 0
    stale = 0
    with (run_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "val_acc", "lr", "epoch_time_sec"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            sums = 0.0
            n = 0
            for batch in train_loader:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    loss = F.cross_entropy(model(x)["logits"], y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["grad_clip"]))
                scaler.step(opt)
                scaler.update()
                sums += float(loss.detach().cpu()) * y.numel()
                n += y.numel()
            scheduler.step()
            val = evaluate(model, val_loader, device)
            if val["acc"] > best:
                best = val["acc"]
                best_epoch = epoch
                stale = 0
                torch.save({"model": model.state_dict(), "cfg": cfg, "best_epoch": best_epoch, "val": val}, run_dir / "best.pt")
            else:
                stale += 1
            writer.writerow({"epoch": epoch, "loss": sums / max(1, n), "val_acc": val["acc"], "lr": opt.param_groups[0]["lr"], "epoch_time_sec": time.time() - t0})
            f.flush()
            print(f"teacher epoch={epoch} loss={sums/max(1,n):.4f} val_acc={val['acc']:.4f} best={best:.4f}", flush=True)
            if stale >= int(tc.get("early_stop_patience", 0)):
                print(f"early_stop epoch={epoch} best_epoch={best_epoch} best={best:.4f}", flush=True)
                break
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_loader, device)
    write_json(run_dir / "metrics_overall.json", {"best_epoch": best_epoch, "best_val_acc": best, "test": test})
    print(f"TEACHER_RUN_OK run_dir={run_dir} test_acc={test['acc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
