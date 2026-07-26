from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml


RML2018_MODS = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
]


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_run_dir(output_root: str | Path, prefix: str) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def normalize_iq(x: np.ndarray) -> np.ndarray:
    """Max-absolute IQ normalization.

    This is kept for compatibility, but it can alter sample power distribution.
    RMS normalization is preferred for controlled AWGN experiments.
    """
    scale = float(np.max(np.abs(x)))
    if scale > 1e-6:
        x = x / scale
    return x.astype(np.float32, copy=False)


def normalize_iq_rms(x: np.ndarray) -> np.ndarray:
    """Normalize IQ by RMS power while preserving relative waveform dynamics."""
    power = float(np.mean(x**2))
    scale = float(np.sqrt(power + 1e-8))
    return (x / scale).astype(np.float32, copy=False)


def add_awgn_torch(x: torch.Tensor, snr_db: torch.Tensor | float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Add AWGN to IQ tensors with shape [B, 2, L].

    Power is mean(I^2 + Q^2) over time. For a requested SNR, total noise
    power is split equally across I/Q, so each channel receives independent
    N(0, sigma^2) noise with sigma = sqrt(noise_power / 2).
    """
    if not torch.is_tensor(snr_db):
        snr_db = torch.full((x.shape[0],), float(snr_db), device=x.device, dtype=x.dtype)
    elif snr_db.ndim == 0:
        snr_db = snr_db.expand(x.shape[0]).to(device=x.device, dtype=x.dtype)
    else:
        snr_db = snr_db.to(device=x.device, dtype=x.dtype)
    power = x.pow(2).sum(dim=1).mean(dim=-1).clamp_min(1e-8)
    noise_power = power / torch.pow(torch.tensor(10.0, device=x.device, dtype=x.dtype), snr_db / 10.0)
    sigma = torch.sqrt(noise_power / 2.0).view(-1, 1, 1)
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return x + sigma * noise


def accuracy_by_snr(logits: torch.Tensor, labels: torch.Tensor, snrs: torch.Tensor) -> dict:
    pred = logits.argmax(dim=-1)
    ok = pred.eq(labels)
    rows: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for i in range(labels.numel()):
        key = int(float(snrs[i]))
        rows[key][0] += int(ok[i].item())
        rows[key][1] += 1
    return {k: v[0] / max(1, v[1]) for k, v in rows.items()}


def merge_acc_rows(rows: Iterable[tuple[int, int, int]]) -> dict[int, float]:
    sums: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for snr, correct, total in rows:
        sums[int(snr)][0] += int(correct)
        sums[int(snr)][1] += int(total)
    return {k: v[0] / max(1, v[1]) for k, v in sorted(sums.items())}


def summarize_snr_acc(by_snr: dict[int, float]) -> dict:
    bands = {
        "low_acc": [-10, -8, -6, -4, -2, 0],
        "mid_acc": [2, 4, 6, 8, 10],
        "high_acc": [12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    }
    vals = list(by_snr.values())
    out = {
        "average_acc": float(np.mean(vals)) if vals else 0.0,
        "snr_-20_acc": float(by_snr.get(-20, 0.0)),
        "best_acc": float(max(vals)) if vals else 0.0,
    }
    for name, snrs in bands.items():
        got = [by_snr[s] for s in snrs if s in by_snr]
        out[name] = float(np.mean(got)) if got else 0.0
    return out


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def sinusoidal_time(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(10000.0) / max(1, half - 1)))
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[-1]))
    return emb
