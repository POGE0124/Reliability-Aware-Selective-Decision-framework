from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tgfm.models import SpectralTemporalTeacher
from tgfm.utils import make_run_dir, normalize_iq, seed_everything, write_csv, write_json
from train_raw_low_snr_expert import RawLowSNRExpert, ReliabilityRouter, route_from_snr


MODS_10A = ["8PSK", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "AM-DSB", "AM-SSB", "WBFM"]
MODS_10B = ["8PSK", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "AM-DSB", "WBFM"]
LOW_SNRS = [-10, -8, -6, -4, -2, 0]
MID_SNRS = [2, 4, 6, 8, 10]
HIGH_SNRS = [12, 14, 16, 18]
TXT_RE = re.compile(r"^(?P<mod>.+) (?P<snr>-?\d+)\.txt$")
COMPLEX_RE = re.compile(r"\(([^()]+j)\)")


def stable_seed(*parts: object) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8"))


def split_indices(n: int, seed: int, train_ratio: float = 0.8, val_ratio: float = 0.1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = np.arange(n, dtype=np.int64)
    rng.shuffle(order)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(1, n_train), n - 2)
    n_val = min(max(1, n_val), n - n_train - 1)
    return {
        "train": np.sort(order[:n_train]),
        "val": np.sort(order[n_train : n_train + n_val]),
        "test": np.sort(order[n_train + n_val :]),
    }


def parse_complex_line(line: str, length: int = 128) -> np.ndarray | None:
    tokens = COMPLEX_RE.findall(line)
    if not tokens:
        return None
    values = np.asarray([complex(tok.replace(" ", "")) for tok in tokens[:length]], dtype=np.complex64)
    x = np.zeros((2, length), dtype=np.float32)
    x[0, : values.shape[0]] = values.real
    x[1, : values.shape[0]] = values.imag
    return normalize_iq(x)


def ensure_txt_cache(source: Path, cache_dir: Path, signal_length: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{stable_seed(str(source), signal_length):08x}.npy"
    if cache_path.exists():
        return cache_path
    samples = []
    with source.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            arr = parse_complex_line(line, signal_length)
            if arr is not None:
                samples.append(arr)
    if not samples:
        raise RuntimeError(f"No samples parsed from {source}")
    tmp = cache_path.with_suffix(".tmp.npy")
    np.save(tmp, np.stack(samples, axis=0).astype(np.float32))
    tmp.replace(cache_path)
    return cache_path


class RML2016Dataset(Dataset):
    def __init__(
        self,
        dataset: str,
        root: str | Path,
        split: str,
        seed: int,
        signal_length: int = 128,
        snr_min: int | None = None,
        snr_values: list[int] | None = None,
        cache_dir: str | Path | None = None,
        max_per_snr_class: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.root = Path(root)
        self.split = split
        self.seed = int(seed)
        self.signal_length = int(signal_length)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.root / "cache" / "rml2016_txt_npy"
        self._pkl = None
        self._mmap_cache: dict[str, np.ndarray] = {}
        self.mods = MODS_10A if dataset == "10a" else MODS_10B
        self.mod_to_label = {m: i for i, m in enumerate(self.mods)}
        self.rows: list[dict] = []
        if dataset == "10a":
            self._build_10a(snr_min, snr_values, max_per_snr_class)
        elif dataset == "10b":
            self._build_10b(snr_min, snr_values, max_per_snr_class)
        else:
            raise ValueError("dataset must be 10a or 10b")

    def _keep_snr(self, snr: int, snr_min: int | None, snr_values: list[int] | None) -> bool:
        if snr_min is not None and snr < int(snr_min):
            return False
        if snr_values is not None and snr not in set(int(s) for s in snr_values):
            return False
        return True

    def _limit(self, indices: np.ndarray, mod: str, snr: int, max_per_snr_class: int | None) -> np.ndarray:
        if max_per_snr_class is None or indices.size <= int(max_per_snr_class):
            return indices
        rng = np.random.default_rng(self.seed + stable_seed(self.dataset, self.split, mod, snr, "limit"))
        order = indices.copy()
        rng.shuffle(order)
        return np.sort(order[: int(max_per_snr_class)])

    def _build_10a(self, snr_min: int | None, snr_values: list[int] | None, max_per_snr_class: int | None) -> None:
        p = self.root / "RML201610a" / "RML2016.10a_dict.pkl"
        with p.open("rb") as f:
            data = pickle.load(f, encoding="latin1")
        self.pkl_path = p
        for mod in self.mods:
            for snr in sorted({int(k[1]) for k in data if k[0] == mod}):
                if not self._keep_snr(snr, snr_min, snr_values):
                    continue
                n = int(data[(mod, snr)].shape[0])
                idxs = split_indices(n, self.seed + stable_seed("10a", mod, snr))[self.split]
                idxs = self._limit(idxs, mod, snr, max_per_snr_class)
                for idx in idxs.tolist():
                    self.rows.append({"kind": "pkl", "mod": mod, "snr": snr, "index": idx, "label": self.mod_to_label[mod]})

    def _build_10b(self, snr_min: int | None, snr_values: list[int] | None, max_per_snr_class: int | None) -> None:
        d = self.root / "RML201610b" / "2016.10b"
        for path in sorted(d.glob("*.txt")):
            m = TXT_RE.match(path.name)
            if not m:
                continue
            mod = m.group("mod")
            snr = int(m.group("snr"))
            if mod not in self.mod_to_label or not self._keep_snr(snr, snr_min, snr_values):
                continue
            cache_path = ensure_txt_cache(path, self.cache_dir, self.signal_length)
            arr = np.load(cache_path, mmap_mode="r")
            n = int(arr.shape[0])
            idxs = split_indices(n, self.seed + stable_seed("10b", mod, snr))[self.split]
            idxs = self._limit(idxs, mod, snr, max_per_snr_class)
            for idx in idxs.tolist():
                self.rows.append({"kind": "npy", "path": str(cache_path), "mod": mod, "snr": snr, "index": idx, "label": self.mod_to_label[mod]})

    def _load_pkl(self):
        if self._pkl is None:
            with self.pkl_path.open("rb") as f:
                self._pkl = pickle.load(f, encoding="latin1")
        return self._pkl

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        if row["kind"] == "pkl":
            x = self._load_pkl()[(row["mod"], row["snr"])][row["index"]].astype(np.float32)
            x = normalize_iq(x[:, : self.signal_length])
        else:
            path = row["path"]
            if path not in self._mmap_cache:
                self._mmap_cache[path] = np.load(path, mmap_mode="r")
            x = np.asarray(self._mmap_cache[path][row["index"]], dtype=np.float32)
        return {
            "x": torch.from_numpy(x.copy()),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "snr": torch.tensor(float(row["snr"]), dtype=torch.float32),
        }


@torch.no_grad()
def evaluate_logits(method: str, logits: torch.Tensor, y: torch.Tensor, snr: torch.Tensor, stats: dict) -> None:
    pred = logits.argmax(dim=-1)
    ok = pred.eq(y)
    for i in range(y.numel()):
        s = int(float(snr[i]))
        stats[method][s][0] += int(ok[i].item())
        stats[method][s][1] += 1


def summarize(by_snr: dict[int, float]) -> dict:
    def avg(snrs: list[int]) -> float:
        vals = [by_snr[s] for s in snrs if s in by_snr]
        return float(sum(vals) / max(1, len(vals)))

    return {
        "avg": float(sum(by_snr.values()) / max(1, len(by_snr))),
        "low": avg(LOW_SNRS),
        "mid": avg(MID_SNRS),
        "high": avg(HIGH_SNRS),
        "minus10": float(by_snr.get(-10, 0.0)),
        "best": float(max(by_snr.values()) if by_snr else 0.0),
    }


@torch.no_grad()
def evaluate_all(teacher, router, expert, loader, device: torch.device, run_dir: Path, num_classes: int) -> dict:
    teacher.eval()
    router.eval()
    expert.eval()
    stats = {m: defaultdict(lambda: [0, 0]) for m in ["teacher", "expert", "hybrid_pred", "hybrid_oracle"]}
    conf = {m: torch.zeros(num_classes, num_classes, dtype=torch.long) for m in stats}
    route_dist = defaultdict(lambda: [0, 0, 0, 0])
    for batch_idx, batch in enumerate(loader):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        snr = batch["snr"].to(device, non_blocking=True)
        out = teacher(x)
        z = out["z"]
        teacher_logits = out["logits"]
        expert_logits = expert(x)
        pred_route = router(z).argmax(dim=-1)
        oracle_route = route_from_snr(snr)
        hybrid_pred = teacher_logits.clone()
        hybrid_oracle = teacher_logits.clone()
        pred_mask = pred_route.le(2)
        oracle_mask = oracle_route.le(2)
        if bool(pred_mask.any()):
            hybrid_pred[pred_mask] = expert_logits[pred_mask]
        if bool(oracle_mask.any()):
            hybrid_oracle[oracle_mask] = expert_logits[oracle_mask]
        logits_by_method = {
            "teacher": teacher_logits,
            "expert": expert_logits,
            "hybrid_pred": hybrid_pred,
            "hybrid_oracle": hybrid_oracle,
        }
        for m, logits in logits_by_method.items():
            evaluate_logits(m, logits, y, snr, stats)
            pred = logits.argmax(dim=-1).cpu()
            for yi, pi in zip(y.cpu(), pred):
                conf[m][int(yi), int(pi)] += 1
        for s, r in zip(snr.cpu(), pred_route.cpu()):
            route_dist[int(float(s))][int(r)] += 1
        if batch_idx % 50 == 0:
            print(f"eval batch={batch_idx}/{len(loader)}", flush=True)

    rows = []
    overall = {}
    for m in stats:
        by_snr = {int(s): c / max(1, n) for s, (c, n) in sorted(stats[m].items())}
        overall[m] = summarize(by_snr)
        overall[m]["confusion_matrix"] = conf[m].tolist()
        for s, acc in by_snr.items():
            rows.append({"method": m, "snr": s, "acc": acc, "correct": stats[m][s][0], "total": stats[m][s][1]})
    route_rows = []
    for s, counts in sorted(route_dist.items()):
        total = sum(counts)
        route_rows.append({"snr": s, "very_low": counts[0], "low": counts[1], "edge": counts[2], "reliable": counts[3], "total": total})
    write_csv(run_dir / "metrics_by_snr.csv", rows)
    write_csv(run_dir / "router_distribution_by_snr.csv", route_rows)
    write_json(run_dir / "metrics_overall.json", overall)
    return overall


@torch.no_grad()
def evaluate_single(model, loader, device: torch.device, mode: str = "teacher") -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits = model(x)["logits"] if mode == "teacher" else model(x)
        correct += int(logits.argmax(dim=-1).eq(y).sum())
        total += int(y.numel())
    return correct / max(1, total)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, workers: int, device: torch.device, drop_last: bool = False) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def write_flexible_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def train_teacher(args, train_ds, val_ds, device: torch.device, run_dir: Path, num_classes: int):
    model = SpectralTemporalTeacher(
        num_classes=num_classes,
        dim=args.teacher_dim,
        z_dim=args.z_dim,
        depth=args.teacher_depth,
        heads=args.teacher_heads,
        mlp_ratio=4.0,
        dropout=0.05,
        max_tokens=128,
    ).to(device)
    train_loader = make_loader(train_ds, args.teacher_batch_size, True, args.num_workers, device)
    val_loader = make_loader(val_ds, args.eval_batch_size, False, args.num_workers, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.teacher_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.teacher_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.0
    stale = 0
    log = []
    for epoch in range(1, args.teacher_epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
        total = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(x)["logits"], y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss.detach().cpu()) * y.numel()
            total += int(y.numel())
        scheduler.step()
        val_acc = evaluate_single(model, val_loader, device, mode="teacher")
        log.append({"stage": "teacher", "epoch": epoch, "loss": loss_sum / max(1, total), "val_acc": val_acc, "epoch_time_sec": time.time() - t0})
        print(f"teacher epoch={epoch} loss={log[-1]['loss']:.4f} val_acc={val_acc:.4f} best={best:.4f}", flush=True)
        if val_acc > best:
            best = val_acc
            stale = 0
            torch.save({"model": model.state_dict(), "best_val_acc": best, "epoch": epoch}, run_dir / "teacher_best.pt")
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(torch.load(run_dir / "teacher_best.pt", map_location=device)["model"])
    return model, log


def train_router(args, teacher, train_ds, val_ds, device: torch.device, run_dir: Path):
    for p in teacher.parameters():
        p.requires_grad = False
    router = ReliabilityRouter(args.z_dim).to(device)
    train_loader = make_loader(train_ds, args.router_batch_size, True, args.num_workers, device)
    val_loader = make_loader(val_ds, args.eval_batch_size, False, args.num_workers, device)
    opt = torch.optim.AdamW(router.parameters(), lr=args.router_lr, weight_decay=0.01)
    best = -1.0
    log = []
    for epoch in range(1, args.router_epochs + 1):
        t0 = time.time()
        router.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            snr = batch["snr"].to(device, non_blocking=True)
            y = route_from_snr(snr)
            with torch.no_grad():
                z = teacher.encode(x)
            opt.zero_grad(set_to_none=True)
            logits = router(z)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            opt.step()
            pred = logits.argmax(dim=-1)
            correct += int(pred.eq(y).sum())
            total += int(y.numel())
            loss_sum += float(loss.detach().cpu()) * y.numel()
        router.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device, non_blocking=True)
                snr = batch["snr"].to(device, non_blocking=True)
                y = route_from_snr(snr)
                pred = router(teacher.encode(x)).argmax(dim=-1)
                val_correct += int(pred.eq(y).sum())
                val_total += int(y.numel())
        val_acc = val_correct / max(1, val_total)
        log.append({"stage": "router", "epoch": epoch, "loss": loss_sum / max(1, total), "train_acc": correct / max(1, total), "val_acc": val_acc, "epoch_time_sec": time.time() - t0})
        print(f"router epoch={epoch} loss={log[-1]['loss']:.4f} val_acc={val_acc:.4f}", flush=True)
        if val_acc > best:
            best = val_acc
            torch.save({"router": router.state_dict(), "best_val_acc": best, "epoch": epoch}, run_dir / "router_best.pt")
    router.load_state_dict(torch.load(run_dir / "router_best.pt", map_location=device)["router"])
    return router, log


def train_expert(args, train_ds, val_ds, device: torch.device, run_dir: Path, num_classes: int):
    expert = RawLowSNRExpert(num_classes, width=args.expert_width, hidden=args.expert_hidden, dropout=0.20, branch_mode="time_only").to(device)
    train_loader = make_loader(train_ds, args.expert_batch_size, True, args.num_workers, device, drop_last=True)
    val_loader = make_loader(val_ds, args.eval_batch_size, False, args.num_workers, device)
    opt = torch.optim.AdamW(expert.parameters(), lr=args.expert_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.expert_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.0
    stale = 0
    log = []
    for epoch in range(1, args.expert_epochs + 1):
        t0 = time.time()
        expert.train()
        loss_sum = 0.0
        total = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = F.cross_entropy(expert(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss.detach().cpu()) * y.numel()
            total += int(y.numel())
        scheduler.step()
        val_acc = evaluate_single(expert, val_loader, device, mode="expert")
        log.append({"stage": "expert", "epoch": epoch, "loss": loss_sum / max(1, total), "val_acc": val_acc, "epoch_time_sec": time.time() - t0})
        print(f"expert epoch={epoch} loss={log[-1]['loss']:.4f} val_acc={val_acc:.4f} best={best:.4f}", flush=True)
        if val_acc > best:
            best = val_acc
            stale = 0
            torch.save({"expert": expert.state_dict(), "best_val_acc": best, "epoch": epoch}, run_dir / "expert_best.pt")
        else:
            stale += 1
        if stale >= args.patience:
            break
    expert.load_state_dict(torch.load(run_dir / "expert_best.pt", map_location=device)["expert"])
    return expert, log


def dataset_sizes(root: str | Path, dataset: str, seed: int) -> dict:
    out = {}
    for split in ["train", "val", "test"]:
        ds = RML2016Dataset(dataset, root, split, seed)
        by_snr = defaultdict(int)
        by_label = defaultdict(int)
        for r in ds.rows:
            by_snr[int(r["snr"])] += 1
            by_label[str(r["mod"])] += 1
        out[split] = {"n": len(ds), "by_snr": dict(sorted(by_snr.items())), "by_label": dict(sorted(by_label.items()))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["10a", "10b"], required=True)
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--output-root", default=str(ROOT / "runs"))
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--teacher-epochs", type=int, default=40)
    ap.add_argument("--router-epochs", type=int, default=12)
    ap.add_argument("--expert-epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--teacher-batch-size", type=int, default=1024)
    ap.add_argument("--router-batch-size", type=int, default=2048)
    ap.add_argument("--expert-batch-size", type=int, default=1024)
    ap.add_argument("--eval-batch-size", type=int, default=2048)
    ap.add_argument("--teacher-lr", type=float, default=4e-4)
    ap.add_argument("--router-lr", type=float, default=5e-4)
    ap.add_argument("--expert-lr", type=float, default=3e-4)
    ap.add_argument("--teacher-dim", type=int, default=256)
    ap.add_argument("--z-dim", type=int, default=256)
    ap.add_argument("--teacher-depth", type=int, default=4)
    ap.add_argument("--teacher-heads", type=int, default=4)
    ap.add_argument("--expert-width", type=int, default=256)
    ap.add_argument("--expert-hidden", type=int, default=512)
    ap.add_argument("--max-per-snr-class", type=int)
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"_{args.run_tag}" if args.run_tag else ""
    run_dir = make_run_dir(args.output_root, f"rml2016_{args.dataset}_tgfm128{tag}")
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    num_classes = len(MODS_10A if args.dataset == "10a" else MODS_10B)

    train_teacher_ds = RML2016Dataset(args.dataset, args.data_root, "train", args.seed, snr_min=0, max_per_snr_class=args.max_per_snr_class)
    val_teacher_ds = RML2016Dataset(args.dataset, args.data_root, "val", args.seed, snr_min=0, max_per_snr_class=args.max_per_snr_class)
    train_router_ds = RML2016Dataset(args.dataset, args.data_root, "train", args.seed, max_per_snr_class=args.max_per_snr_class)
    val_router_ds = RML2016Dataset(args.dataset, args.data_root, "val", args.seed, max_per_snr_class=args.max_per_snr_class)
    train_expert_ds = RML2016Dataset(args.dataset, args.data_root, "train", args.seed, snr_values=LOW_SNRS, max_per_snr_class=args.max_per_snr_class)
    val_expert_ds = RML2016Dataset(args.dataset, args.data_root, "val", args.seed, snr_values=LOW_SNRS, max_per_snr_class=args.max_per_snr_class)
    test_ds = RML2016Dataset(args.dataset, args.data_root, "test", args.seed, max_per_snr_class=args.max_per_snr_class)
    split_summary = {
        "teacher_train": len(train_teacher_ds),
        "teacher_val": len(val_teacher_ds),
        "router_train": len(train_router_ds),
        "router_val": len(val_router_ds),
        "expert_train": len(train_expert_ds),
        "expert_val": len(val_expert_ds),
        "test": len(test_ds),
        "num_classes": num_classes,
    }
    write_json(run_dir / "split_summary.json", split_summary)
    print(f"split_summary={split_summary}", flush=True)

    teacher, teacher_log = train_teacher(args, train_teacher_ds, val_teacher_ds, device, run_dir, num_classes)
    router, router_log = train_router(args, teacher, train_router_ds, val_router_ds, device, run_dir)
    expert, expert_log = train_expert(args, train_expert_ds, val_expert_ds, device, run_dir, num_classes)
    write_flexible_csv(run_dir / "train_log.csv", teacher_log + router_log + expert_log)

    test_loader = make_loader(test_ds, args.eval_batch_size, False, args.num_workers, device)
    overall = evaluate_all(teacher, router, expert, test_loader, device, run_dir, num_classes)
    print(f"RML2016_TGFM128_OK run_dir={run_dir} summary={json.dumps({k: v for k, v in overall.items() if k != 'confusion_matrix'}, indent=2)[:2000]}", flush=True)


if __name__ == "__main__":
    main()
