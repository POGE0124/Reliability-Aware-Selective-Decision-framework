from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import RML2018_MODS, normalize_iq


def create_rml2018_split(h5_path: str | Path, split_path: str | Path, seed: int, train_ratio: float, val_ratio: float) -> Path:
    import h5py

    split_path = Path(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        labels = np.asarray(f["Y"]).argmax(axis=1).astype(np.int64)
        snrs = np.asarray(f["Z"][:, 0]).astype(np.int64)
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in sorted(np.unique(labels).tolist()):
        for snr in sorted(np.unique(snrs[labels == label]).tolist()):
            idx = np.flatnonzero((labels == label) & (snrs == snr)).astype(np.int64)
            rng.shuffle(idx)
            n_train = int(round(idx.size * train_ratio))
            n_val = int(round(idx.size * val_ratio))
            n_train = min(max(1, n_train), idx.size - 2)
            n_val = min(max(1, n_val), idx.size - n_train - 1)
            train_parts.append(idx[:n_train])
            val_parts.append(idx[n_train : n_train + n_val])
            test_parts.append(idx[n_train + n_val :])
    np.savez_compressed(
        split_path,
        train=np.concatenate(train_parts),
        val=np.concatenate(val_parts),
        test=np.concatenate(test_parts),
        seed=np.asarray([seed], dtype=np.int64),
    )
    return split_path


class RML2018Dataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        split_path: str | Path,
        split: str,
        signal_length: int,
        snr_min: int | None = None,
        snr_values: list[int] | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.signal_length = int(signal_length)
        with np.load(split_path) as data:
            indices = data[split].astype(np.int64)
        import h5py

        with h5py.File(self.h5_path, "r") as f:
            all_snrs = np.asarray(f["Z"][:, 0]).astype(np.int64)
            snrs = all_snrs[indices]
        keep = np.ones(indices.shape[0], dtype=bool)
        if snr_min is not None:
            keep &= snrs >= int(snr_min)
        if snr_values is not None:
            keep &= np.isin(snrs, np.asarray(snr_values, dtype=np.int64))
        self.indices = indices[keep]
        if max_samples is not None:
            self.indices = self.indices[: int(max_samples)]
        self._h5 = None

    def _file(self):
        if self._h5 is None:
            import h5py

            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, idx: int) -> dict:
        h5_idx = int(self.indices[idx])
        f = self._file()
        x = np.asarray(f["X"][h5_idx, : self.signal_length, :], dtype=np.float32).T
        label = int(np.asarray(f["Y"][h5_idx]).argmax())
        snr = int(np.asarray(f["Z"][h5_idx, 0]))
        return {
            "x": torch.from_numpy(normalize_iq(x).copy()),
            "label": torch.tensor(label, dtype=torch.long),
            "snr": torch.tensor(float(snr), dtype=torch.float32),
            "sample_id": f"RML2018|{label}|{snr}|{h5_idx}",
            "modulation": RML2018_MODS[label],
            "h5_idx": torch.tensor(h5_idx, dtype=torch.long),
        }


class FeatureShardDataset(Dataset):
    def __init__(self, cache_dir: str | Path, split: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.paths = sorted((self.cache_dir / split).glob("shard_*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No feature shards in {self.cache_dir / split}")
        self.sizes: list[int] = []
        self.offsets: list[int] = []
        total = 0
        for p in self.paths:
            meta = torch.load(p, map_location="cpu", weights_only=False)
            n = int(meta["label"].shape[0])
            self.offsets.append(total)
            self.sizes.append(n)
            total += n
        self.total = total
        self._cache_idx = -1
        self._cache = None

    def __len__(self) -> int:
        return self.total

    def _load_shard(self, shard_idx: int) -> dict:
        if self._cache_idx != shard_idx:
            self._cache = torch.load(self.paths[shard_idx], map_location="cpu", weights_only=False)
            self._cache_idx = shard_idx
        return self._cache

    def __getitem__(self, idx: int) -> dict:
        shard_idx = int(np.searchsorted(np.asarray(self.offsets[1:]), idx, side="right"))
        local = idx - self.offsets[shard_idx]
        shard = self._load_shard(shard_idx)
        return {
            "z_noisy": shard["z_noisy"][local].float(),
            "z_high": shard["z_high"][local].float(),
            "label": shard["label"][local].long(),
            "target_snr": shard["target_snr"][local].float(),
        }


TECH_RE = re.compile(r"^(?P<tech>[^_]+).*?_g(?P<gain>\d+)_.*?_f(?P<freq>\d+MHz)_r(?P<run>\d+)\.bin$")
TECH_MAP = {"wf": 0, "wf10Msps": 0, "lte": 1, "lte10Msps": 1, "dvbt": 2, "dvbt10Msps": 2}


class TechRegDataset(Dataset):
    def __init__(self, root: str | Path, locations: list[str], signal_length: int, stride: int = 1024, max_windows_per_file: int | None = None) -> None:
        self.root = Path(root)
        self.signal_length = int(signal_length)
        self.rows: list[tuple[Path, int, int]] = []
        for loc in locations:
            for path in sorted((self.root / loc).glob("*.bin")):
                m = TECH_RE.match(path.name)
                if not m or m.group("tech") not in TECH_MAP:
                    continue
                label = TECH_MAP[m.group("tech")]
                count = path.stat().st_size // 8
                n_windows = max(0, (count - self.signal_length) // stride + 1)
                if max_windows_per_file is not None:
                    n_windows = min(n_windows, int(max_windows_per_file))
                for w in range(n_windows):
                    self.rows.append((path, label, w * stride))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        path, label, offset = self.rows[idx]
        raw = np.memmap(path, dtype=np.complex64, mode="r")
        seg = np.asarray(raw[offset : offset + self.signal_length])
        x = np.stack([seg.real, seg.imag], axis=0).astype(np.float32)
        return {
            "x": torch.from_numpy(normalize_iq(x).copy()),
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(path),
        }
