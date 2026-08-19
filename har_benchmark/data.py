from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset


SIGNALS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)


def _load_split(root: Path, split: str):
    signal_dir = root / split / "Inertial Signals"
    signals = [
        np.loadtxt(signal_dir / f"{name}_{split}.txt", dtype=np.float32)
        for name in SIGNALS
    ]
    x = np.stack(signals, axis=1)
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(
        root / split / f"subject_{split}.txt", dtype=np.int64
    )
    return x, y, subjects


def load_uci_har(root: str | Path):
    """Load canonical 128x9 windows and normalize with training statistics only."""
    root = Path(root)
    x_train, y_train, subjects_train = _load_split(root, "train")
    x_test, y_test, subjects_test = _load_split(root, "test")

    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-6)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    return {
        "x_train": torch.from_numpy(x_train),
        "y_train": torch.from_numpy(y_train),
        "subjects_train": torch.from_numpy(subjects_train),
        "x_test": torch.from_numpy(x_test),
        "y_test": torch.from_numpy(y_test),
        "subjects_test": torch.from_numpy(subjects_test),
        "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std),
    }


def make_dataset(x: torch.Tensor, y: torch.Tensor):
    return TensorDataset(x, y)


class AdjacentPairDataset(Dataset):
    """Pairs each window with a nearby window from the same subject."""

    def __init__(
        self,
        x: torch.Tensor,
        subjects: torch.Tensor,
        clip_size: int = 10,
    ):
        self.x = x
        self.neighbors: list[torch.Tensor] = []
        subject_to_indices = {
            int(subject): torch.where(subjects == subject)[0]
            for subject in torch.unique(subjects)
        }
        positions = {}
        for indices in subject_to_indices.values():
            for position, index in enumerate(indices.tolist()):
                positions[index] = (indices, position)

        for index in range(len(x)):
            indices, position = positions[index]
            start = max(0, position - clip_size + 1)
            stop = min(len(indices), position + clip_size)
            candidates = indices[start:stop]
            candidates = candidates[candidates != index]
            if len(candidates) == 0:
                candidates = torch.tensor([index])
            self.neighbors.append(candidates)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        candidates = self.neighbors[index]
        neighbor = candidates[torch.randint(len(candidates), (1,)).item()]
        return self.x[index], self.x[neighbor]


def stratified_indices(y: torch.Tensor, fraction: float, seed: int):
    if not 0 < fraction <= 1:
        raise ValueError("label fraction must be in (0, 1]")
    if fraction == 1:
        return torch.arange(len(y))

    generator = torch.Generator().manual_seed(seed)
    selected = []
    for label in torch.unique(y):
        indices = torch.where(y == label)[0]
        count = max(1, round(len(indices) * fraction))
        selected.append(indices[torch.randperm(len(indices), generator=generator)[:count]])
    return torch.cat(selected)

