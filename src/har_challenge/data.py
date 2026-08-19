"""Load only the nine raw UCI-HAR inertial signals."""

from pathlib import Path

import numpy as np
import torch


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
TRAIN_SUBJECTS = (1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30)
TEST_SUBJECTS = (2, 4, 9, 10, 12, 13, 18, 20, 24)


def _read_split(root: Path, split: str):
    signal_root = root / split / "Inertial Signals"
    signals = [
        np.loadtxt(signal_root / f"{name}_{split}.txt", dtype=np.float32)
        for name in SIGNALS
    ]
    x = np.stack(signals, axis=1)
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(
        root / split / f"subject_{split}.txt", dtype=np.int64
    )
    if x.shape[1:] != (9, 128):
        raise ValueError(f"expected {split} shape [N, 9, 128], got {x.shape}")
    return x, y, subjects


def load_uci_har(root: str | Path, validation_subjects: tuple[int, ...] = ()):
    """Return normalized subject-disjoint tensors without engineered features."""
    root = Path(root)
    x_all, y_all, subject_all = _read_split(root, "train")
    x_test, y_test, subject_test = _read_split(root, "test")

    if tuple(np.unique(subject_all)) != TRAIN_SUBJECTS:
        raise ValueError("official training subjects do not match UCI-HAR")
    if tuple(np.unique(subject_test)) != TEST_SUBJECTS:
        raise ValueError("official test subjects do not match UCI-HAR")
    if set(validation_subjects) - set(TRAIN_SUBJECTS):
        raise ValueError("validation subjects must come from the training split")

    validation_mask = np.isin(subject_all, validation_subjects)
    training_mask = ~validation_mask
    if not training_mask.any():
        raise ValueError("validation split cannot consume every training subject")

    x_train = x_all[training_mask]
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = np.maximum(x_train.std(axis=(0, 2), keepdims=True), 1e-6)

    def tensor(values):
        return torch.from_numpy(((values - mean) / std).astype(np.float32))

    return {
        "x_train": tensor(x_train),
        "y_train": torch.from_numpy(y_all[training_mask]),
        "subjects_train": torch.from_numpy(subject_all[training_mask]),
        "x_validation": tensor(x_all[validation_mask]),
        "y_validation": torch.from_numpy(y_all[validation_mask]),
        "subjects_validation": torch.from_numpy(subject_all[validation_mask]),
        "x_test": tensor(x_test),
        "y_test": torch.from_numpy(y_test),
        "subjects_test": torch.from_numpy(subject_test),
        "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std),
    }
