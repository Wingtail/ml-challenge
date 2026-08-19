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

# This is a whole-subject split drawn only from the publisher's training partition.
DEFAULT_VALIDATION_SUBJECTS = (1, 3, 15, 25, 27)
OFFICIAL_TRAIN_SUBJECTS = (1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30)
OFFICIAL_TEST_SUBJECTS = (2, 4, 9, 10, 12, 13, 18, 20, 24)
OFFICIAL_TRAIN_WINDOWS = 7_352
OFFICIAL_TEST_WINDOWS = 2_947


def _load_split(root: Path, split: str):
    signal_dir = root / split / "Inertial Signals"
    signals = [
        np.loadtxt(signal_dir / f"{name}_{split}.txt", dtype=np.float32)
        for name in SIGNALS
    ]
    x = np.stack(signals, axis=1)
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(root / split / f"subject_{split}.txt", dtype=np.int64)
    if x.ndim != 3 or x.shape[1:] != (9, 128):
        raise ValueError(f"unexpected {split} signal shape: {x.shape}")
    if len(x) != len(y) or len(x) != len(subjects):
        raise ValueError(f"misaligned {split} signals, labels, and subjects")
    return x, y, subjects


def load_challenge_data(
    root: str | Path,
    validation_subjects: tuple[int, ...] = DEFAULT_VALIDATION_SUBJECTS,
):
    """Load only the nine raw 128-step signals with subject-disjoint splits.

    Normalization statistics are fitted on the optimization subjects. Validation
    and official test windows have no influence on preprocessing.
    """
    root = Path(root)
    x_official_train, y_official_train, subjects_official_train = _load_split(
        root, "train"
    )
    x_test, y_test, subjects_test = _load_split(root, "test")

    train_subject_ids = tuple(np.unique(subjects_official_train).tolist())
    test_subject_ids = tuple(np.unique(subjects_test).tolist())
    if len(x_official_train) != OFFICIAL_TRAIN_WINDOWS:
        raise ValueError(f"unexpected official training windows: {len(x_official_train)}")
    if len(x_test) != OFFICIAL_TEST_WINDOWS:
        raise ValueError(f"unexpected official test windows: {len(x_test)}")
    if train_subject_ids != OFFICIAL_TRAIN_SUBJECTS:
        raise ValueError(f"unexpected official training subjects: {train_subject_ids}")
    if test_subject_ids != OFFICIAL_TEST_SUBJECTS:
        raise ValueError(f"unexpected official test subjects: {test_subject_ids}")
    if len(set(validation_subjects)) != len(validation_subjects):
        raise ValueError("validation subjects contain duplicates")
    unknown_validation = set(validation_subjects) - set(OFFICIAL_TRAIN_SUBJECTS)
    if unknown_validation:
        raise ValueError(
            f"validation subjects are not in the official training split: "
            f"{sorted(unknown_validation)}"
        )

    validation_mask = np.isin(subjects_official_train, validation_subjects)
    if validation_mask.all():
        raise ValueError("validation subjects consume the entire official train split")
    if np.intersect1d(subjects_official_train, subjects_test).size:
        raise ValueError("official train and test subjects overlap")

    train_mask = ~validation_mask
    x_train = x_official_train[train_mask]
    y_train = y_official_train[train_mask]
    subjects_train = subjects_official_train[train_mask]
    x_validation = x_official_train[validation_mask]
    y_validation = y_official_train[validation_mask]
    subjects_validation = subjects_official_train[validation_mask]

    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = np.maximum(x_train.std(axis=(0, 2), keepdims=True), 1e-6)

    def normalized_tensor(x):
        return torch.from_numpy(((x - mean) / std).astype(np.float32))

    return {
        "x_train": normalized_tensor(x_train),
        "y_train": torch.from_numpy(y_train),
        "subjects_train": torch.from_numpy(subjects_train),
        "x_validation": normalized_tensor(x_validation),
        "y_validation": torch.from_numpy(y_validation),
        "subjects_validation": torch.from_numpy(subjects_validation),
        "x_test": normalized_tensor(x_test),
        "y_test": torch.from_numpy(y_test),
        "subjects_test": torch.from_numpy(subjects_test),
        "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std),
    }
