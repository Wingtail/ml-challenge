import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from challenge.data import (
    OFFICIAL_TEST_SUBJECTS,
    OFFICIAL_TRAIN_SUBJECTS,
    _load_split,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/UCI HAR Dataset")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/group_cv/folds.json")
    return parser.parse_args()


def class_counts(labels):
    return np.bincount(labels, minlength=6).tolist()


def main():
    args = parse_args()
    root = Path(args.data_root)
    _, train_labels, train_subjects = _load_split(root, "train")
    _, test_labels, test_subjects = _load_split(root, "test")
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )

    folds = []
    seen_validation_subjects = []
    for fold_index, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(train_labels)), train_labels, train_subjects),
        start=1,
    ):
        optimization_subjects = sorted(np.unique(train_subjects[train_indices]).tolist())
        validation_subjects = sorted(
            np.unique(train_subjects[validation_indices]).tolist()
        )
        if set(optimization_subjects) & set(validation_subjects):
            raise RuntimeError("optimization and validation subjects overlap")
        if set(validation_subjects) & set(OFFICIAL_TEST_SUBJECTS):
            raise RuntimeError("validation and official test subjects overlap")
        if np.count_nonzero(np.bincount(train_labels[validation_indices], minlength=6)) != 6:
            raise RuntimeError("a validation fold is missing an activity")
        seen_validation_subjects.extend(validation_subjects)
        folds.append(
            {
                "fold": fold_index,
                "optimization_subjects": optimization_subjects,
                "validation_subjects": validation_subjects,
                "optimization_windows": int(len(train_indices)),
                "validation_windows": int(len(validation_indices)),
                "optimization_class_counts": class_counts(train_labels[train_indices]),
                "validation_class_counts": class_counts(train_labels[validation_indices]),
            }
        )

    if sorted(seen_validation_subjects) != list(OFFICIAL_TRAIN_SUBJECTS):
        raise RuntimeError("each official training subject must validate exactly once")
    if tuple(np.unique(test_subjects).tolist()) != OFFICIAL_TEST_SUBJECTS:
        raise RuntimeError("official test subject IDs changed")

    result = {
        "splitter": "StratifiedGroupKFold",
        "fold_count": args.folds,
        "seed": args.seed,
        "official_train_subjects": OFFICIAL_TRAIN_SUBJECTS,
        "official_test_subjects": OFFICIAL_TEST_SUBJECTS,
        "official_train_windows": int(len(train_labels)),
        "official_test_windows": int(len(test_labels)),
        "official_train_class_counts": class_counts(train_labels),
        "official_test_class_counts": class_counts(test_labels),
        "folds": folds,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
