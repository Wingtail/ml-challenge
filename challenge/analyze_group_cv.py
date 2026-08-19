import argparse
import json
import re
from pathlib import Path

import numpy as np


CANDIDATE_PATTERN = re.compile(r"enc_(.+)_adapter_(.+)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="outputs/group_cv/folds.json")
    parser.add_argument("--tuning-root", default="outputs/group_cv/tuning")
    parser.add_argument("--output", default="outputs/group_cv/summary.json")
    return parser.parse_args()


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
        "values": values.tolist(),
    }


def macro_f1_from_confusion(confusion):
    confusion = np.asarray(confusion, dtype=np.float64)
    true_counts = confusion.sum(axis=1)
    predicted_counts = confusion.sum(axis=0)
    denominator = true_counts + predicted_counts
    per_class = np.divide(
        2.0 * np.diag(confusion),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(per_class.mean()), per_class.tolist()


def main():
    args = parse_args()
    fold_manifest = json.loads(Path(args.folds).read_text())
    tuning_root = Path(args.tuning_root)
    candidates = {}

    for candidate_path in sorted(tuning_root.glob("enc_*_adapter_*")):
        match = CANDIDATE_PATTERN.fullmatch(candidate_path.name)
        if match is None:
            continue
        encoder_lr, adapter_lr = match.groups()
        runs = []
        for fold in fold_manifest["folds"]:
            metrics_path = (
                candidate_path
                / f"fold_{fold['fold']}"
                / "masked_aux_vcreg_003"
                / "seed_42"
                / "context.json"
            )
            metrics = json.loads(metrics_path.read_text())
            if "test" in metrics:
                raise RuntimeError(f"test metrics leaked into tuning: {metrics_path}")
            if metrics["validation_subjects"] != fold["validation_subjects"]:
                raise RuntimeError(f"fold subject mismatch: {metrics_path}")
            audit = metrics["gradient_audit"]
            if any(audit[name] <= 0 for name in ("sensor_token", "encoder", "projector", "head")):
                raise RuntimeError(f"missing gradient in {metrics_path}: {audit}")
            if audit["language_model"] != 0 or metrics["trainable_parameters"]["language_model"] != 0:
                raise RuntimeError(f"language model was not frozen: {metrics_path}")
            runs.append(metrics)

        fold_f1 = [run["validation"]["macro_f1"] for run in runs]
        best_epochs = [run["best_epoch"] for run in runs]
        confusion = np.sum(
            [run["validation"]["confusion_matrix"] for run in runs], axis=0
        )
        pooled_macro_f1, pooled_per_class_f1 = macro_f1_from_confusion(confusion)
        candidates[candidate_path.name] = {
            "encoder_learning_rate": float(encoder_lr),
            "adapter_learning_rate": float(adapter_lr),
            "fold_macro_f1": mean_std(fold_f1),
            "pooled_out_of_fold_macro_f1": pooled_macro_f1,
            "pooled_out_of_fold_per_class_f1": pooled_per_class_f1,
            "best_epochs": best_epochs,
            "median_best_epoch": int(np.median(best_epochs)),
            "gradient_audits": [run["gradient_audit"] for run in runs],
        }

    selected_name = max(
        candidates, key=lambda name: candidates[name]["fold_macro_f1"]["mean"]
    )
    result = {
        "protocol": {
            "splitter": fold_manifest["splitter"],
            "fold_count": fold_manifest["fold_count"],
            "fold_seed": fold_manifest["seed"],
            "training_subjects": fold_manifest["official_train_subjects"],
            "untouched_test_subjects": fold_manifest["official_test_subjects"],
            "prompt_style": "answer",
            "selection_metric": "mean fold validation macro-F1",
        },
        "candidates": candidates,
        "selected": {"name": selected_name, **candidates[selected_name]},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"selected {selected_name}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
