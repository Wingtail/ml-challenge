"""Summarize supervised tuning and compare final four-seed results."""

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

from scipy.stats import ttest_rel


TUNING_ROOT = Path("outputs/supervised_tuning")
FINAL_ROOT = Path("outputs/supervised_final")
SSL_SUMMARY = Path("outputs/submission_compliant/summary.json")
OUTPUT = Path("outputs/supervised_comparison/summary.json")


def load_json(path):
    return json.loads(path.read_text())


def summarize_tuning():
    grouped = defaultdict(list)
    for path in TUNING_ROOT.glob("fold_*/*/*/result.json"):
        result = load_json(path)
        key = (
            result["architecture"],
            result["learning_rate"],
            result["batch_size"],
        )
        grouped[key].append(result)

    summaries = []
    for (architecture, learning_rate, batch_size), results in grouped.items():
        scores = [100 * result["validation"]["macro_f1"] for result in results]
        epochs = [result["best_epoch"] for result in results]
        summaries.append(
            {
                "architecture": architecture,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "fold_macro_f1": scores,
                "mean_macro_f1": mean(scores),
                "sample_std_macro_f1": stdev(scores),
                "best_epochs": epochs,
                "median_best_epoch": median(epochs),
            }
        )
    return sorted(
        summaries,
        key=lambda item: (item["architecture"], -item["mean_macro_f1"]),
    )


def final_scores(architecture):
    results = [
        load_json(path)
        for path in sorted((FINAL_ROOT / architecture).glob("*/result.json"))
    ]
    scores = [100 * result["test"]["macro_f1"] for result in results]
    return {
        "seeds": [result["seed"] for result in results],
        "values": scores,
        "mean": mean(scores),
        "sample_std": stdev(scores),
        "trainable_parameters": results[0]["trainable_parameters"],
        "learning_rate": results[0]["learning_rate"],
        "batch_size": results[0]["batch_size"],
        "epochs": results[0]["maximum_epochs"],
        "dropout": results[0]["dropout"],
        "weight_decay": results[0]["weight_decay"],
    }


def paired_difference(first, second):
    differences = [a - b for a, b in zip(first, second)]
    return {
        "differences": differences,
        "mean_difference": mean(differences),
        "paired_t_p_value": float(ttest_rel(first, second).pvalue),
    }


def main():
    tuning = summarize_tuning()
    fcn = final_scores("fcn")
    random_patchtst = final_scores("patchtst")
    ssl_patchtst = load_json(SSL_SUMMARY)["direct_macro_f1"]
    summary = {
        "selection_protocol": {
            "splitter": "StratifiedGroupKFold",
            "folds": 5,
            "selection_metric": "mean validation macro-F1",
            "epoch_rule": "median best epoch across the five folds",
            "test_used_for_selection": False,
        },
        "tuning": tuning,
        "final": {
            "fcn_from_scratch": fcn,
            "patchtst_from_scratch": random_patchtst,
            "patchtst_masked_pretrained": ssl_patchtst,
        },
        "comparisons": {
            "ssl_patchtst_minus_random_patchtst": paired_difference(
                ssl_patchtst["values"], random_patchtst["values"]
            ),
            "random_patchtst_minus_fcn": paired_difference(
                random_patchtst["values"], fcn["values"]
            ),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
