import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_rel

from challenge.analyze_embeddings import (
    classifier_metrics,
    context_representations,
    geometry_metrics,
    pair_effect_size,
)
from challenge.data import load_challenge_data
from challenge.models import ContinuousTokenClassifier
from challenge.train import load_context_state_dict, make_encoder, set_seed


SEEDS = (42, 43, 44, 45)
METHODS = (
    "masked",
    "masked_context_vcreg",
    "masked_aux",
    "masked_aux_vcreg_001",
    "masked_aux_vcreg_003",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/UCI HAR Dataset")
    parser.add_argument("--baseline-root", default="outputs/multiseed")
    parser.add_argument("--rank-root", default="outputs/rank_methods")
    parser.add_argument("--experiment-root", default="outputs/semantic_rank")
    parser.add_argument("--llm-model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--output", default="outputs/semantic_rank_geometry.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
        "values": values.tolist(),
    }


def run_root(args, method, seed):
    if method == "masked":
        root = args.baseline_root
    elif method == "masked_context_vcreg":
        root = args.rank_root
    else:
        root = args.experiment_root
    return Path(root) / method / f"seed_{seed}"


def main():
    args = parse_args()
    set_seed(42)
    data = load_challenge_data(args.data_root)
    x = data["x_test"]
    labels = data["y_test"]
    subjects = data["subjects_test"]
    results = {
        "protocol": {
            "seeds": SEEDS,
            "context_learning_rate": 3e-4,
            "context_batch_size": 128,
            "dropout": 0.1,
            "weight_decay": 0.0,
            "sensor_aux_weight": 0.2,
            "vcreg_weights": [0.001, 0.003, 0.01],
        },
        "runs": {},
        "summary": {},
    }

    model = ContinuousTokenClassifier.from_pretrained(
        make_encoder(0.1), args.llm_model, dtype=torch.bfloat16
    ).to(args.device)
    for seed in SEEDS:
        results["runs"][str(seed)] = {}
        for method in METHODS:
            root = run_root(args, method, seed)
            metrics = json.loads((root / "context.json").read_text())
            state = torch.load(root / "context.pt", map_location="cpu", weights_only=True)
            load_context_state_dict(model, state)
            model.eval()
            representations = context_representations(model, x, args.batch_size, args.device)
            run = {
                "test_macro_f1": metrics["test"]["macro_f1"],
                "test_per_class_f1": metrics["test"]["per_class_f1"],
                "validation_macro_f1": metrics["validation"]["macro_f1"],
                "shuffled_macro_f1": metrics["shuffled_test"]["macro_f1_mean"],
            }
            for stage, representation in representations.items():
                run[stage] = geometry_metrics(representation, labels, subjects)
                run[stage]["sitting_standing_effect_size"] = pair_effect_size(
                    representation, labels
                )
            if "sensor_head" in state:
                model.sensor_head.cpu()
                run["auxiliary_head"] = classifier_metrics(
                    model.sensor_head, representations["sensor_embedding"], labels
                )
                model.sensor_head.to(args.device)
            results["runs"][str(seed)][method] = run
            print(f"measured seed={seed} method={method}")

    fields = {
        "test_macro_f1": lambda run: run["test_macro_f1"],
        "validation_macro_f1": lambda run: run["validation_macro_f1"],
        "sensor_effective_rank": lambda run: run["sensor_embedding"]["effective_rank"],
        "projected_effective_rank": lambda run: run["projected_token"]["effective_rank"],
        "hidden_effective_rank": lambda run: run["final_hidden_state"]["effective_rank"],
        "hidden_top_1_variance": lambda run: run["final_hidden_state"][
            "top_1_variance_fraction"
        ],
        "sitting_standing_effect_size": lambda run: run["final_hidden_state"][
            "sitting_standing_effect_size"
        ],
    }
    results["summary"]["by_method"] = {}
    for method in METHODS:
        runs = [results["runs"][str(seed)][method] for seed in SEEDS]
        summary = {
            name: mean_std([getter(run) for run in runs])
            for name, getter in fields.items()
        }
        summary["test_per_class_f1"] = [
            mean_std([run["test_per_class_f1"][class_index] for run in runs])
            for class_index in range(6)
        ]
        if all("auxiliary_head" in run for run in runs):
            summary["auxiliary_head_macro_f1"] = mean_std(
                [run["auxiliary_head"]["macro_f1"] for run in runs]
            )
        results["summary"]["by_method"][method] = summary

    baseline = [
        results["runs"][str(seed)]["masked"]["test_macro_f1"] for seed in SEEDS
    ]
    results["summary"]["paired_difference_from_masked"] = {}
    for method in METHODS[1:]:
        differences = {}
        for name, getter in fields.items():
            values = [
                getter(results["runs"][str(seed)][method])
                - getter(results["runs"][str(seed)]["masked"])
                for seed in SEEDS
            ]
            differences[name] = mean_std(values)
        candidate = [
            results["runs"][str(seed)][method]["test_macro_f1"] for seed in SEEDS
        ]
        test = ttest_rel(candidate, baseline)
        differences["test_macro_f1"]["paired_t_statistic"] = float(test.statistic)
        differences["test_macro_f1"]["paired_t_p"] = float(test.pvalue)
        results["summary"]["paired_difference_from_masked"][method] = differences

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
