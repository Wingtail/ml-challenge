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
from challenge.models import ContinuousTokenClassifier, MLPContextClassifier
from challenge.train import load_context_state_dict, make_encoder, set_seed


SEEDS = (42, 43, 44, 45)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/UCI HAR Dataset")
    parser.add_argument("--llm-root", default="outputs/semantic_rank")
    parser.add_argument("--mlp-root", default="outputs/readout_control")
    parser.add_argument("--llm-model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--output", default="outputs/readout_control_geometry.json")
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


def trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def measure(model, state, metrics, x, labels, subjects, batch_size, device):
    load_context_state_dict(model, state)
    model.eval()
    representations = context_representations(model, x, batch_size, device)
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
    model.sensor_head.cpu()
    run["direct_sensor_head"] = classifier_metrics(
        model.sensor_head, representations["sensor_embedding"], labels
    )
    model.sensor_head.to(device)
    return run


def main():
    args = parse_args()
    set_seed(42)
    data = load_challenge_data(args.data_root)
    x = data["x_test"]
    labels = data["y_test"]
    subjects = data["subjects_test"]

    llm = ContinuousTokenClassifier.from_pretrained(
        make_encoder(0.1), args.llm_model, dtype=torch.bfloat16
    ).to(args.device)
    mlp = MLPContextClassifier(make_encoder(0.1)).to(args.device)
    results = {
        "protocol": {
            "seeds": SEEDS,
            "sensor_aux_weight": 0.2,
            "context_vcreg_weight": 0.003,
            "llm_trainable_parameters": trainable_parameters(llm),
            "mlp_trainable_parameters": trainable_parameters(mlp),
        },
        "runs": {},
        "summary": {},
    }

    roots = {
        "frozen_llm": Path(args.llm_root) / "masked_aux_vcreg_003",
        "matched_mlp": Path(args.mlp_root) / "masked_mlp_aux_vcreg_003",
    }
    models = {"frozen_llm": llm, "matched_mlp": mlp}
    for seed in SEEDS:
        results["runs"][str(seed)] = {}
        for name in ("frozen_llm", "matched_mlp"):
            root = roots[name] / f"seed_{seed}"
            metrics = json.loads((root / "context.json").read_text())
            state = torch.load(root / "context.pt", map_location="cpu", weights_only=True)
            results["runs"][str(seed)][name] = measure(
                models[name],
                state,
                metrics,
                x,
                labels,
                subjects,
                args.batch_size,
                args.device,
            )
            print(f"measured seed={seed} readout={name}")

    fields = {
        "test_macro_f1": lambda run: run["test_macro_f1"],
        "validation_macro_f1": lambda run: run["validation_macro_f1"],
        "direct_sensor_macro_f1": lambda run: run["direct_sensor_head"]["macro_f1"],
        "sensor_effective_rank": lambda run: run["sensor_embedding"]["effective_rank"],
        "projected_effective_rank": lambda run: run["projected_token"]["effective_rank"],
        "hidden_effective_rank": lambda run: run["final_hidden_state"]["effective_rank"],
        "sitting_standing_effect_size": lambda run: run["final_hidden_state"][
            "sitting_standing_effect_size"
        ],
    }
    results["summary"]["by_readout"] = {}
    for name in ("frozen_llm", "matched_mlp"):
        runs = [results["runs"][str(seed)][name] for seed in SEEDS]
        summary = {
            field: mean_std([getter(run) for run in runs])
            for field, getter in fields.items()
        }
        summary["test_per_class_f1"] = [
            mean_std([run["test_per_class_f1"][class_index] for run in runs])
            for class_index in range(6)
        ]
        results["summary"]["by_readout"][name] = summary

    llm_f1 = [
        results["runs"][str(seed)]["frozen_llm"]["test_macro_f1"] for seed in SEEDS
    ]
    mlp_f1 = [
        results["runs"][str(seed)]["matched_mlp"]["test_macro_f1"] for seed in SEEDS
    ]
    differences = [first - second for first, second in zip(llm_f1, mlp_f1)]
    paired = ttest_rel(llm_f1, mlp_f1)
    results["summary"]["paired_llm_minus_mlp"] = {
        **mean_std(differences),
        "paired_t_statistic": float(paired.statistic),
        "paired_t_p": float(paired.pvalue),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
