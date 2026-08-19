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
from challenge.analyze_group_cv import mean_std
from challenge.data import OFFICIAL_TEST_SUBJECTS, OFFICIAL_TRAIN_SUBJECTS, load_challenge_data
from challenge.models import ContinuousTokenClassifier
from challenge.train import load_context_state_dict, make_encoder, set_seed


SEEDS = (42, 43, 44, 45)
PROMPT_STYLES = ("semantic", "scrambled", "answer")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/UCI HAR Dataset")
    parser.add_argument("--root", default="outputs/group_cv")
    parser.add_argument("--llm-model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--output", default="outputs/group_cv/final_summary.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_root(root, prompt_style, seed):
    directory = "final" if prompt_style == "answer" else f"final_{prompt_style}"
    return (
        root
        / directory
        / "masked_aux_vcreg_003"
        / f"seed_{seed}"
    )


def main():
    args = parse_args()
    set_seed(42)
    root = Path(args.root)
    data = load_challenge_data(args.data_root, validation_subjects=())
    results = {
        "protocol": {
            "seeds": SEEDS,
            "training_subjects": OFFICIAL_TRAIN_SUBJECTS,
            "test_subjects": OFFICIAL_TEST_SUBJECTS,
            "encoder_learning_rate": 1e-4,
            "adapter_learning_rate": 1e-3,
            "epochs": 18,
            "weight_decay": 0.0,
            "prompt_styles": PROMPT_STYLES,
        },
        "runs": {},
        "summary": {},
    }

    for prompt_style in PROMPT_STYLES:
        model = ContinuousTokenClassifier.from_pretrained(
            make_encoder(0.1),
            args.llm_model,
            dtype=torch.bfloat16,
            prompt_style=prompt_style,
        ).to(args.device)
        for seed in SEEDS:
            path = run_root(root, prompt_style, seed)
            metrics = json.loads((path / "context.json").read_text())
            state = torch.load(path / "context.pt", map_location="cpu", weights_only=True)
            if metrics["training_subjects"] != list(OFFICIAL_TRAIN_SUBJECTS):
                raise RuntimeError(f"final training subjects are wrong: {path}")
            if metrics["validation_subjects"]:
                raise RuntimeError(f"final fit unexpectedly has validation subjects: {path}")
            if not torch.allclose(state["normalization_mean"], data["mean"]):
                raise RuntimeError(f"normalization mean mismatch: {path}")
            if not torch.allclose(state["normalization_std"], data["std"]):
                raise RuntimeError(f"normalization standard deviation mismatch: {path}")
            audit = metrics["gradient_audit"]
            if any(audit[name] <= 0 for name in ("sensor_token", "encoder", "projector", "head")):
                raise RuntimeError(f"missing gradient in {path}: {audit}")
            if audit["language_model"] != 0:
                raise RuntimeError(f"language model received a parameter gradient: {path}")

            load_context_state_dict(model, state)
            model.eval()
            representations = context_representations(
                model, data["x_test"], args.batch_size, args.device
            )
            final_geometry = geometry_metrics(
                representations["final_hidden_state"],
                data["y_test"],
                data["subjects_test"],
            )
            final_geometry["sitting_standing_effect_size"] = pair_effect_size(
                representations["final_hidden_state"], data["y_test"]
            )
            model.sensor_head.cpu()
            direct_sensor = classifier_metrics(
                model.sensor_head,
                representations["sensor_embedding"],
                data["y_test"],
            )
            model.sensor_head.to(args.device)
            results["runs"].setdefault(str(seed), {})[prompt_style] = {
                "test": metrics["test"],
                "shuffled_test": metrics["shuffled_test"],
                "direct_sensor_head": direct_sensor,
                "final_hidden_state": final_geometry,
                "gradient_audit": audit,
            }
            print(f"measured seed={seed} prompt={prompt_style}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for prompt_style in PROMPT_STYLES:
        runs = [results["runs"][str(seed)][prompt_style] for seed in SEEDS]
        results["summary"][prompt_style] = {
            "test_accuracy": mean_std([run["test"]["accuracy"] for run in runs]),
            "test_macro_f1": mean_std([run["test"]["macro_f1"] for run in runs]),
            "direct_sensor_macro_f1": mean_std(
                [run["direct_sensor_head"]["macro_f1"] for run in runs]
            ),
            "shuffled_macro_f1": mean_std(
                [run["shuffled_test"]["macro_f1_mean"] for run in runs]
            ),
            "hidden_effective_rank": mean_std(
                [run["final_hidden_state"]["effective_rank"] for run in runs]
            ),
            "sitting_standing_effect_size": mean_std(
                [
                    run["final_hidden_state"]["sitting_standing_effect_size"]
                    for run in runs
                ]
            ),
            "test_per_class_f1": [
                mean_std([run["test"]["per_class_f1"][class_index] for run in runs])
                for class_index in range(6)
            ],
        }

    for first, second in (
        ("semantic", "scrambled"),
        ("semantic", "answer"),
        ("scrambled", "answer"),
    ):
        first_f1 = results["summary"][first]["test_macro_f1"]["values"]
        second_f1 = results["summary"][second]["test_macro_f1"]["values"]
        differences = np.asarray(first_f1) - np.asarray(second_f1)
        paired = ttest_rel(first_f1, second_f1)
        results["summary"][f"paired_{first}_minus_{second}"] = {
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
