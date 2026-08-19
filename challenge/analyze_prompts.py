import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_rel

from challenge.analyze_readouts import mean_std, measure
from challenge.data import load_challenge_data
from challenge.models import ContinuousTokenClassifier
from challenge.train import make_encoder, set_seed


SEEDS = (42, 43, 44, 45)
PROMPT_STYLES = ("semantic", "scrambled", "answer")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/UCI HAR Dataset")
    parser.add_argument("--semantic-root", default="outputs/semantic_rank/masked_aux_vcreg_003")
    parser.add_argument(
        "--scrambled-root",
        default="outputs/prompt_ablation/scrambled/masked_aux_vcreg_003",
    )
    parser.add_argument(
        "--answer-root",
        default="outputs/prompt_ablation/answer/masked_aux_vcreg_003",
    )
    parser.add_argument("--llm-model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--output", default="outputs/prompt_ablation_geometry.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    data = load_challenge_data(args.data_root)
    x = data["x_test"]
    labels = data["y_test"]
    subjects = data["subjects_test"]

    models = {
        prompt_style: ContinuousTokenClassifier.from_pretrained(
            make_encoder(0.1),
            args.llm_model,
            dtype=torch.bfloat16,
            prompt_style=prompt_style,
        ).to(args.device)
        for prompt_style in PROMPT_STYLES
    }
    roots = {
        "semantic": Path(args.semantic_root),
        "scrambled": Path(args.scrambled_root),
        "answer": Path(args.answer_root),
    }
    results = {
        "protocol": {
            "seeds": SEEDS,
            "sensor_aux_weight": 0.2,
            "context_vcreg_weight": 0.003,
            "prompt_token_count": int(
                len(models["semantic"].prefix_ids) + len(models["semantic"].suffix_ids)
            ),
            "sensor_position": int(len(models["semantic"].prefix_ids)),
            "same_token_multiset": bool(
                sorted(torch.cat((models["semantic"].prefix_ids, models["semantic"].suffix_ids)).tolist())
                == sorted(torch.cat((models["scrambled"].prefix_ids, models["scrambled"].suffix_ids)).tolist())
            ),
            "same_final_token": bool(
                models["semantic"].suffix_ids[-1]
                == models["scrambled"].suffix_ids[-1]
            ),
            "prompt_lengths": {
                prompt_style: int(
                    len(models[prompt_style].prefix_ids)
                    + len(models[prompt_style].suffix_ids)
                )
                for prompt_style in PROMPT_STYLES
            },
            "sensor_positions": {
                prompt_style: int(len(models[prompt_style].prefix_ids))
                for prompt_style in PROMPT_STYLES
            },
        },
        "runs": {},
        "summary": {},
    }

    for seed in SEEDS:
        results["runs"][str(seed)] = {}
        for prompt_style in PROMPT_STYLES:
            root = roots[prompt_style] / f"seed_{seed}"
            metrics = json.loads((root / "context.json").read_text())
            state = torch.load(root / "context.pt", map_location="cpu", weights_only=True)
            results["runs"][str(seed)][prompt_style] = measure(
                models[prompt_style],
                state,
                metrics,
                x,
                labels,
                subjects,
                args.batch_size,
                args.device,
            )
            print(f"measured seed={seed} prompt={prompt_style}")

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
    for prompt_style in PROMPT_STYLES:
        runs = [results["runs"][str(seed)][prompt_style] for seed in SEEDS]
        summary = {
            field: mean_std([getter(run) for run in runs])
            for field, getter in fields.items()
        }
        summary["test_per_class_f1"] = [
            mean_std([run["test_per_class_f1"][class_index] for run in runs])
            for class_index in range(6)
        ]
        results["summary"][prompt_style] = summary

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
