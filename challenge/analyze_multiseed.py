import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from challenge.analyze_embeddings import (
    context_representations,
    encode_all,
    geometry_metrics,
    load_pretrained_encoder,
    pair_effect_size,
)
from challenge.data import load_challenge_data
from challenge.models import ContinuousTokenClassifier
from challenge.train import load_context_state_dict, make_encoder, set_seed


METHODS = ("random", "masked", "drop_patch")
SEEDS = (42, 43, 44, 45)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/UCI HAR Dataset")
    parser.add_argument("--checkpoint-root", default="outputs/multiseed")
    parser.add_argument("--llm-model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--output", default="outputs/multiseed_geometry.json")
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


def correlation(x, y):
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "samples": len(x),
    }


def main():
    args = parse_args()
    set_seed(42)
    root = Path(args.checkpoint_root)
    data = load_challenge_data(args.data_root)
    x = data["x_test"]
    labels = data["y_test"]
    subjects = data["subjects_test"]
    results = {"protocol": {}, "runs": {}, "summary": {}}
    results["protocol"] = {
        "seeds": SEEDS,
        "pretrain_learning_rate": 1e-3,
        "pretrain_batch_size": 256,
        "context_learning_rate": 3e-4,
        "context_batch_size": 128,
        "dropout": 0.1,
        "weight_decay": 0.0,
    }

    model = ContinuousTokenClassifier.from_pretrained(
        make_encoder(0.1), args.llm_model, dtype=torch.bfloat16
    ).to(args.device)
    for seed in SEEDS:
        results["runs"][str(seed)] = {}
        for method in METHODS:
            run_root = root / method / f"seed_{seed}"
            metrics = json.loads((run_root / "context.json").read_text())
            state = torch.load(run_root / "context.pt", map_location="cpu", weights_only=True)
            load_context_state_dict(model, state)
            model.eval()
            representations = context_representations(
                model, x, args.batch_size, args.device
            )
            run = {
                "test_macro_f1": metrics["test"]["macro_f1"],
                "shuffled_macro_f1": metrics["shuffled_test"]["macro_f1_mean"],
            }
            for stage, representation in representations.items():
                run[stage] = geometry_metrics(representation, labels, subjects)
                run[stage]["sitting_standing_effect_size"] = pair_effect_size(
                    representation, labels
                )
            if method != "random":
                pretrained = load_pretrained_encoder(
                    run_root / "pretrained.pt", args.device
                )
                embedding = encode_all(pretrained, x, args.batch_size, args.device)
                run["pretrained_embedding"] = geometry_metrics(
                    embedding, labels, subjects
                )
            results["runs"][str(seed)][method] = run
            print(f"measured seed={seed} method={method}")

    fields = {
        "test_macro_f1": lambda run: run["test_macro_f1"],
        "projected_effective_rank": lambda run: run["projected_token"]["effective_rank"],
        "projected_top_1_variance": lambda run: run["projected_token"]["top_1_variance_fraction"],
        "projected_mean_cosine": lambda run: run["projected_token"]["pairwise_cosine_mean"],
        "hidden_effective_rank": lambda run: run["final_hidden_state"]["effective_rank"],
        "hidden_top_1_variance": lambda run: run["final_hidden_state"]["top_1_variance_fraction"],
        "hidden_mean_cosine": lambda run: run["final_hidden_state"]["pairwise_cosine_mean"],
        "sitting_standing_effect_size": lambda run: run["final_hidden_state"][
            "sitting_standing_effect_size"
        ],
    }
    results["summary"]["by_method"] = {}
    for method in METHODS:
        method_runs = [results["runs"][str(seed)][method] for seed in SEEDS]
        results["summary"]["by_method"][method] = {
            name: mean_std([getter(run) for run in method_runs])
            for name, getter in fields.items()
        }

    masked = [results["runs"][str(seed)]["masked"] for seed in SEEDS]
    dropped = [results["runs"][str(seed)]["drop_patch"] for seed in SEEDS]
    results["summary"]["paired_masked_minus_drop_patch"] = {
        name: mean_std(
            [getter(first) - getter(second) for first, second in zip(masked, dropped)]
        )
        for name, getter in fields.items()
    }

    all_runs = [
        results["runs"][str(seed)][method] for seed in SEEDS for method in METHODS
    ]
    ssl_runs = [
        results["runs"][str(seed)][method]
        for seed in SEEDS
        for method in ("masked", "drop_patch")
    ]
    results["summary"]["correlation_with_test_f1"] = {}
    for population_name, population in (("all_12", all_runs), ("ssl_only_8", ssl_runs)):
        target = [run["test_macro_f1"] for run in population]
        results["summary"]["correlation_with_test_f1"][population_name] = {
            name: correlation([getter(run) for run in population], target)
            for name, getter in fields.items()
            if name != "test_macro_f1"
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
