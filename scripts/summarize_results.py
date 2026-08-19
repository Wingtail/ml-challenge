"""Summarize the four required reproduction runs."""

import json
import statistics
from pathlib import Path


SEEDS = (42, 43, 44, 45)
OUTPUT_ROOT = Path("outputs/reproduction")


def read_score(path, *keys):
    value = json.loads(path.read_text())
    for key in keys:
        value = value[key]
    return 100 * value


def summarize(values):
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
    }


def main():
    direct = []
    context = []
    shuffled = []
    for seed in SEEDS:
        direct.append(
            read_score(
                OUTPUT_ROOT / "direct" / f"seed_{seed}" / "result.json",
                "test",
                "macro_f1",
            )
        )
        context.append(
            read_score(
                OUTPUT_ROOT / "context" / f"seed_{seed}" / "result.json",
                "test",
                "macro_f1",
            )
        )
        shuffled.append(
            read_score(
                OUTPUT_ROOT / "context" / f"seed_{seed}" / "shuffle.json",
                "macro_f1_mean",
            )
        )

    summary = {
        "seeds": list(SEEDS),
        "direct_classifier": summarize(direct),
        "pretrained_llm_context": summarize(context),
        "shuffled_context_embeddings": summarize(shuffled),
    }
    output = OUTPUT_ROOT / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")

    for name, result in summary.items():
        if name != "seeds":
            print(f"{name}: {result['mean']:.2f} +/- {result['sample_std']:.2f}")
    print(f"Full-precision summary: {output}")


if __name__ == "__main__":
    main()
