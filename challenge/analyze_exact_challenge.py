import json
from pathlib import Path

import numpy as np
from scipy import stats


SEEDS = (42, 43, 44, 45)
MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def load_llm_runs(backend):
    runs = []
    for seed in SEEDS:
        path = (
            Path("outputs/challenge_exact")
            / backend
            / "masked_aux_vcreg_003"
            / f"seed_{seed}"
            / "context.json"
        )
        result = json.loads(path.read_text())
        assert result["seed"] == seed
        assert result["context_backend"] == backend
        assert result["prompt_style"] == "challenge"
        assert result["best_epoch"] == 18
        assert result["validation_subjects"] == []
        assert len(result["training_subjects"]) == 21
        assert result["gradient_audit"]["sensor_token"] > 0
        assert result["gradient_audit"]["encoder"] > 0
        assert result["gradient_audit"]["projector"] > 0
        assert result["gradient_audit"]["head"] > 0
        assert result["gradient_audit"]["language_model"] == 0.0
        expected_name = MODEL if backend == "llm" else f"random:{MODEL}"
        assert result["llm_model"] == expected_name
        assert result["shuffled_test"]["repeats"] == 10
        runs.append(result)
    return runs


def load_identity_runs():
    runs = []
    for seed in SEEDS:
        path = (
            Path("outputs/llm_value/finetuned/identity/masked_aux_vcreg_003")
            / f"seed_{seed}"
            / "context.json"
        )
        result = json.loads(path.read_text())
        assert result["context_backend"] == "identity"
        assert result["seed"] == seed
        runs.append(result)
    return runs


def values(runs, key, nested=None):
    if nested is None:
        return np.array([run[key]["macro_f1"] * 100 for run in runs])
    return np.array([run[key][nested] * 100 for run in runs])


def describe(array):
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
    }


def paired(first, second):
    difference = first - second
    return {
        "mean_difference": float(difference.mean()),
        "paired_t_p_value": float(stats.ttest_rel(first, second).pvalue),
    }


def main():
    pretrained_runs = load_llm_runs("llm")
    random_runs = load_llm_runs("random_llm")
    identity_runs = load_identity_runs()

    pretrained = values(pretrained_runs, "test")
    random = values(random_runs, "test")
    identity = values(identity_runs, "test")
    shuffled = values(pretrained_runs, "shuffled_test", "macro_f1_mean")

    summary = {
        "model": MODEL,
        "prompt_style": "challenge",
        "seeds": list(SEEDS),
        "pretrained_llm": describe(pretrained),
        "random_llm": describe(random),
        "simple_identity": describe(identity),
        "shuffled_pretrained_llm": describe(shuffled),
        "sensor_dependence_drop": describe(pretrained - shuffled),
        "paired_comparisons": {
            "pretrained_minus_random": paired(pretrained, random),
            "pretrained_minus_simple": paired(pretrained, identity),
        },
    }
    output = Path("outputs/challenge_exact/summary.json")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote audited summary to {output}")


if __name__ == "__main__":
    main()
