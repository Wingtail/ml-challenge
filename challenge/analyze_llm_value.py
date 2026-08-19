import json
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path("outputs/llm_value")
SEEDS = (42, 43, 44, 45)
BACKENDS = ("llm", "random_llm", "identity", "mlp")
FRACTIONS = ("0p01", "0p05", "0p10", "0p25")
MULTI_BACKENDS = (
    "multi_llm",
    "multi_random_llm",
    "multi_transformer",
    "channel_llm",
    "channel_random_llm",
    "channel_transformer",
)


def load_runs(directory: Path, backend: str, frozen: bool, fraction: float):
    runs = []
    for seed in SEEDS:
        path = directory / backend / "masked_aux_vcreg_003" / f"seed_{seed}" / "context.json"
        result = json.loads(path.read_text())
        assert result["seed"] == seed
        assert result["context_backend"] == backend
        assert result["encoder_frozen"] is frozen
        assert result["prompt_style"] == "answer"
        assert result["best_epoch"] == 18
        assert result["weight_decay"] == 0
        assert np.isclose(result["label_fraction"], fraction)
        assert result["validation_subjects"] == []
        assert len(result["training_subjects"]) == 21
        assert result["gradient_audit"]["language_model"] in (None, 0.0)
        if frozen:
            assert result["gradient_audit"]["encoder"] == 0.0
            assert result["trainable_parameters"]["encoder"] == 0
        else:
            assert result["gradient_audit"]["encoder"] > 0
        runs.append(result)
    return runs


def values(runs, key):
    return np.array([run[key]["macro_f1"] * 100 for run in runs])


def describe(array):
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
    }


def paired(first, second):
    differences = np.asarray(first) - np.asarray(second)
    standard_error = stats.sem(differences)
    if standard_error == 0:
        confidence_interval = (float(differences.mean()), float(differences.mean()))
    else:
        confidence_interval = stats.t.interval(
            0.95, len(differences) - 1, loc=differences.mean(), scale=standard_error
        )
    return {
        "mean_difference": float(differences.mean()),
        "confidence_interval_95": [float(value) for value in confidence_interval],
        "paired_t_p_value": float(stats.ttest_rel(first, second).pvalue),
    }


def summarize_runs(runs):
    return {
        "context": describe(values(runs, "test")),
        "sensor_head": describe(values(runs, "test_sensor_head")),
    }


def main():
    summary = {"seeds": list(SEEDS), "full_label": {}, "low_label_frozen": {}, "multi_token_frozen": {}}

    for mode, frozen in (("finetuned", False), ("frozen", True)):
        mode_runs = {
            backend: load_runs(ROOT / mode, backend, frozen=frozen, fraction=1.0)
            for backend in BACKENDS
        }
        summary["full_label"][mode] = {
            backend: summarize_runs(runs) for backend, runs in mode_runs.items()
        }
        llm_values = values(mode_runs["llm"], "test")
        summary["full_label"][mode]["pretrained_llm_paired_comparisons"] = {
            backend: paired(llm_values, values(mode_runs[backend], "test"))
            for backend in BACKENDS[1:]
        }

    fraction_lookup = {"0p01": 0.01, "0p05": 0.05, "0p10": 0.10, "0p25": 0.25}
    for fraction_name in FRACTIONS:
        fraction = fraction_lookup[fraction_name]
        fraction_runs = {
            backend: load_runs(
                ROOT / "low_label" / fraction_name,
                backend,
                frozen=True,
                fraction=fraction,
            )
            for backend in BACKENDS
        }
        section = {backend: summarize_runs(runs) for backend, runs in fraction_runs.items()}
        llm_values = values(fraction_runs["llm"], "test")
        section["pretrained_llm_paired_comparisons"] = {
            backend: paired(llm_values, values(fraction_runs[backend], "test"))
            for backend in BACKENDS[1:]
        }
        summary["low_label_frozen"][fraction_name] = section

    multi_runs = {
        backend: load_runs(
            ROOT / "multi_token", backend, frozen=True, fraction=1.0
        )
        for backend in MULTI_BACKENDS
    }
    summary["multi_token_frozen"] = {
        backend: summarize_runs(runs) for backend, runs in multi_runs.items()
    }
    summary["multi_token_frozen"]["paired_comparisons"] = {
        "temporal_pretrained_minus_random": paired(
            values(multi_runs["multi_llm"], "test"),
            values(multi_runs["multi_random_llm"], "test"),
        ),
        "channel_pretrained_minus_random": paired(
            values(multi_runs["channel_llm"], "test"),
            values(multi_runs["channel_random_llm"], "test"),
        ),
    }

    output = ROOT / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote audited summary to {output}")


if __name__ == "__main__":
    main()
