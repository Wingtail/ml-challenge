"""Audit and summarize the exact 360M frozen-LLM ablation matrix."""

import json
from pathlib import Path

import numpy as np
from scipy import stats


SEEDS = (42, 43, 44, 45)
FRACTIONS = ("0.01", "0.05", "0.10", "0.25", "0.50", "1.0")
ROOT = Path("outputs/llm_ablation_exact")
SUBMISSION_ROOT = Path("outputs/submission_compliant")


def load_json(path):
    return json.loads(path.read_text())


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "values": values.tolist(),
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
    }


def paired(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    difference = first - second
    standard_error = stats.sem(difference)
    interval = stats.t.interval(
        0.95,
        len(difference) - 1,
        loc=difference.mean(),
        scale=standard_error,
    )
    return {
        "differences": difference.tolist(),
        "mean_difference": float(difference.mean()),
        "confidence_interval_95": [float(value) for value in interval],
        "paired_t_p_value": float(stats.ttest_rel(first, second).pvalue),
    }


def context_runs(root, expected_backend, frozen=False, fraction=1.0):
    runs = []
    for seed in SEEDS:
        result = load_json(root / f"seed_{seed}" / "context.json")
        assert result["seed"] == seed
        assert result["context_backend"] == expected_backend
        assert result["encoder_frozen"] is frozen
        assert result["prompt_style"] == "challenge"
        assert result["best_epoch"] == 18
        assert result["weight_decay"] == 0
        assert result["context_vcreg_weight"] == 0
        assert result["sensor_aux_weight"] == 0
        assert np.isclose(result["label_fraction"], fraction)
        assert result["validation_subjects"] == []
        assert result["gradient_audit"]["language_model"] in (None, 0.0)
        if frozen:
            assert result["gradient_audit"]["encoder"] == 0.0
        runs.append(result)
    return runs


def scores(runs):
    return np.asarray([100 * run["test"]["macro_f1"] for run in runs])


def summarize_context(runs):
    first = runs[0]
    trainable = first["trainable_parameters"]
    return {
        "macro_f1": describe(scores(runs)),
        "trainable_parameters": trainable,
        "total_trainable_parameters": sum(trainable.values()),
    }


def full_label_summary():
    roots = {
        "pretrained_llm": (
            SUBMISSION_ROOT / "context" / "masked",
            "llm",
        ),
        "random_llm": (ROOT / "random_llm" / "masked", "random_llm"),
        "identity": (ROOT / "identity" / "masked", "identity"),
        "mlp": (ROOT / "mlp" / "masked", "mlp"),
        "nonlinear_projector_llm": (
            ROOT / "nonlinear_llm" / "masked",
            "nonlinear_llm",
        ),
    }
    runs = {
        name: context_runs(path, backend)
        for name, (path, backend) in roots.items()
    }
    pretrained = scores(runs["pretrained_llm"])
    return {
        "models": {name: summarize_context(items) for name, items in runs.items()},
        "paired_pretrained_minus_control": {
            name: paired(pretrained, scores(items))
            for name, items in runs.items()
            if name != "pretrained_llm"
        },
    }


def layer_probe_summary():
    runs = [
        load_json(ROOT / "layer_probes" / f"seed_{seed}.json")
        for seed in SEEDS
    ]
    sensor = [100 * run["sensor_embedding"]["macro_f1"] for run in runs]
    projected = [100 * run["projected_token"]["macro_f1"] for run in runs]
    layer_count = len(runs[0]["llm_layers"])
    layers = []
    for layer in range(layer_count):
        values = [
            100 * run["llm_layers"][layer]["macro_f1"] for run in runs
        ]
        layers.append({"layer": layer, "macro_f1": describe(values)})
    best_layer = max(layers, key=lambda item: item["macro_f1"]["mean"])
    final_values = layers[-1]["macro_f1"]["values"]
    return {
        "sensor_embedding": describe(sensor),
        "projected_token": describe(projected),
        "llm_layers": layers,
        "best_mean_layer": best_layer,
        "projected_minus_best_layer": paired(
            projected, best_layer["macro_f1"]["values"]
        ),
        "projected_minus_final_layer": paired(projected, final_values),
    }


def low_label_summary():
    summary = {}
    for fraction_name in FRACTIONS:
        fraction = float(fraction_name)
        runs = {
            backend: context_runs(
                ROOT
                / "low_label"
                / fraction_name
                / backend
                / "masked",
                backend,
                frozen=True,
                fraction=fraction,
            )
            for backend in ("llm", "random_llm", "identity")
        }
        pretrained = scores(runs["llm"])
        summary[fraction_name] = {
            backend: describe(scores(items)) for backend, items in runs.items()
        }
        summary[fraction_name]["pretrained_minus_random"] = paired(
            pretrained, scores(runs["random_llm"])
        )
        summary[fraction_name]["pretrained_minus_identity"] = paired(
            pretrained, scores(runs["identity"])
        )
    return summary


def main():
    direct = load_json(SUBMISSION_ROOT / "summary.json")["direct_macro_f1"]
    summary = {
        "protocol": {
            "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "prompt_style": "challenge",
            "seeds": list(SEEDS),
            "epochs": 18,
            "batch_size": 128,
            "projector_head_learning_rate": 1e-3,
            "encoder_learning_rate_when_trainable": 1e-4,
            "weight_decay": 0.0,
            "auxiliary_loss": False,
            "language_model_frozen": True,
        },
        "direct_patchtst": direct,
        "full_label": full_label_summary(),
        "layer_probes": layer_probe_summary(),
        "low_label_frozen_encoder": low_label_summary(),
        "scope": {
            "lora_tested": False,
            "lora_reason": (
                "Training LLM adapters violates the challenge rule that only the "
                "sensor encoder, projector, and classification head may train."
            ),
            "ood_regime": "official unseen-subject test split",
            "external_domain_transfer_tested": False,
        },
    }
    output = ROOT / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
