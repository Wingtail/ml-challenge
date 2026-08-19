import json
from pathlib import Path

import numpy as np
from scipy import stats


SEEDS = (42, 43, 44, 45)
MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def describe(values):
    values = np.asarray(values) * 100
    return {
        "values": values.tolist(),
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
    }


def main():
    direct_runs = []
    context_runs = []
    frozen_context_runs = []
    linear_probe_runs = []
    for seed in SEEDS:
        direct_path = (
            Path("outputs/submission_compliant/direct/masked")
            / f"seed_{seed}"
            / "finetune.json"
        )
        context_path = (
            Path("outputs/submission_compliant/context/masked")
            / f"seed_{seed}"
            / "context.json"
        )
        frozen_context_path = (
            Path("outputs/submission_compliant/frozen_encoder_context/masked")
            / f"seed_{seed}"
            / "context.json"
        )
        linear_probe_path = (
            Path("outputs/submission_compliant/linear_probe/masked")
            / f"seed_{seed}"
            / "linear_probe.json"
        )
        direct = json.loads(direct_path.read_text())
        context = json.loads(context_path.read_text())
        frozen_context = json.loads(frozen_context_path.read_text())
        linear_probe = json.loads(linear_probe_path.read_text())

        assert direct["seed"] == context["seed"] == seed
        assert direct["training_subjects"] == context["training_subjects"]
        assert direct["validation_subjects"] == context["validation_subjects"] == []
        assert len(direct["training_subjects"]) == 21
        assert direct["trainable_parameters"] == {"encoder": 400_768, "head": 6_918}
        assert direct["gradient_audit"]["encoder"] > 0
        assert direct["gradient_audit"]["head"] > 0

        assert context["llm_model"] == MODEL
        assert context["prompt_style"] == "challenge"
        assert context["sensor_aux_weight"] == 0
        assert context["context_vcreg_weight"] == 0
        assert context["trainable_parameters"] == {
            "encoder": 400_768,
            "projector": 1_106_880,
            "head": 5_766,
            "sensor_head": 0,
            "language_model": 0,
            "context_model": 0,
        }
        assert context["gradient_audit"]["sensor_token"] > 0
        assert context["gradient_audit"]["encoder"] > 0
        assert context["gradient_audit"]["projector"] > 0
        assert context["gradient_audit"]["head"] > 0
        assert context["gradient_audit"]["language_model"] == 0
        assert context["shuffled_test"]["repeats"] == 10

        assert frozen_context["seed"] == seed
        assert frozen_context["encoder_frozen"] is True
        assert frozen_context["llm_model"] == MODEL
        assert frozen_context["prompt_style"] == "challenge"
        assert frozen_context["sensor_aux_weight"] == 0
        assert frozen_context["context_vcreg_weight"] == 0
        assert frozen_context["trainable_parameters"] == {
            "encoder": 0,
            "projector": 1_106_880,
            "head": 5_766,
            "sensor_head": 0,
            "language_model": 0,
            "context_model": 0,
        }
        assert frozen_context["gradient_audit"]["encoder"] == 0
        assert frozen_context["gradient_audit"]["projector"] > 0
        assert frozen_context["gradient_audit"]["head"] > 0
        assert frozen_context["gradient_audit"]["language_model"] == 0
        assert frozen_context["shuffled_test"]["repeats"] == 10

        assert linear_probe["seed"] == seed
        assert linear_probe["stage"] == "final_linear_probe"
        assert linear_probe["trainable_parameters"] == {
            "encoder": 0,
            "head": 6_918,
        }
        assert linear_probe["gradient_audit"]["encoder"] == 0
        assert linear_probe["gradient_audit"]["head"] > 0

        direct_runs.append(direct)
        context_runs.append(context)
        frozen_context_runs.append(frozen_context)
        linear_probe_runs.append(linear_probe)

    context_values = np.array(
        [run["test"]["macro_f1"] for run in context_runs]
    )
    frozen_context_values = np.array(
        [run["test"]["macro_f1"] for run in frozen_context_runs]
    )
    linear_probe_values = np.array(
        [run["test"]["macro_f1"] for run in linear_probe_runs]
    )

    summary = {
        "model": MODEL,
        "seeds": list(SEEDS),
        "primary_seed": 42,
        "direct_macro_f1": describe(
            [run["test"]["macro_f1"] for run in direct_runs]
        ),
        "context_macro_f1": describe(
            [run["test"]["macro_f1"] for run in context_runs]
        ),
        "shuffled_macro_f1": describe(
            [run["shuffled_test"]["macro_f1_mean"] for run in context_runs]
        ),
        "frozen_encoder_context_macro_f1": describe(frozen_context_values),
        "frozen_encoder_linear_probe_macro_f1": describe(linear_probe_values),
        "frozen_encoder_shuffled_macro_f1": describe(
            [
                run["shuffled_test"]["macro_f1_mean"]
                for run in frozen_context_runs
            ]
        ),
        "frozen_minus_trainable_encoder": {
            "differences": ((frozen_context_values - context_values) * 100).tolist(),
            "mean_difference": float(
                (frozen_context_values - context_values).mean() * 100
            ),
            "paired_t_p_value": float(
                stats.ttest_rel(frozen_context_values, context_values).pvalue
            ),
        },
        "linear_probe_minus_frozen_encoder_llm": {
            "differences": ((linear_probe_values - frozen_context_values) * 100).tolist(),
            "mean_difference": float(
                (linear_probe_values - frozen_context_values).mean() * 100
            ),
            "paired_t_p_value": float(
                stats.ttest_rel(linear_probe_values, frozen_context_values).pvalue
            ),
        },
        "trainable_parameters": {
            "direct": 407_686,
            "context": 1_513_414,
            "language_model": 0,
        },
    }
    output = Path("outputs/submission_compliant/summary.json")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote audited summary to {output}")


if __name__ == "__main__":
    main()
