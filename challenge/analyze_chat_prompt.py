"""Summarize the matched literal-prompt and SmolLM2 chat-template runs."""

import json
from pathlib import Path

import numpy as np
from scipy import stats


SEEDS = (42, 43, 44, 45)


def load_scores(root: str, expected_prompt: str):
    scores = []
    shuffled = []
    for seed in SEEDS:
        path = Path(root) / "masked" / f"seed_{seed}" / "context.json"
        result = json.loads(path.read_text())
        assert result["seed"] == seed
        assert result["prompt_style"] == expected_prompt
        assert result["best_epoch"] == 18
        assert result["gradient_audit"]["language_model"] == 0.0
        scores.append(100 * result["test"]["macro_f1"])
        shuffled.append(100 * result["shuffled_test"]["macro_f1_mean"])
    return np.asarray(scores), np.asarray(shuffled)


def describe(values):
    return {
        "values": values.tolist(),
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
    }


def paired(first, second):
    difference = first - second
    interval = stats.t.interval(
        0.95,
        len(difference) - 1,
        loc=difference.mean(),
        scale=stats.sem(difference),
    )
    return {
        "differences": difference.tolist(),
        "mean_difference": float(difference.mean()),
        "confidence_interval_95": [float(value) for value in interval],
        "paired_t_p_value": float(stats.ttest_rel(first, second).pvalue),
    }


def main():
    literal, _ = load_scores(
        "outputs/submission_compliant/context", "challenge"
    )
    chat_pretrained, chat_shuffled = load_scores(
        "outputs/chat_prompt", "challenge_chat"
    )
    chat_random, random_shuffled = load_scores(
        "outputs/chat_prompt_random", "challenge_chat"
    )
    summary = {
        "seeds": list(SEEDS),
        "literal_pretrained_llm": describe(literal),
        "chat_pretrained_llm": describe(chat_pretrained),
        "chat_random_llm": describe(chat_random),
        "chat_pretrained_shuffled": describe(chat_shuffled),
        "chat_random_shuffled": describe(random_shuffled),
        "chat_pretrained_minus_literal": paired(chat_pretrained, literal),
        "chat_pretrained_minus_random": paired(chat_pretrained, chat_random),
    }
    output = Path("outputs/chat_prompt/summary.json")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
