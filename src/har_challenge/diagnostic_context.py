"""Alternative context backends used only by diagnostic experiments."""

import torch
from torch import nn

from .context import (
    MODEL_NAME,
    PROMPT_AFTER_SENSOR,
    PROMPT_BEFORE_SENSOR,
    FrozenLanguageClassifier,
    tokenize_prompt,
)


SENSOR_SLOT = "<SENSOR>"


def chat_prompt_ids(tokenizer):
    messages = [
        {"role": "user", "content": PROMPT_BEFORE_SENSOR + SENSOR_SLOT},
        {"role": "assistant", "content": "Activity:"},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        continue_final_message=True,
    )
    if text.count(SENSOR_SLOT) != 1:
        raise ValueError("chat template must contain exactly one sensor slot")

    before, after = text.split(SENSOR_SLOT)
    before_ids = tokenizer(
        before, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0]
    after_ids = tokenizer(
        after, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0]
    return before_ids, after_ids


class IdentityContextClassifier(nn.Module):
    """Matched control that sends the projected token directly to the head."""

    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder
        self.language_model = None
        self.projector = projector
        self.head = nn.Linear(960, 6)
        self._study_rng_compatibility_readout = nn.Linear(1152, 6)
        self._study_rng_compatibility_readout.requires_grad_(False)

    def project(self, sensor_embedding):
        return self.projector(sensor_embedding)

    def encode_sensor_token(self, x):
        return self.project(self.encoder(x))

    def contextualize(self, sensor_token, return_all_layers=False):
        return (sensor_token,) if return_all_layers else sensor_token

    def classify_sensor_token(self, sensor_token):
        return self.head(sensor_token)

    def forward(self, x):
        return self.classify_sensor_token(self.encode_sensor_token(x))


def build_diagnostic_context(
    encoder,
    *,
    backend="pretrained",
    projector="linear",
    prompt_style="literal",
):
    """Build one explicitly requested diagnostic variation."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    def make_projector():
        if projector == "linear":
            return nn.Linear(1152, 960)
        if projector == "mlp":
            return nn.Sequential(
                nn.Linear(1152, 960),
                nn.GELU(),
                nn.Linear(960, 960),
            )
        raise ValueError("projector must be 'linear' or 'mlp'")

    if backend == "identity":
        return IdentityContextClassifier(encoder, make_projector())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if backend == "pretrained":
        language_model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    elif backend == "random":
        config = AutoConfig.from_pretrained(MODEL_NAME)
        language_model = AutoModel.from_config(config).to(torch.bfloat16)
    else:
        raise ValueError("backend must be 'pretrained', 'random', or 'identity'")

    if prompt_style == "literal":
        before_ids, after_ids = tokenize_prompt(tokenizer)
    elif prompt_style == "chat":
        before_ids, after_ids = chat_prompt_ids(tokenizer)
    else:
        raise ValueError("prompt_style must be 'literal' or 'chat'")

    return FrozenLanguageClassifier(
        encoder,
        language_model,
        before_ids,
        after_ids,
        make_projector(),
    )
