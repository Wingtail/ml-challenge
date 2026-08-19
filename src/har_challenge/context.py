"""The required PatchTST -> frozen SmolLM2 classifier."""

import torch
from torch import nn


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
PROMPT_BEFORE_SENSOR = (
    "Classify the activity as walking, walking upstairs, walking downstairs, "
    "sitting, standing, or laying.\n\nSensor context: "
)
PROMPT_AFTER_SENSOR = "\n\nActivity:"


def tokenize_prompt(tokenizer):
    """Tokenize the fixed text before and after the continuous sensor token."""
    before_ids = tokenizer(
        PROMPT_BEFORE_SENSOR,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids[0]
    after_ids = tokenizer(
        PROMPT_AFTER_SENSOR,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0]
    return before_ids, after_ids


class FrozenLanguageClassifier(nn.Module):
    """Shared continuous-token mechanics for a supplied frozen language model."""

    def __init__(self, encoder, language_model, before_ids, after_ids, projector):
        super().__init__()
        self.encoder = encoder
        self.language_model = language_model
        self.projector = projector
        self.head = nn.Linear(960, 6)

        # The original study initialized this unused readout before creating the
        # shuffled training loader. Keeping it frozen preserves the reported RNG
        # sequence and therefore the exact fixed-seed checkpoints.
        self._study_rng_compatibility_readout = nn.Linear(1152, 6)
        self._study_rng_compatibility_readout.requires_grad_(False)

        self.language_model.requires_grad_(False)
        self.language_model.eval()
        self.register_buffer("before_ids", before_ids.long())
        self.register_buffer("after_ids", after_ids.long())

        with torch.no_grad():
            prompt_ids = torch.cat((before_ids, after_ids)).long()
            embeddings = self.language_model.get_input_embeddings()(prompt_ids)
            prompt_rms = embeddings.float().square().mean().sqrt()
        self.register_buffer("prompt_rms", prompt_rms)

    def train(self, mode=True):
        super().train(mode)
        self.language_model.eval()
        return self

    def project(self, sensor_embedding):
        token = self.projector(sensor_embedding)
        token_rms = token.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return token * (self.prompt_rms / token_rms)

    def encode_sensor_token(self, x):
        return self.project(self.encoder(x))

    def contextualize(self, sensor_token, return_all_layers=False):
        batch_size = len(sensor_token)
        word_embeddings = self.language_model.get_input_embeddings()
        before = word_embeddings(self.before_ids).unsqueeze(0)
        after = word_embeddings(self.after_ids).unsqueeze(0)
        before = before.expand(batch_size, -1, -1)
        after = after.expand(batch_size, -1, -1)

        inputs = torch.cat(
            (before, sensor_token.to(before.dtype).unsqueeze(1), after), dim=1
        )
        output = self.language_model(
            inputs_embeds=inputs,
            attention_mask=torch.ones(
                inputs.shape[:2], dtype=torch.long, device=inputs.device
            ),
            use_cache=False,
            return_dict=True,
            output_hidden_states=return_all_layers,
        )
        if return_all_layers:
            return tuple(layer[:, -1].float() for layer in output.hidden_states)
        return output.last_hidden_state[:, -1].float()

    def classify_sensor_token(self, sensor_token):
        return self.head(self.contextualize(sensor_token))

    def forward(self, x):
        return self.classify_sensor_token(self.encode_sensor_token(x))


class ContextClassifier(FrozenLanguageClassifier):
    """Required model with a linear projector and pretrained SmolLM2."""

    def __init__(self, encoder):
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        language_model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
        before_ids, after_ids = tokenize_prompt(tokenizer)
        projector = nn.Linear(1152, 960)
        super().__init__(
            encoder,
            language_model,
            before_ids,
            after_ids,
            projector,
        )
