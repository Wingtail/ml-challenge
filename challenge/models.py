import torch
from torch import nn

from challenge.objectives import EmbeddingProjector, joint_embedding_loss, sensor_view


CHALLENGE_PREFIX = (
    "Classify the activity as walking, walking upstairs, walking downstairs, "
    "sitting, standing, or laying.\n\nSensor context: "
)
CHALLENGE_SUFFIX = "\n\nActivity:"
SENSOR_PLACEHOLDER = "<SENSOR>"


def challenge_chat_parts(tokenizer):
    """Render the challenge with SmolLM2's chat template around one sensor slot."""
    messages = [
        {
            "role": "user",
            "content": CHALLENGE_PREFIX + SENSOR_PLACEHOLDER,
        },
        {"role": "assistant", "content": "Activity:"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        continue_final_message=True,
    )
    if prompt.count(SENSOR_PLACEHOLDER) != 1:
        raise ValueError("chat template must preserve exactly one sensor placeholder")
    prefix, suffix = prompt.split(SENSOR_PLACEHOLDER)
    if not suffix.endswith("Activity:"):
        raise ValueError("chat-formatted challenge prompt must end with Activity:")
    return prefix, suffix


class PatchTSTEncoder(nn.Module):
    """Channel-independent PatchTST encoder for [batch, 9, 128] input."""

    def __init__(
        self,
        channels: int = 9,
        window_size: int = 128,
        patch_length: int = 8,
        stride: int = 8,
        d_model: int = 128,
        layers: int = 3,
        heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if window_size < patch_length:
            raise ValueError("patch length cannot exceed the input window")
        self.channels = channels
        self.window_size = window_size
        self.patch_length = patch_length
        self.stride = stride
        self.patch_count = 1 + (window_size - patch_length) // stride
        self.d_model = d_model
        self.embedding_dim = channels * d_model

        # No bias means an all-zero masked patch contributes only its position.
        self.patch_projection = nn.Linear(patch_length, d_model, bias=False)
        self.position = nn.Parameter(torch.empty(1, self.patch_count, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.position, std=0.02)

    def make_patches(self, x: torch.Tensor):
        if x.ndim != 3 or x.shape[1:] != (self.channels, self.window_size):
            raise ValueError(
                f"expected [batch, {self.channels}, {self.window_size}], got {tuple(x.shape)}"
            )
        return x.unfold(dimension=-1, size=self.patch_length, step=self.stride)

    def encode_patches(self, patches: torch.Tensor, positions: torch.Tensor):
        """Encode [batch, channel, patch, sample] with original patch positions."""
        batch, channels, patch_count, _ = patches.shape
        patches = patches.reshape(batch * channels, patch_count, self.patch_length)
        positions = positions.reshape(batch * channels, patch_count)

        positional = self.position.expand(batch * channels, -1, -1)
        positional = positional.gather(
            1, positions.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        tokens = self.patch_projection(patches) + positional
        tokens = self.norm(self.transformer(tokens))
        return tokens.reshape(batch, channels, patch_count, self.d_model)

    def forward(self, x: torch.Tensor):
        tokens = self.forward_patch_tokens(x)
        # Concatenation retains the identity of each inertial channel. The linear
        # downstream head is responsible for channel mixing.
        return tokens.mean(dim=2).flatten(start_dim=1)

    def forward_patch_tokens(self, x: torch.Tensor):
        """Return all encoded channel/temporal tokens as [batch, 9, 16, d_model]."""
        patches = self.make_patches(x)
        batch = x.shape[0]
        positions = torch.arange(self.patch_count, device=x.device)
        positions = positions.view(1, 1, -1).expand(batch, self.channels, -1)
        return self.encode_patches(patches, positions)


class PatchTSTPretrainer(nn.Module):
    """Masked reconstruction with optional DropPatch token removal."""

    def __init__(
        self,
        encoder: PatchTSTEncoder,
        drop_ratio: float = 0.5,
        mask_ratio: float = 0.4,
    ):
        super().__init__()
        if not 0 <= drop_ratio < 1:
            raise ValueError("drop_ratio must be in [0, 1)")
        if not 0 < mask_ratio <= 1:
            raise ValueError("mask_ratio must be in (0, 1]")
        self.encoder = encoder
        self.drop_ratio = drop_ratio
        self.mask_ratio = mask_ratio
        self.reconstruction_head = nn.Linear(encoder.d_model, encoder.patch_length)

    def forward(self, x: torch.Tensor, return_details: bool = False):
        patches = self.encoder.make_patches(x)
        batch, channels, patch_count, patch_length = patches.shape
        rows = batch * channels
        keep_count = max(1, patch_count - round(patch_count * self.drop_ratio))

        # Each channel/sample gets an independent selection. Sorting the retained
        # indices restores temporal order while preserving original positions.
        random_order = torch.rand(rows, patch_count, device=x.device).argsort(dim=1)
        kept_positions = random_order[:, :keep_count].sort(dim=1).values
        flat_patches = patches.reshape(rows, patch_count, patch_length)
        retained = flat_patches.gather(
            1, kept_positions.unsqueeze(-1).expand(-1, -1, patch_length)
        )

        mask_count = max(1, round(keep_count * self.mask_ratio))
        mask_order = torch.rand(rows, keep_count, device=x.device).argsort(dim=1)
        masked = torch.zeros(rows, keep_count, dtype=torch.bool, device=x.device)
        masked.scatter_(1, mask_order[:, :mask_count], True)

        encoder_input = retained.masked_fill(masked.unsqueeze(-1), 0.0)
        encoder_input = encoder_input.reshape(
            batch, channels, keep_count, patch_length
        )
        positions = kept_positions.reshape(batch, channels, keep_count)
        tokens = self.encoder.encode_patches(encoder_input, positions)
        reconstructed = self.reconstruction_head(tokens).reshape(
            rows, keep_count, patch_length
        )
        patch_loss = (reconstructed - retained).square().mean(dim=-1)
        loss = patch_loss[masked].mean()

        if not return_details:
            return loss
        return loss, {
            "kept_positions": kept_positions,
            "masked": masked,
            "kept_count": keep_count,
            "masked_count": mask_count,
        }


class JointEmbeddingPretrainer(nn.Module):
    """VICReg pretraining, optionally combined with masked reconstruction."""

    def __init__(
        self,
        encoder: PatchTSTEncoder,
        covariance: str = "vicreg",
        reconstruction_weight: float = 0.0,
        representation_weight: float = 1.0,
        mask_ratio: float = 0.4,
    ):
        super().__init__()
        self.encoder = encoder
        self.covariance = covariance
        self.reconstruction_weight = reconstruction_weight
        self.representation_weight = representation_weight
        self.projector = EmbeddingProjector(encoder.embedding_dim)
        self.reconstructor = None
        if reconstruction_weight > 0:
            self.reconstructor = PatchTSTPretrainer(
                encoder, drop_ratio=0.0, mask_ratio=mask_ratio
            )

    def forward(self, x: torch.Tensor, return_details: bool = False):
        first = self.projector(self.encoder(sensor_view(x)))
        second = self.projector(self.encoder(sensor_view(x)))
        representation, parts = joint_embedding_loss(first, second, self.covariance)
        reconstruction = x.new_tensor(0.0)
        if self.reconstructor is not None:
            reconstruction = self.reconstructor(x)
        loss = (
            self.reconstruction_weight * reconstruction
            + self.representation_weight * representation
        )
        if not return_details:
            return loss
        return loss, {"reconstruction": reconstruction, **parts}


class SensorClassifier(nn.Module):
    def __init__(self, encoder: PatchTSTEncoder, classes: int = 6):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embedding_dim, classes)

    def forward(self, x: torch.Tensor):
        return self.head(self.encoder(x))


class FullyConvolutionalClassifier(nn.Module):
    """Simple end-to-end FCN for [batch, 9, 128] inertial windows."""

    def __init__(self, channels: int = 9, classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, 128, kernel_size=8, padding="same", bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding="same", bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.head = nn.Linear(128, classes)

    def forward(self, x: torch.Tensor):
        if x.ndim != 3 or x.shape[1:] != (9, 128):
            raise ValueError(f"expected [batch, 9, 128], got {tuple(x.shape)}")
        features = self.features(x)
        return self.head(features.mean(dim=-1))


class MLPContextClassifier(nn.Module):
    """A one-layer nonlinear readout with the same trainable LLM interface."""

    def __init__(
        self,
        encoder: PatchTSTEncoder,
        hidden_size: int = 576,
        classes: int = 6,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Linear(encoder.embedding_dim, hidden_size)
        self.head = nn.Linear(hidden_size, classes)
        self.sensor_head = nn.Linear(encoder.embedding_dim, classes)

    def project_sensor_embedding(self, embedding: torch.Tensor):
        return self.projector(embedding)

    def encode_sensor_token(self, x: torch.Tensor):
        return self.project_sensor_embedding(self.encoder(x))

    def contextualize_sensor_token(self, sensor_token: torch.Tensor):
        return torch.nn.functional.gelu(sensor_token)

    def classify_sensor_token(self, sensor_token: torch.Tensor):
        return self.head(self.contextualize_sensor_token(sensor_token))

    def forward(self, x: torch.Tensor):
        return self.classify_sensor_token(self.encode_sensor_token(x))


class IdentityContextClassifier(nn.Module):
    """Linear projector and head control with no contextual transformation."""

    def __init__(
        self,
        encoder: PatchTSTEncoder,
        hidden_size: int = 576,
        classes: int = 6,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Linear(encoder.embedding_dim, hidden_size)
        self.head = nn.Linear(hidden_size, classes)
        self.sensor_head = nn.Linear(encoder.embedding_dim, classes)

    def project_sensor_embedding(self, embedding: torch.Tensor):
        return self.projector(embedding)

    def encode_sensor_token(self, x: torch.Tensor):
        return self.project_sensor_embedding(self.encoder(x))

    def contextualize_sensor_token(self, sensor_token: torch.Tensor):
        return sensor_token

    def classify_sensor_token(self, sensor_token: torch.Tensor):
        return self.head(self.contextualize_sensor_token(sensor_token))

    def forward(self, x: torch.Tensor):
        return self.classify_sensor_token(self.encode_sensor_token(x))


class MultiTokenTransformerClassifier(nn.Module):
    """Process channel or temporal sensor tokens with a small Transformer."""

    def __init__(
        self,
        encoder: PatchTSTEncoder,
        hidden_size: int = 576,
        classes: int = 6,
        dropout: float = 0.1,
        token_layout: str = "temporal",
    ):
        super().__init__()
        if token_layout not in {"temporal", "channel"}:
            raise ValueError(f"unknown token layout: {token_layout}")
        self.encoder = encoder
        self.token_layout = token_layout
        self.projector = nn.Linear(encoder.d_model, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=2 * hidden_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_model = nn.TransformerEncoder(layer, num_layers=2)
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, classes)
        self.sensor_head = nn.Linear(encoder.embedding_dim, classes)

    def encode_context_input(self, x: torch.Tensor):
        patch_tokens = self.encoder.forward_patch_tokens(x)
        sensor_embedding = patch_tokens.mean(dim=2).flatten(start_dim=1)
        if self.token_layout == "temporal":
            context_tokens = patch_tokens.mean(dim=1)
        else:
            context_tokens = patch_tokens.mean(dim=2)
        return sensor_embedding, self.projector(context_tokens)

    def encode_sensor_token(self, x: torch.Tensor):
        return self.encode_context_input(x)[1]

    def contextualize_sensor_token(self, sensor_token: torch.Tensor):
        query = self.query.expand(sensor_token.shape[0], -1, -1)
        hidden = self.context_model(torch.cat((sensor_token, query), dim=1))
        return self.norm(hidden[:, -1])

    def classify_sensor_token(self, sensor_token: torch.Tensor):
        return self.head(self.contextualize_sensor_token(sensor_token))

    def context_parameters(self):
        yield self.query
        yield from self.context_model.parameters()
        yield from self.norm.parameters()

    def forward(self, x: torch.Tensor):
        return self.classify_sensor_token(self.encode_sensor_token(x))


class ContinuousTokenClassifier(nn.Module):
    """Insert one learned sensor vector into a frozen language model sequence."""

    def __init__(
        self,
        encoder: PatchTSTEncoder,
        language_model: nn.Module,
        prefix_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
        classes: int = 6,
        temporal_tokens: bool = False,
        channel_tokens: bool = False,
        nonlinear_projector: bool = False,
    ):
        super().__init__()
        if temporal_tokens and channel_tokens:
            raise ValueError("choose either temporal or channel tokens")
        self.encoder = encoder
        self.language_model = language_model
        for parameter in self.language_model.parameters():
            parameter.requires_grad = False
        self.language_model.eval()

        word_embeddings = language_model.get_input_embeddings()
        hidden_size = word_embeddings.embedding_dim
        self.temporal_tokens = temporal_tokens
        self.channel_tokens = channel_tokens
        projector_input = (
            encoder.d_model
            if temporal_tokens or channel_tokens
            else encoder.embedding_dim
        )
        if nonlinear_projector:
            self.projector = nn.Sequential(
                nn.Linear(projector_input, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            self.projector = nn.Linear(projector_input, hidden_size)
        self.head = nn.Linear(hidden_size, classes)
        self.sensor_head = nn.Linear(encoder.embedding_dim, classes)
        self.register_buffer("prefix_ids", prefix_ids.long(), persistent=False)
        self.register_buffer("suffix_ids", suffix_ids.long(), persistent=False)

        with torch.no_grad():
            prompt_ids = torch.cat((prefix_ids, suffix_ids)).long()
            prompt_embeddings = word_embeddings(prompt_ids)
            token_rms = prompt_embeddings.float().square().mean().sqrt()
        self.register_buffer("token_rms", token_rms)

    @classmethod
    def from_pretrained(
        cls,
        encoder: PatchTSTEncoder,
        model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        dtype: torch.dtype = torch.bfloat16,
        prompt_style: str = "semantic",
        random_weights: bool = False,
        temporal_tokens: bool = False,
        channel_tokens: bool = False,
        nonlinear_projector: bool = False,
    ):
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if random_weights:
            config = AutoConfig.from_pretrained(model_name)
            language_model = AutoModel.from_config(config).to(dtype=dtype)
        else:
            language_model = AutoModel.from_pretrained(model_name, dtype=dtype)
        if prompt_style == "answer":
            prefix_ids = torch.empty(0, dtype=torch.long)
            suffix_ids = tokenizer(
                "Answer:", add_special_tokens=False, return_tensors="pt"
            ).input_ids[0]
        else:
            if prompt_style == "challenge_chat":
                prefix, suffix = challenge_chat_parts(tokenizer)
                add_prefix_special_tokens = False
            elif prompt_style == "challenge":
                prefix = CHALLENGE_PREFIX
                suffix = CHALLENGE_SUFFIX
                add_prefix_special_tokens = True
            else:
                prefix = "Classify the human activity from this sensor measurement:\n"
                suffix = "\nActivity:"
                add_prefix_special_tokens = True
            prefix_ids = tokenizer(
                prefix,
                add_special_tokens=add_prefix_special_tokens,
                return_tensors="pt",
            ).input_ids[0]
            suffix_ids = tokenizer(
                suffix, add_special_tokens=False, return_tensors="pt"
            ).input_ids[0]
        if prompt_style == "scrambled":
            prefix_ids, suffix_ids = scramble_prompt_tokens(prefix_ids, suffix_ids)
        elif prompt_style not in {
            "semantic",
            "answer",
            "challenge",
            "challenge_chat",
        }:
            raise ValueError(f"unknown prompt style: {prompt_style}")
        return cls(
            encoder,
            language_model,
            prefix_ids,
            suffix_ids,
            temporal_tokens=temporal_tokens,
            channel_tokens=channel_tokens,
            nonlinear_projector=nonlinear_projector,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # A frozen LLM should also have deterministic dropout behavior.
        self.language_model.eval()
        return self

    def project_sensor_embedding(self, embedding: torch.Tensor):
        sensor = self.projector(embedding)
        sensor_rms = sensor.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return sensor * (self.token_rms / sensor_rms)

    def encode_sensor_token(self, x: torch.Tensor):
        return self.encode_context_input(x)[1]

    def encode_context_input(self, x: torch.Tensor):
        if not self.temporal_tokens and not self.channel_tokens:
            sensor_embedding = self.encoder(x)
            return sensor_embedding, self.project_sensor_embedding(sensor_embedding)
        patch_tokens = self.encoder.forward_patch_tokens(x)
        sensor_embedding = patch_tokens.mean(dim=2).flatten(start_dim=1)
        if self.temporal_tokens:
            context_tokens = patch_tokens.mean(dim=1)
        else:
            context_tokens = patch_tokens.mean(dim=2)
        return sensor_embedding, self.project_sensor_embedding(context_tokens)

    def contextualize_sensor_token(self, sensor_token: torch.Tensor):
        batch = sensor_token.shape[0]
        embeddings = self.language_model.get_input_embeddings()
        prefix = embeddings(self.prefix_ids).unsqueeze(0).expand(batch, -1, -1)
        suffix = embeddings(self.suffix_ids).unsqueeze(0).expand(batch, -1, -1)
        sensor_token = sensor_token.to(prefix.dtype)
        if sensor_token.ndim == 2:
            sensor_token = sensor_token.unsqueeze(1)
        inputs = torch.cat((prefix, sensor_token, suffix), dim=1)
        attention_mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
        output = self.language_model(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        # The final suffix token can attend to the continuous sensor token.
        return output.last_hidden_state[:, -1].float()

    def contextualize_all_layers(self, sensor_token: torch.Tensor):
        """Return the final prompt-token state before and after every LLM block."""
        batch = sensor_token.shape[0]
        embeddings = self.language_model.get_input_embeddings()
        prefix = embeddings(self.prefix_ids).unsqueeze(0).expand(batch, -1, -1)
        suffix = embeddings(self.suffix_ids).unsqueeze(0).expand(batch, -1, -1)
        sensor_token = sensor_token.to(prefix.dtype)
        if sensor_token.ndim == 2:
            sensor_token = sensor_token.unsqueeze(1)
        inputs = torch.cat((prefix, sensor_token, suffix), dim=1)
        attention_mask = torch.ones(
            inputs.shape[:2], dtype=torch.long, device=inputs.device
        )
        output = self.language_model(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        # Every suffix state can attend to the preceding continuous sensor token.
        return tuple(hidden[:, -1].float() for hidden in output.hidden_states)

    def classify_sensor_token(self, sensor_token: torch.Tensor):
        return self.head(self.contextualize_sensor_token(sensor_token))

    def forward(self, x: torch.Tensor):
        return self.classify_sensor_token(self.encode_sensor_token(x))


def scramble_prompt_tokens(
    prefix_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    seed: int = 20_250_818,
):
    """Destroy prompt word order while preserving length and the final token."""
    prompt_ids = torch.cat((prefix_ids, suffix_ids))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(prompt_ids) - 1, generator=generator)
    scrambled = torch.cat((prompt_ids[:-1][permutation], prompt_ids[-1:]))
    prefix_length = len(prefix_ids)
    return scrambled[:prefix_length], scrambled[prefix_length:]
