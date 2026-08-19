import torch
from torch import nn
from types import SimpleNamespace

from challenge.data import (
    OFFICIAL_TEST_SUBJECTS,
    OFFICIAL_TRAIN_SUBJECTS,
    load_challenge_data,
)
from challenge.models import (
    CHALLENGE_PREFIX,
    CHALLENGE_SUFFIX,
    ContinuousTokenClassifier,
    FullyConvolutionalClassifier,
    IdentityContextClassifier,
    JointEmbeddingPretrainer,
    MLPContextClassifier,
    MultiTokenTransformerClassifier,
    scramble_prompt_tokens,
    PatchTSTEncoder,
    PatchTSTPretrainer,
    challenge_chat_parts,
)
from challenge.train import context_optimizer_groups


def test_scrambled_prompt_preserves_controlled_properties():
    prefix = torch.tensor([1, 2, 3, 4])
    suffix = torch.tensor([5, 6, 7])
    scrambled_prefix, scrambled_suffix = scramble_prompt_tokens(prefix, suffix)

    original = torch.cat((prefix, suffix))
    scrambled = torch.cat((scrambled_prefix, scrambled_suffix))
    assert len(scrambled_prefix) == len(prefix)
    assert len(scrambled_suffix) == len(suffix)
    assert scrambled[-1] == original[-1]
    assert sorted(scrambled.tolist()) == sorted(original.tolist())
    assert not torch.equal(scrambled, original)


def test_challenge_prompt_matches_specification():
    prompt = CHALLENGE_PREFIX + "<SENSOR>" + CHALLENGE_SUFFIX
    assert prompt == (
        "Classify the activity as walking, walking upstairs, walking downstairs, "
        "sitting, standing, or laying.\n\nSensor context: <SENSOR>\n\nActivity:"
    )


def test_challenge_chat_prompt_uses_assistant_prefill_and_preserves_sensor_slot():
    class FakeTokenizer:
        def apply_chat_template(
            self, messages, tokenize, continue_final_message
        ):
            assert tokenize is False
            assert continue_final_message is True
            assert messages == [
                {
                    "role": "user",
                    "content": CHALLENGE_PREFIX + "<SENSOR>",
                },
                {"role": "assistant", "content": "Activity:"},
            ]
            return (
                "<|im_start|>user\n"
                + messages[0]["content"]
                + "<|im_end|>\n<|im_start|>assistant\n"
                + messages[1]["content"]
            )

    prefix, suffix = challenge_chat_parts(FakeTokenizer())

    assert prefix.endswith("Sensor context: ")
    assert suffix == "<|im_end|>\n<|im_start|>assistant\nActivity:"
from challenge.objectives import covariance_loss, normalized_covariance_loss, variance_loss
from har_benchmark.models import make_pretrainer
from har_benchmark.objectives import nt_xent


def test_challenge_data_uses_official_subject_disjoint_partitions():
    data = load_challenge_data("dataset/UCI HAR Dataset")
    training_subjects = set(data["subjects_train"].tolist())
    validation_subjects = set(data["subjects_validation"].tolist())
    test_subjects = set(data["subjects_test"].tolist())

    assert data["x_train"].shape == (5551, 9, 128)
    assert data["x_validation"].shape == (1801, 9, 128)
    assert data["x_test"].shape == (2947, 9, 128)
    assert not training_subjects & validation_subjects
    assert not (training_subjects | validation_subjects) & test_subjects
    assert training_subjects | validation_subjects == set(OFFICIAL_TRAIN_SUBJECTS)
    assert test_subjects == set(OFFICIAL_TEST_SUBJECTS)

    final_data = load_challenge_data(
        "dataset/UCI HAR Dataset", validation_subjects=()
    )
    assert final_data["x_train"].shape == (7352, 9, 128)
    assert final_data["x_validation"].shape == (0, 9, 128)


def test_all_models_forward_and_backward():
    x = torch.randn(4, 9, 128)
    for name in ("dctcss", "tclhar", "autocl", "maskcae"):
        model, encoder = make_pretrainer(name)
        embedding = encoder(x)
        assert embedding.shape == (4, encoder.embedding_dim)

        if name == "dctcss":
            loss = model.online(x).square().mean()
        elif name == "tclhar":
            loss = nt_xent(model(x), model(x.roll(1, dims=2)), 1.0)
        elif name == "autocl":
            first, second, augmented = model(x)
            assert augmented.shape == x.shape
            loss = nt_xent(first, second, 0.1)
        else:
            loss = model(x)
        loss.backward()
        assert torch.isfinite(loss)


def test_drop_patch_forward_backward_and_counts():
    x = torch.randn(4, 9, 128)
    encoder = PatchTSTEncoder(d_model=32, layers=1, heads=4, feedforward_dim=64)
    model = PatchTSTPretrainer(encoder, drop_ratio=0.5, mask_ratio=0.4)
    loss, details = model(x, return_details=True)

    assert encoder(x).shape == (4, 9 * 32)
    assert details["kept_count"] == 8
    assert details["masked_count"] == 3
    assert details["kept_positions"].shape == (4 * 9, 8)
    assert details["masked"].sum().item() == 4 * 9 * 3
    assert torch.all(details["kept_positions"][:, 1:] > details["kept_positions"][:, :-1])
    loss.backward()
    assert torch.isfinite(loss)


def test_fully_convolutional_classifier_forward_and_backward():
    model = FullyConvolutionalClassifier()
    logits = model(torch.randn(4, 9, 128))
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 2, 3]))
    loss.backward()

    assert logits.shape == (4, 6)
    assert model.features[0].weight.grad is not None
    assert model.head.weight.grad is not None


def test_joint_embedding_objectives_forward_and_backward():
    x = torch.randn(8, 9, 128)
    for covariance, reconstruction_weight in (("vicreg", 1.0), ("vibcreg", 0.0)):
        encoder = PatchTSTEncoder(d_model=16, layers=1, heads=4, feedforward_dim=32)
        model = JointEmbeddingPretrainer(
            encoder,
            covariance=covariance,
            reconstruction_weight=reconstruction_weight,
            representation_weight=0.05 if reconstruction_weight else 1.0,
        )
        loss, details = model(x, return_details=True)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(value) for value in details.values())
        assert encoder.patch_projection.weight.grad is not None


def test_rank_regularizers_penalize_collapsed_embeddings():
    collapsed = torch.zeros(8, 16)
    varied = torch.randn(8, 16)
    assert variance_loss(collapsed) > variance_loss(varied)
    assert torch.isfinite(covariance_loss(varied))
    assert torch.isfinite(normalized_covariance_loss(varied))


class TinyFrozenLanguageModel(nn.Module):
    def __init__(self, vocabulary=32, hidden_size=16):
        super().__init__()
        self.embeddings = nn.Embedding(vocabulary, hidden_size)
        self.mixer = nn.Linear(hidden_size, hidden_size)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        inputs_embeds,
        attention_mask,
        use_cache,
        return_dict,
        output_hidden_states=False,
    ):
        del attention_mask, use_cache, return_dict
        # Cumulative mixing makes the final suffix state depend on the sensor token.
        hidden = self.mixer(inputs_embeds.cumsum(dim=1))
        hidden_states = (hidden,) if output_hidden_states else None
        return SimpleNamespace(last_hidden_state=hidden, hidden_states=hidden_states)


def test_continuous_token_backpropagates_through_frozen_language_model():
    encoder = PatchTSTEncoder(d_model=16, layers=1, heads=4, feedforward_dim=32)
    language_model = TinyFrozenLanguageModel()
    model = ContinuousTokenClassifier(
        encoder,
        language_model,
        prefix_ids=torch.tensor([1, 2]),
        suffix_ids=torch.tensor([3, 4]),
    )
    model.train()
    sensor_embedding = model.encoder(torch.randn(3, 9, 128))
    sensor_token = model.project_sensor_embedding(sensor_embedding)
    sensor_token.retain_grad()
    hidden = model.contextualize_sensor_token(sensor_token)
    loss = model.head(hidden).square().mean()
    loss.backward()

    assert not language_model.training
    assert all(parameter.grad is None for parameter in language_model.parameters())
    assert sensor_token.grad is not None
    assert sensor_token.grad.norm() > 0
    assert model.projector.weight.grad is not None
    assert encoder.patch_projection.weight.grad is not None
    assert model.head.weight.grad is not None


def test_continuous_token_nonlinear_projector_and_layer_states():
    encoder = PatchTSTEncoder(d_model=16, layers=1, heads=4, feedforward_dim=32)
    model = ContinuousTokenClassifier(
        encoder,
        TinyFrozenLanguageModel(),
        prefix_ids=torch.tensor([1, 2]),
        suffix_ids=torch.tensor([3, 4]),
        nonlinear_projector=True,
    )
    sensor_token = model.encode_sensor_token(torch.randn(3, 9, 128))
    hidden_states = model.contextualize_all_layers(sensor_token)
    loss = model.head(hidden_states[-1]).square().mean()
    loss.backward()

    assert len(hidden_states) == 1
    assert hidden_states[0].shape == (3, 16)
    assert model.projector[0].weight.grad is not None
    assert model.projector[2].weight.grad is not None


def test_context_optimizer_uses_differential_learning_rates():
    encoder = PatchTSTEncoder(d_model=16, layers=1, heads=4, feedforward_dim=32)
    model = ContinuousTokenClassifier(
        encoder,
        TinyFrozenLanguageModel(),
        prefix_ids=torch.tensor([1, 2]),
        suffix_ids=torch.tensor([3, 4]),
    )
    groups = context_optimizer_groups(model, 1e-5, 3e-4)

    assert [group["lr"] for group in groups] == [1e-5, 3e-4]
    grouped_ids = {id(parameter) for group in groups for parameter in group["params"]}
    assert all(id(parameter) in grouped_ids for parameter in model.encoder.parameters())
    assert all(id(parameter) in grouped_ids for parameter in model.projector.parameters())
    assert all(id(parameter) in grouped_ids for parameter in model.head.parameters())
    assert all(
        id(parameter) not in grouped_ids for parameter in model.language_model.parameters()
    )


def test_continuous_token_supports_an_empty_prefix():
    encoder = PatchTSTEncoder(
        channels=9,
        window_size=128,
        patch_length=16,
        stride=16,
        d_model=16,
        layers=1,
        heads=2,
        feedforward_dim=32,
        dropout=0.0,
    )
    language_model = TinyFrozenLanguageModel(vocabulary=32, hidden_size=16)
    model = ContinuousTokenClassifier(
        encoder,
        language_model,
        prefix_ids=torch.empty(0, dtype=torch.long),
        suffix_ids=torch.tensor([5, 6]),
    )

    assert model(torch.randn(2, 9, 128)).shape == (2, 6)


def test_auxiliary_sensor_head_backpropagates_to_encoder():
    encoder = PatchTSTEncoder(d_model=16, layers=1, heads=4, feedforward_dim=32)
    model = ContinuousTokenClassifier(
        encoder,
        TinyFrozenLanguageModel(),
        prefix_ids=torch.tensor([1, 2]),
        suffix_ids=torch.tensor([3, 4]),
    )
    embedding = model.encoder(torch.randn(4, 9, 128))
    loss = nn.CrossEntropyLoss()(model.sensor_head(embedding), torch.tensor([0, 1, 2, 3]))
    loss.backward()

    assert model.sensor_head.weight.grad is not None
    assert encoder.patch_projection.weight.grad is not None


def test_parameter_matched_mlp_forward_and_backward():
    encoder = PatchTSTEncoder(d_model=16, layers=1, heads=4, feedforward_dim=32)
    model = MLPContextClassifier(encoder, hidden_size=16)
    x = torch.randn(4, 9, 128)
    token = model.encode_sensor_token(x)
    hidden = model.contextualize_sensor_token(token)
    loss = model.head(hidden).square().mean()
    loss.backward()

    assert token.shape == hidden.shape == (4, 16)
    assert model.projector.weight.grad is not None
    assert encoder.patch_projection.weight.grad is not None


def test_identity_control_forward_and_backward():
    encoder = PatchTSTEncoder(d_model=16, layers=1, heads=4, feedforward_dim=32)
    model = IdentityContextClassifier(encoder, hidden_size=16)
    x = torch.randn(4, 9, 128)
    token = model.encode_sensor_token(x)
    hidden = model.contextualize_sensor_token(token)
    loss = model.head(hidden).square().mean()
    loss.backward()

    assert torch.equal(token, hidden)
    assert model.projector.weight.grad is not None
    assert encoder.patch_projection.weight.grad is not None


def test_temporal_tokens_and_small_transformer_control():
    encoder = PatchTSTEncoder(d_model=16, layers=1, heads=4, feedforward_dim=32)
    x = torch.randn(4, 9, 128)
    assert encoder.forward_patch_tokens(x).shape == (4, 9, 16, 16)

    model = MultiTokenTransformerClassifier(encoder, hidden_size=16, dropout=0.0)
    sensor_embedding, tokens = model.encode_context_input(x)
    hidden = model.contextualize_sensor_token(tokens)
    loss = model.head(hidden).square().mean()
    loss.backward()

    assert sensor_embedding.shape == (4, 9 * 16)
    assert tokens.shape == (4, 16, 16)
    assert hidden.shape == (4, 16)
    assert model.projector.weight.grad is not None
    assert encoder.patch_projection.weight.grad is not None


def test_continuous_model_accepts_multiple_sensor_tokens():
    encoder = PatchTSTEncoder(d_model=16, layers=1, heads=4, feedforward_dim=32)
    model = ContinuousTokenClassifier(
        encoder,
        TinyFrozenLanguageModel(hidden_size=16),
        prefix_ids=torch.empty(0, dtype=torch.long),
        suffix_ids=torch.tensor([3, 4]),
        temporal_tokens=True,
    )
    sensor_embedding, tokens = model.encode_context_input(torch.randn(3, 9, 128))
    logits = model.classify_sensor_token(tokens)
    logits.square().mean().backward()

    assert sensor_embedding.shape == (3, 9 * 16)
    assert tokens.shape == (3, 16, 16)
    assert logits.shape == (3, 6)
    assert model.projector.weight.grad is not None

    channel_model = ContinuousTokenClassifier(
        encoder,
        TinyFrozenLanguageModel(hidden_size=16),
        prefix_ids=torch.empty(0, dtype=torch.long),
        suffix_ids=torch.tensor([3, 4]),
        channel_tokens=True,
    )
    _, channel_tokens = channel_model.encode_context_input(torch.randn(3, 9, 128))
    assert channel_tokens.shape == (3, 9, 16)
