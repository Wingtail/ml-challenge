import inspect
from types import SimpleNamespace

import torch

from har_challenge.context import ContextClassifier, PROMPT_BEFORE_SENSOR
from har_challenge.data import load_uci_har
from har_challenge.diagnostic_context import (
    IdentityContextClassifier,
    chat_prompt_ids,
)
from har_challenge.patchtst import (
    DirectClassifier,
    MaskedPatchPretrainer,
    PatchTSTEncoder,
)
from har_challenge.train import train_context, train_direct


def test_refactored_loader_uses_only_official_raw_signal_shapes():
    data = load_uci_har("dataset/UCI HAR Dataset")
    assert data["x_train"].shape == (7352, 9, 128)
    assert data["x_test"].shape == (2947, 9, 128)
    assert set(data["subjects_train"].tolist()).isdisjoint(
        data["subjects_test"].tolist()
    )


def test_refactored_model_parameter_counts_match_the_study():
    direct = DirectClassifier(PatchTSTEncoder())
    identity = IdentityContextClassifier(
        PatchTSTEncoder(), torch.nn.Linear(1152, 960)
    )
    context_trainable = sum(
        parameter.numel()
        for parameter in identity.parameters()
        if parameter.requires_grad
    )
    assert sum(parameter.numel() for parameter in direct.parameters()) == 407_686
    assert context_trainable == 1_513_414
    assert not any(
        parameter.requires_grad
        for parameter in identity._study_rng_compatibility_readout.parameters()
    )


def test_refactored_pretrainer_and_identity_backend_backpropagate():
    x = torch.randn(2, 9, 128)
    pretrainer = MaskedPatchPretrainer(PatchTSTEncoder())
    pretrainer(x).backward()
    assert pretrainer.encoder.patch_projection.weight.grad is not None

    model = IdentityContextClassifier(
        PatchTSTEncoder(), torch.nn.Linear(1152, 960)
    )
    model(x).square().mean().backward()
    assert model.encoder.patch_projection.weight.grad is not None
    assert model.projector.weight.grad is not None
    assert model.head.weight.grad is not None


def test_chat_template_preserves_one_sensor_slot_and_activity_prefill():
    class FakeTokenizer:
        def __init__(self):
            self.parts = []

        def apply_chat_template(self, messages, tokenize, continue_final_message):
            assert tokenize is False
            assert continue_final_message is True
            return (
                "<|im_start|>user\n"
                + messages[0]["content"]
                + "<|im_end|>\n<|im_start|>assistant\n"
                + messages[1]["content"]
            )

        def __call__(self, text, add_special_tokens, return_tensors):
            assert add_special_tokens is False
            assert return_tensors == "pt"
            self.parts.append(text)
            return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))

    tokenizer = FakeTokenizer()
    before_ids, after_ids = chat_prompt_ids(tokenizer)
    assert tokenizer.parts[0].endswith(PROMPT_BEFORE_SENSOR)
    assert tokenizer.parts[1].endswith("Activity:")
    assert before_ids.tolist() == after_ids.tolist() == [1, 2]


def test_required_training_functions_expose_only_seed_and_smoke_mode():
    direct = inspect.signature(train_direct).parameters
    context = inspect.signature(train_context).parameters
    assert list(direct) == ["seed", "smoke"]
    assert list(context) == ["seed", "smoke"]
    assert direct["seed"].default == context["seed"].default == 42
