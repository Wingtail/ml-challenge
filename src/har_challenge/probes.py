"""Linear probes before and after each frozen SmolLM2 layer."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .context import ContextClassifier
from .data import load_uci_har
from .evaluation import classification_metrics
from .patchtst import PatchTSTEncoder
from ._training import set_seed


def extract_representations(model, windows, batch_size, device):
    loader = DataLoader(TensorDataset(windows), batch_size=batch_size)
    sensor_parts, projected_parts, layer_parts = [], [], None
    model.eval()
    with torch.inference_mode():
        for (x,) in loader:
            sensor = model.encoder(x.to(device))
            projected = model.project(sensor)
            layers = model.contextualize(projected, return_all_layers=True)
            sensor_parts.append(sensor.cpu())
            projected_parts.append(projected.cpu())
            if layer_parts is None:
                layer_parts = [[] for _ in layers]
            for index, layer in enumerate(layers):
                layer_parts[index].append(layer.to(torch.float16).cpu())
    return {
        "sensor": torch.cat(sensor_parts),
        "projected": torch.cat(projected_parts),
        "layers": torch.stack([torch.cat(parts) for parts in layer_parts], dim=1),
    }


class ParallelLinearProbes(nn.Module):
    """One independent linear classifier for each representation layer."""

    def __init__(self, layers, features):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(layers, features, 6))
        self.bias = nn.Parameter(torch.zeros(layers, 6))

    def forward(self, x):
        return torch.einsum("blf,lfc->blc", x, self.weight) + self.bias


def fit_linear_probes(
    train_x,
    train_y,
    test_x,
    test_y,
    *,
    epochs,
    batch_size,
    learning_rate,
    device,
):
    if train_x.ndim == 2:
        train_x = train_x.unsqueeze(1)
        test_x = test_x.unsqueeze(1)
    model = ParallelLinearProbes(train_x.shape[1], train_x.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=0.0)
    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True
    )
    loss_function = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for features, labels in loader:
            features = features.to(device, dtype=torch.float32)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            repeated_labels = labels[:, None].expand(-1, logits.shape[1])
            loss = loss_function(logits.flatten(0, 1), repeated_labels.flatten())
            loss.backward()
            optimizer.step()

    predictions = []
    model.eval()
    loader = DataLoader(TensorDataset(test_x), batch_size=batch_size)
    with torch.inference_mode():
        for (features,) in loader:
            logits = model(features.to(device, dtype=torch.float32))
            predictions.append(logits.argmax(dim=-1).cpu())
    predictions = torch.cat(predictions)
    return [
        classification_metrics(test_y.tolist(), predictions[:, index].tolist())
        for index in range(predictions.shape[1])
    ]


def run_layer_probes(
    checkpoint_root,
    output_dir,
    *,
    data_root="dataset/UCI HAR Dataset",
    seeds=(42, 43, 44, 45),
    extract_batch_size=64,
    probe_batch_size=512,
    probe_epochs=100,
    probe_learning_rate=1e-2,
    device="cuda",
):
    """Run fixed-representation probes for each trained context checkpoint."""
    data = load_uci_har(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        set_seed(seed)
        model = ContextClassifier(PatchTSTEncoder()).to(device)
        checkpoint = Path(checkpoint_root) / f"seed_{seed}" / "model.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.encoder.load_state_dict(state["encoder"])
        model.projector.load_state_dict(state["projector"])
        model.head.load_state_dict(state["head"])

        train = extract_representations(
            model, data["x_train"], extract_batch_size, device
        )
        test = extract_representations(model, data["x_test"], extract_batch_size, device)
        probe_options = {
            "epochs": probe_epochs,
            "batch_size": probe_batch_size,
            "learning_rate": probe_learning_rate,
            "device": device,
        }
        sensor = fit_linear_probes(
            train["sensor"], data["y_train"], test["sensor"], data["y_test"],
            **probe_options,
        )[0]
        projected = fit_linear_probes(
            train["projected"], data["y_train"], test["projected"], data["y_test"],
            **probe_options,
        )[0]
        layers = fit_linear_probes(
            train["layers"], data["y_train"], test["layers"], data["y_test"],
            **probe_options,
        )
        result = {
            "seed": seed,
            "checkpoint": str(checkpoint),
            "sensor_embedding": sensor,
            "projected_token": projected,
            "llm_layers": layers,
        }
        (output_dir / f"seed_{seed}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        layer_scores = [100 * layer["macro_f1"] for layer in layers]
        best = max(range(len(layer_scores)), key=layer_scores.__getitem__)
        print(
            f"seed={seed} projected={100 * projected['macro_f1']:.2f} "
            f"best_layer={best} f1={layer_scores[best]:.2f}"
        )
