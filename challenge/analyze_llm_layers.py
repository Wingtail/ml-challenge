"""Linear-probe sensor, projector, and every frozen-LLM layer representation."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from challenge.data import load_challenge_data
from challenge.models import ContinuousTokenClassifier
from challenge.train import (
    load_context_state_dict,
    make_encoder,
    metrics_from_predictions,
    set_seed,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/UCI HAR Dataset")
    parser.add_argument("--checkpoint-root", default="outputs/submission_compliant/context/masked")
    parser.add_argument("--output-dir", default="outputs/llm_ablation_exact/layer_probes")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44, 45))
    parser.add_argument("--extract-batch-size", type=int, default=64)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def extract_representations(model, x, batch_size, device):
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    sensor_parts = []
    projected_parts = []
    layer_parts = None
    model.eval()
    with torch.inference_mode():
        for (batch,) in loader:
            sensor, projected = model.encode_context_input(batch.to(device))
            layers = model.contextualize_all_layers(projected)
            sensor_parts.append(sensor.cpu())
            projected_parts.append(projected.cpu())
            if layer_parts is None:
                layer_parts = [[] for _ in layers]
            for index, hidden in enumerate(layers):
                layer_parts[index].append(hidden.to(dtype=torch.float16).cpu())
    return {
        "sensor": torch.cat(sensor_parts),
        "projected": torch.cat(projected_parts),
        "layers": torch.stack(
            [torch.cat(parts) for parts in layer_parts], dim=1
        ),
    }


class ParallelLinearProbes(nn.Module):
    """One independent linear classifier for each representation layer."""

    def __init__(self, layer_count, feature_size, classes=6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(layer_count, feature_size, classes))
        self.bias = nn.Parameter(torch.zeros(layer_count, classes))

    def forward(self, features):
        return torch.einsum("bld,ldc->blc", features, self.weight) + self.bias


def fit_linear_probes(train_x, train_y, test_x, test_y, args):
    if train_x.ndim == 2:
        train_x = train_x.unsqueeze(1)
        test_x = test_x.unsqueeze(1)
    model = ParallelLinearProbes(train_x.shape[1], train_x.shape[2]).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.probe_learning_rate, weight_decay=0.0
    )
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.probe_batch_size,
        shuffle=True,
    )
    criterion = nn.CrossEntropyLoss()
    for _ in range(args.probe_epochs):
        model.train()
        for features, labels in loader:
            features = features.to(args.device, dtype=torch.float32)
            labels = labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            targets = labels[:, None].expand(-1, logits.shape[1])
            loss = criterion(logits.flatten(0, 1), targets.flatten())
            loss.backward()
            optimizer.step()

    predictions = []
    model.eval()
    test_loader = DataLoader(
        TensorDataset(test_x), batch_size=args.probe_batch_size, shuffle=False
    )
    with torch.inference_mode():
        for (features,) in test_loader:
            logits = model(features.to(args.device, dtype=torch.float32))
            predictions.append(logits.argmax(dim=-1).cpu())
    predictions = torch.cat(predictions, dim=0)
    return [
        metrics_from_predictions(test_y.tolist(), predictions[:, layer].tolist())
        for layer in range(predictions.shape[1])
    ]


def run_seed(seed, data, args):
    set_seed(seed)
    model = ContinuousTokenClassifier.from_pretrained(
        make_encoder(0.1),
        model_name=args.model,
        dtype=torch.bfloat16,
        prompt_style="challenge",
    ).to(args.device)
    checkpoint_path = Path(args.checkpoint_root) / f"seed_{seed}" / "context.pt"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    load_context_state_dict(model, state)

    train = extract_representations(
        model, data["x_train"], args.extract_batch_size, args.device
    )
    test = extract_representations(
        model, data["x_test"], args.extract_batch_size, args.device
    )
    sensor = fit_linear_probes(
        train["sensor"], data["y_train"], test["sensor"], data["y_test"], args
    )[0]
    projected = fit_linear_probes(
        train["projected"],
        data["y_train"],
        test["projected"],
        data["y_test"],
        args,
    )[0]
    layers = fit_linear_probes(
        train["layers"], data["y_train"], test["layers"], data["y_test"], args
    )
    result = {
        "seed": seed,
        "model": args.model,
        "prompt_style": "challenge",
        "checkpoint": str(checkpoint_path),
        "probe_training_windows": len(data["x_train"]),
        "probe_test_windows": len(data["x_test"]),
        "probe_epochs": args.probe_epochs,
        "probe_learning_rate": args.probe_learning_rate,
        "sensor_embedding": sensor,
        "projected_token": projected,
        "llm_layers": layers,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"seed_{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    args = parse_args(argv)
    data = load_challenge_data(args.data_root, validation_subjects=())
    for seed in args.seeds:
        result = run_seed(seed, data, args)
        layer_scores = [100 * item["macro_f1"] for item in result["llm_layers"]]
        best_layer = max(range(len(layer_scores)), key=layer_scores.__getitem__)
        print(
            f"seed={seed} sensor={100 * result['sensor_embedding']['macro_f1']:.2f} "
            f"projected={100 * result['projected_token']['macro_f1']:.2f} "
            f"best_llm_layer={best_layer} f1={layer_scores[best_layer]:.2f}"
        )


if __name__ == "__main__":
    main()
