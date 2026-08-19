import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from challenge.data import load_challenge_data
from challenge.models import ContinuousTokenClassifier
from challenge.train import load_context_state_dict, make_encoder, set_seed


CONTEXT_CHECKPOINTS = {
    "random": "outputs/challenge/random/seed_42/context.pt",
    "masked": "outputs/challenge/masked/seed_42/context.pt",
    "drop_patch": "outputs/challenge/drop_patch/seed_42/context.pt",
    "drop_patch_tuned": "outputs/tuned/drop_patch/seed_42/context.pt",
}

PRETRAINED_CHECKPOINTS = {
    "masked": "outputs/challenge/masked/seed_42/pretrained.pt",
    "drop_patch": "outputs/challenge/drop_patch/seed_42/pretrained.pt",
    "drop_patch_tuned": "outputs/tuned/drop_patch/seed_42/pretrained.pt",
}

CONTEXT_TEST_F1 = {
    "random": 0.9356803798953422,
    "masked": 0.9513800410682146,
    "drop_patch": 0.893335399138565,
    "drop_patch_tuned": 0.8961185264871948,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/UCI HAR Dataset")
    parser.add_argument("--llm-model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--output", default="outputs/embedding_geometry.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def group_variance_fraction(x: torch.Tensor, groups: torch.Tensor):
    centered = x - x.mean(dim=0)
    total = centered.square().sum().clamp_min(1e-12)
    between = x.new_tensor(0.0)
    for group in torch.unique(groups):
        members = x[groups == group]
        between += len(members) * (members.mean(dim=0) - x.mean(dim=0)).square().sum()
    return (between / total).item()


def geometry_metrics(x: torch.Tensor, labels=None, subjects=None):
    """Measure directional anisotropy and covariance dimensionality."""
    x = x.float().cpu()
    count, dimensions = x.shape
    norms = x.norm(dim=1)
    unit = x / norms.unsqueeze(1).clamp_min(1e-12)

    generator = torch.Generator().manual_seed(12345)
    pairs = min(200_000, count * 100)
    first = torch.randint(count, (pairs,), generator=generator)
    second = torch.randint(count, (pairs,), generator=generator)
    pairwise_cosine = (unit[first] * unit[second]).sum(dim=1)

    centered = x - x.mean(dim=0)
    singular_values = torch.linalg.svdvals(centered)
    variance = singular_values.square()
    spectrum = variance / variance.sum().clamp_min(1e-12)
    nonzero = spectrum[spectrum > 0]
    effective_rank = torch.exp(-(nonzero * nonzero.log()).sum()).item()
    participation_ratio = (1 / spectrum.square().sum()).item()

    metrics = {
        "samples": count,
        "dimensions": dimensions,
        "norm_mean": norms.mean().item(),
        "norm_std": norms.std(unbiased=False).item(),
        "centroid_norm_over_mean_norm": (x.mean(dim=0).norm() / norms.mean()).item(),
        "pairwise_cosine_mean": pairwise_cosine.mean().item(),
        "pairwise_cosine_std": pairwise_cosine.std(unbiased=False).item(),
        "pairwise_absolute_cosine_mean": pairwise_cosine.abs().mean().item(),
        "effective_rank": effective_rank,
        "effective_rank_fraction": effective_rank / dimensions,
        "participation_ratio": participation_ratio,
        "top_1_variance_fraction": spectrum[0].item(),
        "top_10_variance_fraction": spectrum[:10].sum().item(),
        "top_50_variance_fraction": spectrum[:50].sum().item(),
    }
    if labels is not None:
        metrics["activity_variance_fraction"] = group_variance_fraction(x, labels.cpu())
    if subjects is not None:
        metrics["subject_variance_fraction"] = group_variance_fraction(x, subjects.cpu())
    return metrics


def load_pretrained_encoder(path, device):
    encoder = make_encoder(0.1)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(checkpoint["encoder"])
    return encoder.to(device).eval()


def encode_all(encoder, x, batch_size, device):
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            embeddings.append(encoder(x[start : start + batch_size].to(device)).cpu())
    return torch.cat(embeddings)


def context_representations(model, x, batch_size, device):
    sensor_embeddings = []
    projected_tokens = []
    final_hidden_states = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = x[start : start + batch_size].to(device)
            sensor = model.encoder(batch)
            token = model.encode_sensor_token(batch)
            hidden = model.contextualize_sensor_token(token)
            sensor_embeddings.append(sensor.cpu())
            projected_tokens.append(token.cpu())
            final_hidden_states.append(hidden.cpu())
    return {
        "sensor_embedding": torch.cat(sensor_embeddings),
        "projected_token": torch.cat(projected_tokens),
        "final_hidden_state": torch.cat(final_hidden_states),
    }


def vocabulary_alignment(sensor_tokens, vocabulary_embeddings, batch_size=256):
    sensor = sensor_tokens.float()
    vocabulary = vocabulary_embeddings.float().cpu()
    sensor_unit = sensor / sensor.norm(dim=1, keepdim=True).clamp_min(1e-12)
    vocabulary_unit = vocabulary / vocabulary.norm(dim=1, keepdim=True).clamp_min(1e-12)
    maximum_cosines = []
    for start in range(0, len(sensor), batch_size):
        cosine = sensor_unit[start : start + batch_size] @ vocabulary_unit.T
        maximum_cosines.append(cosine.max(dim=1).values)
    maximum_cosines = torch.cat(maximum_cosines)
    vocabulary_centroid = vocabulary.mean(dim=0)
    centroid_cosine = torch.nn.functional.cosine_similarity(
        sensor, vocabulary_centroid.unsqueeze(0), dim=1
    )
    return {
        "nearest_vocabulary_cosine_mean": maximum_cosines.mean().item(),
        "nearest_vocabulary_cosine_std": maximum_cosines.std(unbiased=False).item(),
        "vocabulary_centroid_cosine_mean": centroid_cosine.mean().item(),
        "vocabulary_centroid_cosine_std": centroid_cosine.std(unbiased=False).item(),
    }


def class_direction_alignment(first, first_labels, second, second_labels):
    """Compare centered class-centroid directions across subject splits."""
    first = first.float()
    second = second.float()
    first_center = first.mean(dim=0)
    second_center = second.mean(dim=0)
    first_directions = []
    second_directions = []
    for label in torch.unique(first_labels):
        first_directions.append(first[first_labels == label].mean(dim=0) - first_center)
        second_directions.append(second[second_labels == label].mean(dim=0) - second_center)
    first_directions = torch.stack(first_directions)
    second_directions = torch.stack(second_directions)
    first_directions = first_directions / first_directions.norm(dim=1, keepdim=True)
    second_directions = second_directions / second_directions.norm(dim=1, keepdim=True)
    cosine = first_directions @ second_directions.T
    matched = cosine.diag()
    wrong = cosine.masked_fill(torch.eye(len(cosine), dtype=torch.bool), -torch.inf)
    return {
        "matched_class_cosine_mean": matched.mean().item(),
        "matched_class_cosines": matched.tolist(),
        "class_matching_margin_mean": (matched - wrong.max(dim=1).values).mean().item(),
        "class_direction_retrieval_accuracy": (
            cosine.argmax(dim=1) == torch.arange(len(cosine))
        ).float().mean().item(),
    }


def classifier_metrics(head, hidden, labels):
    with torch.inference_mode():
        predictions = head(hidden).argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


def pair_effect_size(x, labels, first_label=3, second_label=4):
    """Cohen-like separation along the two class centroids' connecting axis."""
    first = x[labels == first_label].float()
    second = x[labels == second_label].float()
    direction = first.mean(dim=0) - second.mean(dim=0)
    direction = direction / direction.norm().clamp_min(1e-12)
    first_projection = first @ direction
    second_projection = second @ direction
    pooled_std = torch.sqrt(
        0.5
        * (
            first_projection.var(unbiased=False)
            + second_projection.var(unbiased=False)
        )
    ).clamp_min(1e-12)
    return ((first_projection.mean() - second_projection.mean()).abs() / pooled_std).item()


def main():
    args = parse_args()
    set_seed(42)
    data = load_challenge_data(args.data_root)
    x = data["x_test"]
    labels = data["y_test"]
    subjects = data["subjects_test"]
    results = {"pretrained_encoder": {}, "context_model": {}}

    random_encoder = make_encoder(0.1).to(args.device).eval()
    random_embedding = encode_all(random_encoder, x, args.batch_size, args.device)
    results["pretrained_encoder"]["random"] = geometry_metrics(
        random_embedding, labels, subjects
    )
    for name, path in PRETRAINED_CHECKPOINTS.items():
        encoder = load_pretrained_encoder(path, args.device)
        embedding = encode_all(encoder, x, args.batch_size, args.device)
        results["pretrained_encoder"][name] = geometry_metrics(
            embedding, labels, subjects
        )
        print(f"measured pretrained encoder: {name}")

    context_model = ContinuousTokenClassifier.from_pretrained(
        make_encoder(0.1), args.llm_model, dtype=torch.bfloat16
    ).to(args.device)
    vocabulary = context_model.language_model.get_input_embeddings().weight.detach().cpu()
    results["llm_vocabulary"] = geometry_metrics(vocabulary)

    for name, path in CONTEXT_CHECKPOINTS.items():
        state = torch.load(path, map_location="cpu", weights_only=True)
        load_context_state_dict(context_model, state)
        context_model.eval()
        representations = context_representations(
            context_model, x, args.batch_size, args.device
        )
        train_representations = context_representations(
            context_model, data["x_train"], args.batch_size, args.device
        )
        validation_representations = context_representations(
            context_model, data["x_validation"], args.batch_size, args.device
        )
        condition = {"test_macro_f1": CONTEXT_TEST_F1[name]}
        for stage, representation in representations.items():
            condition[stage] = geometry_metrics(representation, labels, subjects)
            condition[stage]["sitting_standing_effect_size"] = pair_effect_size(
                representation, labels
            )
        condition["projected_token"].update(
            vocabulary_alignment(representations["projected_token"], vocabulary)
        )
        condition["head_performance"] = {
            "train": classifier_metrics(
                context_model.head.cpu(),
                train_representations["final_hidden_state"],
                data["y_train"],
            ),
            "validation": classifier_metrics(
                context_model.head.cpu(),
                validation_representations["final_hidden_state"],
                data["y_validation"],
            ),
            "test": classifier_metrics(
                context_model.head.cpu(), representations["final_hidden_state"], labels
            ),
        }
        context_model.head.to(args.device)
        condition["split_alignment"] = {}
        for stage in representations:
            condition["split_alignment"][stage] = {
                "train_to_validation": class_direction_alignment(
                    train_representations[stage],
                    data["y_train"],
                    validation_representations[stage],
                    data["y_validation"],
                ),
                "train_to_test": class_direction_alignment(
                    train_representations[stage],
                    data["y_train"],
                    representations[stage],
                    labels,
                ),
            }
        results["context_model"][name] = condition
        print(f"measured context model: {name}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
