import torch
from torch.nn import functional as F


def negative_cosine_similarity(prediction, target):
    prediction = F.normalize(prediction, dim=1)
    target = F.normalize(target.detach(), dim=1)
    return 2 - 2 * (prediction * target).sum(dim=1).mean()


def nt_xent(first, second, temperature):
    batch_size = first.shape[0]
    features = F.normalize(torch.cat((first, second), dim=0), dim=1)
    logits = features @ features.T / temperature
    logits.fill_diagonal_(torch.finfo(logits.dtype).min)
    targets = torch.arange(2 * batch_size, device=features.device)
    targets = (targets + batch_size) % (2 * batch_size)
    return F.cross_entropy(logits, targets)


def correlation(first, second):
    first = first.flatten(start_dim=1)
    second = second.flatten(start_dim=1)
    first = first - first.mean(dim=1, keepdim=True)
    second = second - second.mean(dim=1, keepdim=True)
    numerator = (first * second).sum(dim=1)
    denominator = first.norm(dim=1) * second.norm(dim=1)
    return (numerator / denominator.clamp_min(1e-8)).mean()


def random_sensor_augmentation(x):
    """Apply two randomly chosen, label-preserving sensor transformations."""
    result = x.clone()
    operations = torch.randperm(7)[:2].tolist()
    for operation in operations:
        if operation == 0:
            result = result + 0.05 * torch.randn_like(result)
        elif operation == 1:
            scale = torch.empty(result.shape[0], result.shape[1], 1, device=result.device)
            result = result * scale.uniform_(0.8, 1.2)
        elif operation == 2:
            result = result.flip(dims=(2,))
        elif operation == 3:
            shift = int(torch.randint(-16, 17, (1,), device=result.device))
            result = torch.roll(result, shifts=shift, dims=2)
        elif operation == 4:
            start = int(torch.randint(0, result.shape[2] - 16 + 1, (1,), device=result.device))
            result[:, :, start : start + 16] = 0
        elif operation == 5:
            channel = int(torch.randint(0, result.shape[1], (1,), device=result.device))
            result[:, channel] = 0
        else:
            chunks = result.chunk(4, dim=2)
            order = torch.randperm(4, device=result.device).tolist()
            result = torch.cat([chunks[index] for index in order], dim=2)
    return result

