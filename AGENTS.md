# Repository Guidelines

## PyTorch Models and Training Code

- Do not use, install into, activate, or modify any Conda environment.
- Use the repository-local environment managed by `uv`: run `uv sync` for setup and `uv run ...` for Python, tests, and training.
- Prefer the simplest implementation that is correct and easy to read.
- Avoid unnecessary abstractions, frameworks, helper layers, configuration systems, and clever metaprogramming.
- Use standard, idiomatic PyTorch (`torch` and `torch.nn`) unless the task requires another dependency.
- Keep model definitions small and explicit. Make tensor shapes and the expected input/output contract clear.
- Keep training and evaluation loops straightforward: move data and models to the selected device, zero gradients, compute the loss, backpropagate, and update parameters in an obvious order.
- Use `model.train()` for training and `model.eval()` with `torch.no_grad()` or `torch.inference_mode()` for evaluation.
- Match output activations, target formats, and loss functions correctly. For example, pass raw logits to `CrossEntropyLoss` rather than applying softmax first.
- Favor correctness and clarity over premature optimization. Add complexity only when it solves a demonstrated requirement.
- Handle reproducibility, checkpointing, metrics, and data loading plainly and only to the extent required by the task.
- When changing model or training code, run a small smoke test that exercises a forward pass and, when practical, one training step.
