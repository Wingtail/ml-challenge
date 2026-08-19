# Sensor Context Encoder Challenge

This repository tests whether one learned inertial-sensor embedding can be
consumed usefully by a frozen language model. The required submission compares
a direct PatchTST classifier, a frozen-SmolLM2 context model, and the same
context model after complete projected embeddings are shuffled between test
examples.

The README is the reproduction guide. Concise supporting evidence is retained
in [`VALIDATION.md`](VALIDATION.md) for grouped model selection and
[`EXPERIMENTS.md`](EXPERIMENTS.md) for the matched LLM diagnostics.

## Required results

All experiments use only the nine raw `128 x 9` signals in the UCI-HAR
`Inertial Signals` folders. The 561 engineered features are never loaded.

| Required condition | Seed 42 macro-F1 | Seeds 42--45 |
|---|---:|---:|
| Masked-pretrained PatchTST + linear head | **96.05** | **95.93 +/- 0.25** |
| PatchTST + projector + frozen pretrained SmolLM2 + head | 95.12 | 93.65 +/- 1.63 |
| Context model with projected embeddings shuffled | 16.40 | 16.48 +/- 0.06 |

The exact values and artifact locations are in
[`results/required_results.csv`](results/required_results.csv).

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and a repository-local
`.venv`; Install `uv` using its official
[installation instructions](https://docs.astral.sh/uv/getting-started/installation/),
then run:

```bash
uv sync --locked
```

`uv sync` installs a managed Python 3.11 interpreter when needed, creates the
isolated `.venv`, and installs the project and its locked dependencies. It does
not use or modify a Conda environment. Exact versions are recorded in
[`uv.lock`](uv.lock); broad project requirements remain in
[`pyproject.toml`](pyproject.toml).

## Dataset preparation

The repository includes the official 58.2 MB archive from the
[UCI Human Activity Recognition Using Smartphones dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones):

```text
dataset/UCI HAR Dataset.zip
```

Its SHA-256 checksum is:

```text
2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0
```

Extract it once from the repository root:

```bash
uv run python -m zipfile -e "dataset/UCI HAR Dataset.zip" dataset
```

This creates the directory expected by every training script:

```text
dataset/UCI HAR Dataset/
    train/Inertial Signals/
    test/Inertial Signals/
```

Verify the extracted files and official split before training:

```bash
uv run python -c "from har_challenge.data import load_uci_har; d = load_uci_har('dataset/UCI HAR Dataset'); print(d['x_train'].shape, d['x_test'].shape)"
```

The expected output is `torch.Size([7352, 9, 128])` and
`torch.Size([2947, 9, 128])`. Extraction may also create a harmless
`dataset/__MACOSX/` metadata directory; it is ignored by Git and by the loader.

After preparing the dataset, run the test suite:

```bash
uv run python -m pytest -q
```

The loader asserts the official 21-subject training and nine-subject test
partitions. It computes one mean and standard deviation per channel using only
the optimization subjects and applies that dataset-level Z-score transform to
validation and test windows. It reads only the nine files under
`Inertial Signals`, plus labels and subject IDs; it never loads the supplied
561-feature `X_train.txt` or `X_test.txt` files.

## Reproduce the challenge results

The easiest way to reproduce every required condition for seed 42 is:

```bash
./scripts/reproduce_seed.sh
```

Pass another integer to reproduce one different seed, for example
`./scripts/reproduce_seed.sh 43`. The wrapper extracts the included dataset if
needed, then runs masked pretraining, direct training, context-model training,
and the shuffled-embedding evaluation in the correct order.

The individual Python entry points are documented below for readers who want
to run one stage. All paths and hyperparameters are plain constants in
[`src/har_challenge/settings.py`](src/har_challenge/settings.py). The required
training functions accept only a seed; there is no YAML loader, configuration
framework, or diagnostic switch hidden in the main workflow.

### 0. Masked sensor pretraining

```bash
uv run python scripts/pretrain_encoder.py
```

This trains the PatchTST encoder for 100 epochs with 40% masked-patch MSE,
batch size 256, AdamW at `1e-3`, dropout 0.1, and zero weight decay. It writes
`outputs/reproduction/pretrain/seed_42/pretrained.pt`.

### 1. Direct sensor classifier

```bash
uv run python scripts/train_direct.py
```

The encoder and `Linear(1152, 6)` head train for 18 epochs with encoder LR
`1e-4`, head LR `1e-3`, batch size 128, and cross-entropy. Expected seed-42
macro-F1 is approximately **96.05**. Results are written under
`outputs/reproduction/direct/seed_42/`.

### 2. Frozen-LLM context model

```bash
uv run python scripts/train_context.py
```

The separately initialized encoder, `Linear(1152, 960)` projector, and
`Linear(960, 6)` head train for 18 epochs. SmolLM2-360M-Instruct and its token
embeddings remain frozen and in evaluation mode, but autograd passes through
the LLM into the projector and encoder. Expected seed-42 macro-F1 is
approximately **95.12**. Results are written under
`outputs/reproduction/context/seed_42/`.

The required configuration uses the challenge prompt literally. The native
SmolLM2 chat-template version is retained as a diagnostic experiment below.

### 3. Shuffled-embedding dependence check

```bash
uv run python scripts/eval_shuffle.py
```

This reloads the trained context checkpoint, computes all 2,947 projected test
embeddings once, permutes complete embeddings between examples, keeps labels
fixed, and does not retrain. Expected seed-42 macro-F1 is approximately
**16.40**. Results are written to
`outputs/reproduction/context/seed_42/shuffle.json`.

### Quick smoke run

```bash
uv run python scripts/smoke_test.py
```

Smoke scores are not comparable to the reported results.

### Four-seed reproduction

Reproduce the complete aggregate reported for seeds 42--45 with one command:

```bash
./scripts/reproduce_all.sh
```

The script stops immediately if any stage fails. Each seed has separate
checkpoint and result directories under `outputs/reproduction/`. At the end it
prints the three mean and sample-standard-deviation results and writes their
full-precision values to `outputs/reproduction/summary.json`.

For exact fixed-seed compatibility, the context model preserves the original
study's module-initialization order. This keeps the shuffled training batches
and the resulting checkpoints consistent with the reported runs.

## Architecture map

The editable submission implementation is intentionally small:

```text
src/har_challenge/
    settings.py          all editable paths and hyperparameters
    data.py              raw signals, official split, normalization
    patchtst.py          encoder, masked pretrainer, direct head
    context.py           fixed projector, prompt injection, frozen SmolLM2
    train.py             three required PyTorch training workflows
    _training.py         short mechanics shared by the training loops
    evaluation.py        metrics and embedding shuffle
    probes.py            frozen-representation linear probes
    diagnostic_context.py    optional backend and interface variations
    diagnostic_training.py   training for optional diagnostic variations

scripts/                 required runs, one-command wrappers, and smoke test
experiments/             five grouped diagnostic studies
results/                 compact CSV summaries
challenge/               retained original development-study code
har_benchmark/           separate SSL representation benchmark
```

The required files contain no random/identity backend selection, chat-prompt
selection, label-fraction logic, or projector selection. Those options are
isolated in the two clearly named `diagnostic_*.py` modules so they do not
obscure the submitted model.

PatchTST divides each channel into 16 non-overlapping length-eight patches,
uses a shared `Linear(8, 128)` embedding, three pre-norm Transformer layers,
four heads, and FFN width 256. Temporal mean pooling followed by channel
concatenation produces the 1,152-dimensional sensor embedding. The direct
model has 407,686 trainable parameters; the linear-projector LLM path has
1,513,414 trainable parameters, below the 10M limit.

## Additional diagnostic experiments

The required comparison establishes that the frozen-LLM system is worse than
the direct classifier, but it does not explain why. The diagnostics below test
two plausible alternatives before attributing the gap to the pretrained LLM:

1. The longer LLM pipeline may be harder to optimize, so its potential benefit
   could be hidden by poor gradients into the sensor encoder.
2. The sensor--LLM interface may be too weak or improperly formatted, so the
   experiment may not give SmolLM2 a fair opportunity to use the sensor token.

For clarity, let `E` denote the PatchTST encoder, `P` the projector, and `H`
the classification head. The tested context model is
`x -> E -> P -> frozen LLM -> H`. Unless stated otherwise, every diagnostic
uses the same four SSL checkpoints, official train/test subjects, seeds
42--45, 18-epoch schedule, batch size, optimizer, and learning rates. We change
one component at a time so that each comparison answers a specific question.

### Do pretrained language weights add useful structure?

We compare three otherwise matched backends:

| Backend | What happens after `P(E(x))` | Why this control is needed | Macro-F1 |
|---|---|---|---:|
| Pretrained SmolLM2 | The projected token passes through frozen language-pretrained SmolLM2 before `H` | System being evaluated | 93.65 +/- 1.63 |
| Random SmolLM2 | It passes through the same frozen architecture with randomly initialized weights | Separates language pretraining from the effect of a large fixed transformation | 95.40 +/- 0.38 |
| Identity/no LLM | It goes directly to `H`; the `1152 -> 960` projector is retained | Tests whether applying any LLM transformation is better than using the projected sensor token itself | **95.42 +/- 1.44** |

In all three cases, `E`, `P`, and `H` are trained; both LLM variants remain
frozen. The random model controls for architecture and parameter count, while
the identity model removes the LLM but retains the projector and head. The
identity control is therefore not the smaller direct PatchTST classifier.
Pretrained SmolLM2 beats neither matched control. The sensor-shuffle result
still shows that the model uses the matching sensor input, but these controls
show that its accuracy does not come from useful language-pretrained structure.

Reproduce the two controls with:

```bash
for seed in 42 43 44 45; do
  uv run python experiments/backend_controls.py "$seed"
done
```

### Is end-to-end encoder optimization hiding an LLM benefit?

In the normal experiment, the classification gradient passes backward through
the frozen 360M-parameter LLM before reaching `P` and `E`. Although the LLM
weights do not update, this long gradient path could still train the sensor
encoder poorly. To remove that possibility, we repeat all three backends with
`E` fixed at its masked-pretrained checkpoint and train only `P` and `H`.

| Encoder setting | Pretrained LLM | Random LLM | Identity/no LLM |
|---|---:|---:|---:|
| Fine-tuned `E` | 93.65 +/- 1.63 | 95.40 +/- 0.38 | **95.42 +/- 1.44** |
| Frozen `E` | 91.06 +/- 1.25 | 91.86 +/- 0.54 | **92.36 +/- 0.54** |

Freezing `E` makes every backend worse. End-to-end adaptation therefore helps
rather than distracting or overfitting the sensor encoder. More importantly,
pretrained SmolLM2 remains behind the random and identity controls even after
encoder-gradient difficulty is removed. Optimization affects absolute
performance and stability, but it does not hide a pretrained-LLM advantage in
this protocol.

Reproduce the frozen-encoder comparison with:

```bash
for seed in 42 43 44 45; do
  uv run python experiments/frozen_encoder.py "$seed"
done
```

### What happens to the activity information inside SmolLM2?

A lower end-to-end score could mean either that SmolLM2 destroys the activity
signal or that it keeps the signal but makes it harder for the final linear
head to access. We distinguish these possibilities with linear probes. After
training the context model, we freeze `E`, `P`, and SmolLM2, extract a fixed
representation for every window, and train a fresh linear classifier at each
location. The probe before the LLM sees `P(E(x))`; the other probes see the
final `Activity:` token after each frozen LLM layer. The probes do not update
the original model.

| Fixed representation given to a fresh linear probe | Macro-F1 |
|---|---:|
| Projected sensor token before SmolLM2 | **95.08 +/- 0.97** |
| Best SmolLM2 intermediate layer | 94.26 +/- 0.77 |
| Final SmolLM2 layer | 93.80 +/- 1.16 |

The activity remains recoverable after the LLM, so SmolLM2 does not erase the
sensor information. However, no LLM layer improves on the representation that
entered it, and the final layer is 1.28 points worse (`p = 0.023`). Under this
interface, the language-trained transformation makes the existing activity
information less linearly accessible rather than adding useful class
structure.

Reproduce the probes with:

```bash
uv run python experiments/layer_probes.py
```

The fresh probes use Adam at `1e-2`, batch size 512, 100 epochs, and no weight
decay. Their purpose is diagnosis, not improving the submitted model.

### Supporting robustness checks

The following experiments test narrower ways in which the original comparison
might have disadvantaged the pretrained LLM.

**Low-label supervised learning.** Language pretraining might be useful when
labels are scarce even if it does not improve the full-data result. We freeze
the same SSL sensor encoder and train `P` and `H` with 1%, 5%, 10%, 25%, 50%,
or 100% of the labeled training windows, sampling within each activity class.
Pretrained SmolLM2 never beats random SmolLM2 at any tested fraction. These are
ordinary supervised low-label runs—not in-context few-shot learning—because a
new projector and head are still optimized for every condition.

```bash
for seed in 42 43 44 45; do
  uv run python experiments/low_label.py "$seed"
done
```

**Projector capacity.** A single linear map might be too weak to align sensor
features with SmolLM2's token space. Replacing it with
`Linear(1152,960) -> GELU -> Linear(960,960)` raises pretrained-LLM performance
from 93.65 to 94.48 macro-F1. This shows that interface capacity matters, but
it does not close the gap to the 95.40 random-LLM or 95.42 identity controls.

**Prompt formatting.** The required result uses the challenge prompt
literally, whereas SmolLM2-Instruct was trained with a chat template. We render
the same instruction as a native user message, preserve one `<SENSOR>` slot,
and use `Activity:` as the unfinished assistant response. Chat formatting
raises the pretrained result to 94.21 +/- 0.71, but a matched random SmolLM2
with the same chat sequence reaches 95.25 +/- 0.37. Correct formatting improves
stability but does not expose an advantage from language pretraining.

```bash
for seed in 42 43 44 45; do
  uv run python experiments/interface_checks.py "$seed"
done
```

**Representation rank.** We also tested whether an effectively low-rank sensor
embedding was an unfriendly input for the LLM. VICReg-style variance and
covariance regularizers successfully increased effective rank—for example,
VIbCReg-lite raised the encoder rank from 8.58 to 60.23—but did not improve
classification and in that case reduced macro-F1. Effective rank is therefore
a useful collapse diagnostic, not evidence by itself that an embedding
contains activity-relevant semantics. This earlier study used a different
SmolLM2 size and training schedule, so it is supporting evidence rather than a
matched component of the final 360M comparison.

Together, these experiments address the main alternative explanations: better
encoder optimization, a larger projector, conventional chat formatting, more
effective rank, and fewer labels do not reveal a pretrained-LLM advantage. The
complete low-label matrix and all four-seed summaries are in
[`results/diagnostic_results.csv`](results/diagnostic_results.csv). The full
experimental argument is summarized in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Validation and model selection

Hyperparameters were selected with five-fold `StratifiedGroupKFold` over the
21 official training subjects. Subject grouping prevents participant and
overlapping-window leakage; stratification keeps all six activities reasonably
balanced. The selected encoder LR is `1e-4`, the new-layer LR is `1e-3`, and
18 epochs is the median of the five fold-best epochs (`29, 6, 10, 21, 18`).
The official test split was used only after selection in the corrected
protocol. Fold definitions and tuning evidence remain in `outputs/group_cv/`
and [`VALIDATION.md`](VALIDATION.md).

Macro-F1 is primary because it computes F1 separately for every activity and
then gives all six equal weight. This exposes minority-class and
sitting-versus-standing failures that accuracy or micro-F1 can conceal.

## Preserved study scope

The repository retains the experiments needed to distinguish two explanations:

1. End-to-end optimization through a frozen 360M-parameter LLM hides a useful
   pretrained representation.
2. The frozen language-trained transformation adds no classification-relevant
   structure for this single-sensor closed-set task.

Fine-tuning improves all backends, so encoder adaptation is beneficial rather
than harmful. Pretrained SmolLM2 nevertheless trails random and identity
controls with both trainable and frozen encoders. Layer probes show that the
activity signal remains present but becomes less linearly accessible through
the LLM. The sensor shuffle proves dependence, not usefulness of language
pretraining.

The proposed next study is prototype-conditioned in-context few-shot
classification with medoid and boundary support examples and matched centroid,
linear, small-Transformer, random-LLM, and identity controls. The longer-term
hypothesis is multimodal context, where motion is one token beside orthogonal
image, environment, behavior, health, or metadata tokens. Neither is claimed
as a completed result.

Four seeds provide limited statistical power, and the official test set was
inspected during earlier exploration. The ablations should therefore be read
as diagnostic evidence rather than a pristine one-shot benchmark.

## Archived research notes

The other development-era Markdown reports were consolidated into
`backups/historical_markdown_before_cleanup_20260820.tar.gz`. The two documents
that directly support this README remain as `VALIDATION.md` and
`EXPERIMENTS.md`; current numerical summaries remain in `results/`, and the
original checkpoints and machine-readable artifacts remain in `outputs/`.
