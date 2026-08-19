# Exact Ablations for the Frozen-LLM Direction

## Decision

For this challenge, the frozen-LLM branch is not worth continuing. The exact
matched controls show that pretrained SmolLM2 does not improve classification,
label efficiency, or linear separability over simpler alternatives. This is a
decision about the tested frozen-LLM approach on UCI HAR, not a universal claim
that language models can never help sensor learning.

## Exact protocol

All new runs use:

- the nine raw `128 x 9` inertial signals and no engineered features;
- the official subject-disjoint train/test split;
- the four masked-pretrained PatchTST checkpoints from seeds 42--45;
- `HuggingFaceTB/SmolLM2-360M-Instruct` and the specified challenge prompt;
- 18 epochs, batch size 128, dropout 0.1, and no weight decay;
- projector/head learning rate `1e-3` and encoder learning rate `1e-4` when
  the encoder is trainable;
- no auxiliary sensor loss and no representation regularizer;
- a completely frozen language model, with gradients propagated through it.

These are diagnostic comparisons on a test set examined earlier in the
project. They should not be presented as a pristine new test estimate.

## Matched full-label controls

Test macro-F1, mean +/- sample standard deviation over four seeds:

| Readout | Trainable parameters | Macro-F1 |
|---|---:|---:|
| Direct PatchTST head | 407,686 | **95.93 +/- 0.25** |
| Linear projector + pretrained LLM | 1,513,414 | 93.65 +/- 1.63 |
| Linear projector + random LLM | 1,513,414 | 95.40 +/- 0.38 |
| Linear projector + identity | 1,513,414 | 95.42 +/- 1.44 |
| Linear projector + GELU | 1,513,414 | 93.30 +/- 1.65 |
| Nonlinear projector + pretrained LLM | 2,435,974 | 94.48 +/- 0.56 |

The nonlinear projector recovers 0.83 points relative to the linear
pretrained-LLM interface. Projector alignment therefore explains part of the
failure, but not enough to beat the parameter-matched random or identity
controls. Pretrained SmolLM2 trails the random model by 1.74 points. The
four-seed confidence interval is wide (`-4.29` to `+0.80`, paired `p=0.117`),
so that single comparison is not conclusive by itself.

## Information-survival and layer probes

For each trained linear-interface checkpoint, the encoder, projector, and LLM
were frozen. Fresh linear probes were trained on the official training
representations and evaluated on official test subjects.

| Representation | Macro-F1 |
|---|---:|
| Encoder embedding | 94.94 +/- 0.96 |
| Projected token before LLM | **95.08 +/- 0.97** |
| Best fixed intermediate layer by mean, layer 10 | 94.26 +/- 0.77 |
| Final LLM layer | 93.80 +/- 1.16 |

Activity information enters the prompt state quickly and remains recoverable,
so the LLM does not simply erase the signal. However, every fixed LLM layer is
worse than probing the projected token directly. The projected token exceeds
layer 10 by 0.82 points (`p=0.046`) and the final layer by 1.28 points
(`p=0.023`). This is evidence that the frozen transformation entangles some
task-relevant information rather than adding useful class structure.

## Exact low-label curve

The masked-pretrained sensor encoder is frozen here. Only the projector and
head train from the stated fraction of class-stratified labeled windows.

| Labels | Pretrained LLM | Random LLM | Identity |
|---:|---:|---:|---:|
| 1% | 48.74 +/- 32.86 | **85.55 +/- 2.28** | 84.95 +/- 2.60 |
| 5% | 81.29 +/- 9.97 | 87.57 +/- 2.18 | **88.81 +/- 0.55** |
| 10% | 88.71 +/- 1.46 | **90.54 +/- 0.39** | 90.08 +/- 0.88 |
| 25% | 90.53 +/- 1.62 | **91.09 +/- 0.46** | 90.54 +/- 1.41 |
| 50% | 89.90 +/- 1.46 | **91.87 +/- 0.37** | 90.90 +/- 0.82 |
| 100% | 91.06 +/- 1.25 | 91.86 +/- 0.54 | **92.36 +/- 0.54** |

The pretrained model never beats the random model. Its paired disadvantages
are 1.83 points at 10% (`p=0.046`) and 1.96 points at 50% (`p=0.049`). At 1%,
the pretrained interface is highly unstable across seeds, scoring 65.67,
8.16, 37.93, and 83.22 macro-F1.

## Scope of the conclusion

The official test split already measures generalization to unseen subjects,
but it is not an external sensor domain or a semantic transfer task. A claim
that LLMs improve cross-dataset semantic transfer would require an untouched
second dataset with a predeclared label mapping and evaluation protocol.

LoRA was not tested because it would violate the challenge instruction that
only the sensor encoder, projector, and classification head may train. It can
answer a broader research question about whether freezing is the bottleneck,
but it cannot rescue the challenge-compliant frozen-LLM approach.

## Artifacts

- Implementation: `src/har_challenge/` and the entry points in `scripts/`
- Layer probes: `src/har_challenge/probes.py` and `experiments/layer_probes.py`
- Audit and statistics: `challenge/analyze_llm_ablation_exact.py`
- Machine-readable summary: `outputs/llm_ablation_exact/summary.json`
- Individual checkpoints, metrics, and logs: `outputs/llm_ablation_exact/`
