# Subject-Grouped Validation and Final Prompt Results

> **Development-stage tuning report.** This search used SmolLM2-135M. The
> selected optimizer recipe was subsequently locked and evaluated with the
> specified SmolLM2-360M-Instruct checkpoint and exact prompt. The final
> challenge results are summarized in `README.md`.

This experiment replaces the earlier single validation-subject holdout with
five-fold `StratifiedGroupKFold` model selection. It also separates the encoder
learning rate from the projector and classification-head learning rate, then
retests the prompt ablation after fitting on every official training subject.

## Dataset protocol

The data loader now asserts the canonical UCI-HAR partition:

- official training: 7,352 raw `9 x 128` windows from 21 subjects;
- official test: 2,947 raw `9 x 128` windows from subjects
  `2, 4, 9, 10, 12, 13, 18, 20, 24`;
- no subject overlap and no use of the 561 engineered features.

The 21 training subjects are partitioned as follows. Every subject validates
exactly once, and every validation fold contains all six activities.

| Fold | Validation subjects | Windows |
|---:|---|---:|
| 1 | 5, 7, 22, 25 | 1,340 |
| 2 | 1, 19, 21, 28 | 1,497 |
| 3 | 8, 11, 23, 27, 30 | 1,728 |
| 4 | 14, 15, 16, 26 | 1,409 |
| 5 | 3, 6, 17, 29 | 1,378 |

For each fold, normalization and 100-epoch masked pretraining use only that
fold's optimization subjects. Context runs write validation metrics only; the
analysis rejects a tuning artifact if it contains official-test metrics.

## What receives gradients?

The trained path is exactly:

```text
encoder -> projector -> frozen SmolLM2 -> classification head
```

An auxiliary linear sensor head is also trained with weight `0.2`. SmolLM2 is
in evaluation mode and all of its parameters have `requires_grad=False`, but
its forward pass is not wrapped in `no_grad` or `inference_mode`. Autograd
therefore differentiates through SmolLM2 into the continuous sensor token.

| Component | Trainable parameters | Learning rate after selection |
|---|---:|---:|
| PatchTST encoder | 400,768 | `1e-4` |
| Sensor projector | 664,128 | `1e-3` |
| LLM classification head | 3,462 | `1e-3` |
| Auxiliary sensor head | 6,918 | `1e-3` |
| SmolLM2 | **0** | frozen |

Every run records a first-batch gradient audit. For final `Answer:` seed 42,
the gradient norms are sensor token `0.310`, encoder `0.929`, projector `6.865`,
classification head `17.105`, auxiliary head `1.148`, and LLM parameters
`0.000`. This proves that gradients traverse the frozen LLM while its weights
remain unchanged.

## Cross-validation tuning

All candidates use `Answer:`, no weight decay, batch size 128, dropout `0.1`,
and at most 30 epochs with patience 10. The selection metric is unweighted mean
fold macro-F1.

| Encoder LR | Projector/head LR | Fold macro-F1 | Pooled OOF F1 | Median best epoch |
|---:|---:|---:|---:|---:|
| `1e-5` | `3e-4` | 91.22 +/- 4.90 | 91.41 | 4 |
| `1e-5` | `1e-3` | 91.22 +/- 4.48 | 91.45 | 21 |
| `3e-5` | `3e-4` | 91.85 +/- 2.38 | 91.94 | 9 |
| `3e-5` | `1e-3` | 92.33 +/- 2.86 | 92.44 | 7 |
| `3e-5` | `3e-3` | 91.56 +/- 3.27 | 91.67 | 11 |
| **`1e-4`** | **`1e-3`** | **93.30 +/- 2.78** | **93.40** | **18** |
| `1e-4` | `3e-3` | 92.72 +/- 4.22 | 92.89 | 6 |

The locked recipe is encoder LR `1e-4`, projector/head LR `1e-3`, and 18
epochs. The tenfold LR separation lets the pretrained encoder adapt without
moving as aggressively as the randomly initialized interface layers.

## Final official-test results

After selection, masked pretraining was repeated from scratch on all 21
official training subjects for seeds 42--45. Each prompt path was then trained
for exactly 18 epochs. The prompt comparison uses the same pretrained encoder,
optimizer settings, and epoch count within each seed.

| Prompt | Test macro-F1 | Direct sensor-head F1 | Final hidden rank | Static-pair effect size |
|---|---:|---:|---:|---:|
| Coherent instruction | 94.27 +/- 0.98 | **95.74 +/- 0.45** | 23.54 +/- 8.20 | 2.68 +/- 0.09 |
| Scrambled full prompt | **94.42 +/- 1.37** | **95.77 +/- 0.36** | 27.52 +/- 6.66 | 2.78 +/- 0.09 |
| `<sensor> Answer:` | 93.47 +/- 2.57 | **95.73 +/- 0.50** | 20.19 +/- 1.90 | 2.32 +/- 0.21 |

The four `Answer:` test F1 scores are `94.72`, `95.12`, `94.40`, and `89.64`.
It trails the coherent prompt by `0.80` mean points (`p=0.674`) and the
scrambled prompt by `0.95` points (`p=0.627`). These differences are not
statistically conclusive with four seeds. All shuffled-sensor checks are near
the six-class chance level (`16.36%` mean for `Answer:`).

The direct sensor head is the strongest classifier. It exceeds the LLM path by
`1.47` points with the coherent prompt, `1.35` with the scrambled prompt, and
`2.26` with `Answer:`. Thus, the frozen LLM remains a valid differentiable
embedding consumer, but it does not improve classification over the learned
sensor embedding in this corrected protocol.

## Interpretation

The earlier conclusion that prompt context clearly helped was too dependent on
one validation-subject selection and incomplete use of the official training
partition. Under grouped model selection and final all-subject fitting:

- coherent word order still provides no measurable advantage;
- `Answer:` is competitive on three seeds but less stable;
- a longer prompt raises effective rank and static-activity separation;
- higher effective rank still does not guarantee higher F1;
- direct classification of the encoder embedding is better than the LLM path.

The official test labels had already been inspected during earlier exploratory
experiments. The corrected search itself does not use them, but the broader
project history means these results should not be described as a completely
pristine one-shot test estimate.

## Artifacts

- Fold manifest: `outputs/group_cv/folds.json`
- Fold-specific masked checkpoints: `outputs/group_cv/fold_*/`
- Tuning runs: `outputs/group_cv/tuning/`
- Tuning summary: `outputs/group_cv/summary.json`
- Final checkpoints: `outputs/group_cv/final*/`
- Final prompt and geometry summary: `outputs/group_cv/final_summary.json`
- Fold construction: `challenge/make_subject_folds.py`
- Tuning analysis: `challenge/analyze_group_cv.py`
- Final analysis: `challenge/analyze_final_group_cv.py`
