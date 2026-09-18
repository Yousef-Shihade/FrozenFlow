# Step 03 — Linear probe

**A Computer Vision Project — *Flow Matching as a Layer*** · Stage 1
University of Haifa

---

## Purpose

Our first baseline, and the one **Stage 3** will replace with a Flow-Matching layer. It
answers: given a frozen representation, how far does a single trained linear layer get?

`s = W z + b`, with `z` a cached frozen feature from step 2. Only `W` and `b` are trained,
with softmax cross-entropy. The encoders are never loaded — this notebook reads tensors.

---

## Quick start

```text
1. Open  code/03_linear_probe.ipynb  in VS Code
2. Select the kernel:  Python (CVLAB Stage 1)
3. Run All
```

**Runtime ~3 minutes** for all 27 runs. Requires step 2's feature cache.

---

## The experiment grid

3 encoder-dataset pairs × K ∈ {5, 10, full} × 3 seeds = **27 runs**.

The two kinds of seed are easy to conflate:

| Setting | What the seed controls |
| --- | --- |
| 5-shot, 10-shot | **which images** form the balanced training subset |
| full | the classifier's **weight initialisation** and batch order |

Both come from the same `seed` argument; what differs is whether a subsampled or complete
training set is passed in.

---

## Results

Top-1 accuracy on the complete official test split, mean ± sample standard deviation
(`ddof=1`) over 3 runs.

| Dataset / encoder | 5-shot | 10-shot | full |
| --- | --- | --- | --- |
| ResNet-18 / DTD | 44.86 ± 1.83 | 52.04 ± 1.16 | 62.82 ± 0.42 |
| ResNet-18 / FGVC-Aircraft | 20.76 ± 0.26 | 27.87 ± 1.33 | 38.15 ± 0.29 |
| DINOv2 / FGVC-Aircraft | 38.19 ± 2.49 | 51.54 ± 1.70 | 67.55 ± 0.28 |

Every combination improves monotonically with K, and error bars shrink as the training set
grows — both as expected.

**DINOv2 beats ResNet-18 on Aircraft by 17–29 pp, and the gap widens with data.** This is
exactly the prediction step 2 made from the frozen features alone: separability margins of
0.041 (ResNet-18) vs 0.420 (DINOv2). The classifier results and the feature geometry agree,
which is a genuine cross-check rather than a restatement.

---

## Independent reproduction

We ran this sweep twice, through two independently written implementations. The final
pipeline differs from our first one in every moving part — re-extracted features, a
rewritten training loop, different batching — so comparing them is an independent
reproduction.

| Dataset | Banner crop applied | mean Δ | range of Δ |
| --- | --- | --- | --- |
| DTD | no | −0.06 pp | −0.26 … +0.06 pp |
| FGVC-Aircraft | yes | +0.33 pp | −0.36 … +1.15 pp |

**DTD reproduces to within 0.26 pp on all three K settings.** Because DTD receives no banner
crop, it isolates the reimplementation itself — and it agrees about as closely as two
independent implementations can. That is strong evidence both are correct.

**Aircraft scatters more but mixed in sign** (two of six settings negative). If the banner
crop were a real improvement the shift would be consistently positive; it is not. This
independently corroborates step 2's ablation: the crop's effect is indistinguishable from
run-to-run variation.

---

## Two findings worth being able to explain

### 1. Why the Aircraft checkpoint sits at epoch ~199 while validation loss is rising

The loss-curve figure below looks self-contradictory at first glance, and it is the most
likely thing to be challenged on.

**Validation loss and validation accuracy stop agreeing.** Cross-entropy punishes
*confidence*, not just correctness. Once the probe predicts its already-correct examples
with ever-higher confidence it also becomes more confidently wrong on its mistakes, pushing
the loss up — while the *ranking* of the logits, which is all `argmax` responds to, keeps
improving.

The figure therefore has two rows: the training/validation loss curves, and the validation
accuracy the checkpoint rule actually maximises. On Aircraft, accuracy is still climbing at
epoch 200 while loss has been rising since ~epoch 35. The checkpoint rule selects on
**highest validation accuracy**, so the loss minimum (grey dotted line) is deliberately not
what we pick. Selecting on loss would choose epoch ~35 and cost real accuracy.

DTD behaves conventionally — both criteria agree early — which makes the contrast
instructive rather than worrying.

### 2. The two datasets overfit on completely different timescales

| Combination | mean selected epoch (of 200) |
| --- | --- |
| ResNet-18 / DTD | **26** |
| ResNet-18 / FGVC-Aircraft | **152** |
| DINOv2 / FGVC-Aircraft | **160** |

On DTD the probe peaks around epoch 26 and spends ~87% of the budget purely overfitting. On
Aircraft it is still improving past epoch 150. That follows from the difficulty gap step 2
measured: DTD's classes are nearly linearly separable in ResNet-18 features, so the probe
finds its boundary fast and then memorises; Aircraft's 100 near-identical variants take far
longer to separate.

9 of 27 runs select a checkpoint in the last 15% of the budget (DINOv2 10-shot averages
epoch 187 of 200), which initially looked like 200 epochs might be cutting Aircraft training
short. **This was tested rather than left as a guess:** section 9b of the notebook reruns
K=10 and K=full at 3× the epoch budget (600), seed 0, all three combinations.

| Combination | 200 ep | 600 ep | gain |
| --- | --- | --- | --- |
| DINOv2 / Aircraft, 10-shot | 53.20 | 53.41 | +0.21 pp |
| DINOv2 / Aircraft, full | 67.36 | 67.36 | 0.00 pp |
| ResNet-18 / DTD, 10-shot | 51.17 | 51.17 | 0.00 pp |
| ResNet-18 / DTD, full | 63.30 | 63.30 | 0.00 pp |
| ResNet-18 / Aircraft, 10-shot | 26.70 | 26.85 | +0.15 pp |
| ResNet-18 / Aircraft, full | 37.86 | 37.65 | −0.21 pp |

**Tripling the epoch budget moves test accuracy by at most 0.21 pp, in either direction —
noise.** Late checkpoint selection is real, but past that point the runs were fitting
validation-set particulars that did not transfer to test, not leaving genuine accuracy on
the table. **200 epochs was already enough**, and the reported numbers are not pessimistic.
This is a case where the initial concern did not survive being tested — worth recording as
such rather than quietly dropping it.

---

## Outputs

### Shared artefacts (`Stage_1/results/`) — consumed by step 5

| File | Contents |
| --- | --- |
| `linear_probe_runs.csv` | One row per run: accuracies, selected epoch, losses, timing |
| `linear_probe_curves.json` | Per-epoch train loss, val loss, val accuracy — **all 27 runs** |
| `linear_probe_predictions.pt` | Test predictions per run |

Keeping predictions for every run means step 5 can build confusion matrices **without
retraining anything**. Retraining a probe just to obtain them would risk the confusion
matrix describing a slightly different model from the one in the accuracy table.

### Tables (`tables/`)

`methodology.csv` · `probe_config.csv` · `run_grid.csv` ·
`accuracy_summary.csv` · `reproduction_check.csv` · `overfitting_summary.csv` ·
`epoch_budget_check.csv`

### Plots (`plots/`)

| File | Shows |
| --- | --- |
| `loss_curves_representative.png` | 10-shot curves + val accuracy |
| `training_stability.png` | All 27 runs, 3 seeds per panel |
| `overfitting_analysis.png` | Selected epoch, val-loss climb, final train loss |
| `reproduction_check.png` | The two implementations, per setting |
| `accuracy_vs_k.png` | Accuracy vs training-set size with error bars |

---

## Configuration

A standard configuration, **kept unmodified**: AdamW, lr 1e-3, weight decay 1e-4, batch
size 64, max 200 epochs, checkpoint on highest validation accuracy.

We would have adjusted it had it behaved poorly. It did not need to: no run diverged, none
produced NaN or Inf, and every run selected a real checkpoint. Section 9 of the notebook is
the evidence. Stating this explicitly is better than leaving a reader to wonder whether the
defaults were used or quietly tuned.

---

## Next step

**Step 04 — Image-derived class prototypes**, the second baseline. No training:
L2-normalise, average each class's training features, re-normalise, classify by cosine
similarity. 21 runs (5-shot ×3, 10-shot ×3, full ×1 per combination — the full setting needs
one run because averaging every training image leaves nothing stochastic).

Because it calls the same `cvlab.data.make_kshot_subset`, it trains on **byte-identical
subsets** to the ones used here for a given `(K, seed)`. Any accuracy difference between the
two baselines therefore comes from the classifier design — which is the whole point of
running both.
