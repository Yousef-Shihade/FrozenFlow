# Step 03 — Rolled-out flow matching

**A Computer Vision Project — *Flow Matching as a Layer*** · Stage 2
University of Haifa

---

## Purpose

Train the *same* velocity network with the **rolled-out** objective and compare it against
both the Stage 1 prototype baseline and step 2's standard FM.

$$\hat z_{k+1} = \hat z_k + \tfrac{1}{T}\,v_\theta\!\left(\hat z_k, \tfrac{k}{T}\right),
\qquad \mathcal{L}_\mathrm{roll} = \lVert \hat z_T - p_{y_i} \rVert_2^2$$

Gradients flow back through **all $T$ velocity predictions**, so the network trains on exactly
the states inference produces — removing step 2's train/inference mismatch by construction.

---

## Quick start

```text
1. Open  code/03_rolled_out_fm.ipynb  in VS Code
2. Select the kernel:  Python (CVLAB Stage 1)
3. Run All
```

**Runtime: ~40 minutes** (54 models; the $T=12$ runs dominate). Requires step 1's
`prototypes.pt` and step 2's `standard_fm_*` artefacts.

---

## What changes, and what does not

The network, its size, the optimiser, gradient clipping, the batch size, the epoch budget,
the k-shot subsets and the seeds are **identical to step 2**. Only the loss differs — which
is what makes this a statement about the objective.

| | Standard FM (step 2) | Rolled-out FM (this step) |
| --- | --- | --- |
| States seen in training | on the ideal straight path | the network's **own** rollout |
| Supervision | every intermediate velocity | the final position only |
| Train/inference mismatch | yes | **none, by construction** |
| Depends on $T$ | no — one model serves all $T$ | **yes** — one model per $T$ |
| Models trained | 27 | **54** |
| Cost per step | 1 forward/backward | $T$ forward/backward |

---

## Headline result: removing the mismatch does **not** help

Rolled-out FM beats standard FM in only **4 of 18 cells**, and beats the Stage 1 baseline in
**10 of 18**.

| Encoder / dataset | $K$ | $T$ | Baseline | Standard FM | Rolled-out FM | vs. base | vs. std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-18 / DTD | 5 | 4 | 46.10 | 44.47 | 40.99 ± 1.60 | −5.11 | **−3.48** |
| ResNet-18 / DTD | 5 | 12 | 46.10 | 44.79 | 40.87 ± 1.46 | −5.23 | **−3.92** |
| ResNet-18 / DTD | 10 | 4 | 51.67 | 50.11 | 44.77 ± 1.58 | −6.90 | **−5.34** |
| ResNet-18 / DTD | 10 | 12 | 51.67 | 50.34 | 45.32 ± 1.23 | −6.35 | **−5.02** |
| ResNet-18 / DTD | full | 4 | 58.83 | 59.59 | 55.85 ± 0.47 | −2.98 | −3.74 |
| ResNet-18 / DTD | full | 12 | 58.83 | 59.56 | 55.82 ± 0.32 | −3.01 | −3.74 |
| ResNet-18 / Aircraft | 5 | 4 | 16.46 | 18.81 | 16.77 ± 0.84 | +0.31 | −2.04 |
| ResNet-18 / Aircraft | 5 | 12 | 16.46 | 18.94 | 17.03 ± 0.87 | +0.57 | −1.91 |
| ResNet-18 / Aircraft | 10 | 4 | 20.19 | 23.68 | 20.34 ± 0.53 | +0.15 | −3.34 |
| ResNet-18 / Aircraft | 10 | 12 | 20.19 | 23.91 | 20.09 ± 0.24 | −0.10 | −3.82 |
| ResNet-18 / Aircraft | full | 4 | 25.47 | 31.04 | 25.64 ± 1.72 | +0.17 | **−5.40** |
| ResNet-18 / Aircraft | full | 12 | 25.47 | 30.97 | 25.28 ± 1.69 | −0.19 | **−5.69** |
| DINOv2 / Aircraft | 5 | 4 | 23.79 | 33.30 | **34.13 ± 0.59** | +10.34 | **+0.83** |
| DINOv2 / Aircraft | 5 | 12 | 23.79 | 32.82 | **33.86 ± 0.58** | +10.07 | **+1.04** |
| DINOv2 / Aircraft | 10 | 4 | 27.74 | 43.27 | **45.42 ± 1.10** | +17.68 | **+2.15** |
| DINOv2 / Aircraft | 10 | 12 | 27.74 | 42.48 | **45.35 ± 0.83** | +17.61 | **+2.87** |
| DINOv2 / Aircraft | full | 4 | 34.26 | 56.52 | 52.05 ± 0.91 | +17.78 | −4.47 |
| DINOv2 / Aircraft | full | 12 | 34.26 | 55.70 | 52.08 ± 0.82 | +17.81 | −3.62 |

**Where it does win** is narrow and consistent: DINOv2/Aircraft at $K = 5$ and $K = 10$ —
the strongest encoder in the lowest-data settings, where +0.83 to +2.87 pp is small but holds
across both $T$ and all three seeds. At full data on the same combination it loses by ~4 pp.

---

## Why it loses

### It is not underfitting, and it is not overfitting the transport

`plots/transport_diagnostic.png` measures the mean distance $\lVert \hat z_T - p_y \rVert$
separately on train and test, for both objectives.

| Encoder / dataset | method ($T=4$) | train gap | test gap | generalisation gap |
| --- | --- | --- | --- | --- |
| ResNet-18 / DTD | no transport | 0.6854 | 0.7068 | 0.0214 |
| | standard FM | 0.3343 | 0.4234 | 0.0891 |
| | rolled-out FM | **0.2759** | **0.3674** | 0.0915 |
| DINOv2 / Aircraft | no transport | 0.6997 | 0.7225 | 0.0228 |
| | standard FM | 0.2901 | 0.3336 | 0.0435 |
| | rolled-out FM | **0.2360** | **0.2809** | 0.0449 |

Rolled-out reaches the prototype **more** closely than standard FM — on the *test* split as
well as on train — and its train/test gap is no larger. The transport does exactly what it
was trained to do, and it generalises.

### The objective is misaligned with the decision rule

$$\mathcal{L}_\mathrm{roll} = \lVert \hat z_T - p_{y} \rVert_2^2
\qquad\text{but}\qquad
\hat y = \arg\max_c \cos(\hat z_T,\, \mu_c)$$

The loss rewards **absolute proximity** to one prototype. The classifier reads a **relative**
quantity — whether the correct prototype beats every other one.

| Encoder / dataset ($T=4$) | method | $\cos(\hat z_T, \mu_y)$ | $\max_{c\neq y}\cos$ | margin | accuracy |
| --- | --- | --- | --- | --- | --- |
| ResNet-18 / DTD | standard | 0.9071 | 0.8961 | +0.0110 | 58.88 |
| | rolled-out | **0.9265** | **0.9219** | +0.0046 | 55.32 |
| ResNet-18 / Aircraft | standard | 0.9766 | 0.9792 | −0.0027 | 31.53 |
| | rolled-out | **0.9851** | **0.9880** | −0.0029 | 26.13 |
| DINOv2 / Aircraft | standard | 0.9390 | 0.9331 | +0.0059 | 56.56 |
| | rolled-out | **0.9544** | **0.9528** | +0.0016 | 52.45 |

**Consistent across all three:** rolled-out raises similarity to the correct prototype *and*
to the competing ones together. It compresses features toward the whole prototype cloud
rather than sharpening the decision — minimising distance to one prototype simply does not
require the result to be more *distinguishable* from the others.

**A caveat on the margin column.** Accuracy *is* the fraction of examples with margin $> 0$,
so "accuracy fell because the margin fell" is closer to a restatement than an explanation.
The mean margin also fails as a summary on ResNet-18/Aircraft, where it is essentially
unchanged ($-0.0027 \to -0.0029$) while accuracy drops 5.4 pp — at ~26 % accuracy most
examples sit on the wrong side of zero and the mean is dominated by how badly they fail. The
conclusion therefore rests on the two similarity columns, which move consistently everywhere.

---

## Training stability and gradient clipping

Backpropagating through $T$ composed Euler steps can amplify gradients. Gradient-norm
clipping at 1.0 is applied — and applied to **both** objectives, not only the one that needed
it, so the two remain configuration-identical. It is inert for standard FM (every cell moved
by ≤ 0.25 pp, headline +22.26 → +22.25), which is the evidence that it hands step 2 no
advantage.

### The instability it addresses — kept as evidence, not removed

`plots/clipping_evidence.png` shows the same run with and without clipping
(`resnet18`/Aircraft, $K = \text{full}$, $T = 4$, seed 2):

| | min loss | max loss | final loss |
| --- | --- | --- | --- |
| without clipping | 0.0221 (ep 96) | **24.47** (ep 99) | 1.0987 |
| with clipping | 0.0221 (ep 100) | 0.24 (ep 78) | **0.0221** |

**Clipping does not prevent the instability — it bounds it.** A spike still occurs around
epoch 78; the difference is that training recovers within one epoch instead of ending 50×
above its own minimum. Without clipping the fixed-budget rule captured that run mid-spike and
the cell read 21.29 ± 5.74; with clipping it reads 25.64 ± 1.72. Worst seed spread across all
18 cells fell from **5.74 to 1.72**.

The unclipped run is preserved at `results/rolled_out_fm_curves_noclip.json` and
`results/rolled_out_fm_runs_noclip.csv`, because the difference in numerical robustness is
itself a real property of the two objectives: **standard FM never diverged in 27 unclipped
runs; rolled-out did in 1 of 54.**

### With clipping, training is stable everywhere

Loss falls 79.7–92.5 % in every configuration; final loss ≤ 0.088; seed spread at the final
epoch ≤ 0.0013. The rolled-out objective reaches a much lower final loss than standard FM
(0.022–0.088 vs 0.080–0.207) — but the two losses measure different quantities (endpoint
distance vs. velocity error at a sampled time) and **must not be compared to each other**.

One entry needs reading with care: `resnet18`/Aircraft, $K=\text{full}$, $T=4$ shows a
last-10-epoch improvement of 44.1 %, which looks like it had not converged. It is the tail of
the bounded spike shown above — the loss recovering back to its plateau, not continued
learning.

---

## Outputs

### Shared artefacts (`Stage_2/results/`) — consumed by steps 4–5

| File | Contents |
| --- | --- |
| `rolled_out_fm_runs.csv` | One row per (combination, $K$, seed, $T$) — 54 rows |
| `rolled_out_fm_curves.json` | Per-epoch training loss for all 54 models |
| `rolled_out_fm_predictions.pt` | Test predictions, so step 4 needs no re-running |
| `rolled_out_fm_models.pt` | Weights for the 3 representative runs per $T$, for step 5 — **not tracked in git** (see below) |
| `rolled_out_fm_curves_noclip.json` | The pre-clipping run, kept as stability evidence |
| `rolled_out_fm_runs_noclip.csv` | Its accuracies, including the 21.29 ± 5.74 cell |

`rolled_out_fm_models.pt` is **not committed to the repository**: it is regenerated by
re-running this notebook, and everything needed to check the results — the per-run
accuracies, the training curves, and the per-example test predictions — is tracked instead.

### Tables (`tables/`)

`accuracy_summary.csv` · `training_stability.csv` · `transport_diagnostic.csv` ·
`margin_analysis.csv` · `methodology.csv`

### Plots (`plots/`)

| File | Shows |
| --- | --- |
| `train_inference_mismatch.png` | Step 2's trained model rolled out against the straight path its objective assumed — the motivation for this step |
| `accuracy_vs_k.png` | All three methods, per combination, per $T$, with error bars |
| `delta_comparisons.png` | $\Delta$Acc against the baseline and against standard FM |
| `training_curves.png` | Loss per epoch, mean over 3 seeds, min/max band |
| `clipping_evidence.png` | The unclipped divergence beside the clipped run |
| `transport_diagnostic.png` | Distance to the prototype, train vs. test, both objectives |
| `margin_analysis.png` | Similarity to the correct prototype vs. to its best competitor |

---

## What this means for the project

That question was: does compare standard FM training with rolled-out training favor the
latter? The answer is not the intuitive one: **removing the train/inference mismatch, on its own, does not produce a
better classifier.** It produces a better *transport* — measurably closer to the target
prototype on held-out data — which is a different thing from a better *decision*.

The one regime where it does help is the strongest encoder at the smallest training sizes
(DINOv2/Aircraft, $K \in \{5, 10\}$, +0.8 to +2.9 pp), consistent with rolled-out training
being a stronger constraint that pays off when data is scarce and the features are already
well separated.

---

## Next step

**Step 4 — Evaluation.** Assembles the headline results from steps 2 and 3's saved
results: the full accuracy table across all five methods (baseline, standard FM at
$T \in \{4, 12\}$, rolled-out FM at $T \in \{4, 12\}$), $\Delta\mathrm{Acc}$ against the
baseline, the accuracy-versus-$K$ plot with error bars, and representative training curves
for both objectives side by side.
