# Step 02 — Strategy 1: end-to-end rolled-out classification

For each training feature $z$, run the complete $T$-step Euler rollout to get $\hat z$, push it
through the **frozen** classifier, and minimise

$$\mathcal{L}_\mathrm{cls} = \mathrm{CE}(W\hat z + b,\; y)$$

backpropagating through the whole rollout and updating **only** the flow's parameters. The flow
starts at identity, so before any gradient step the system *is* the Stage 1 linear probe.

**81 runs**: 9 configurations × 3 combinations × 3 seeds, 32 minutes of training.

---

## Headline

| Encoder / dataset | Frozen probe | Strategy 1 (lr $10^{-5}$) | Δ |
| --- | --- | --- | --- |
| ResNet-18 / DTD | 52.04% | 52.36% | **+0.32 ± 0.24 pp** |
| ResNet-18 / FGVC-Aircraft | 27.87% | 28.96% | **+1.09 ± 0.47 pp** |
| DINOv2 / FGVC-Aircraft | 51.54% | 53.75% | **+2.21 ± 0.60 pp** |

A small, consistent gain. Excluding the overfitting default configuration, **63 of 72 runs
improve on the frozen probe**; ResNet-18/Aircraft improves in **24 of 24**.

---

## 1 · The objective is almost exhausted before training starts

The loss being minimised is the frozen probe's classification loss **on the probe's own
training set** — the same 470 or 1000 k-shot features it was fitted on in Stage 1. Measured at
identity:

| Encoder / dataset | train acc | val acc | CE(train) | CE(val) | ratio |
| --- | --- | --- | --- | --- | --- |
| ResNet-18 / DTD | 98.44% | 51.52% | 0.327 | 1.823 | 7× |
| DINOv2 / FGVC-Aircraft | **100.00%** | 52.11% | **0.005** | 2.304 | **449×** |
| ResNet-18 / FGVC-Aircraft | **100.00%** | 27.67% | 0.031 | 3.573 | 122× |

It is worth being precise about what this does and does not imply. It does **not** mean there
is no gradient to follow — the training curves show the flow driving training cross-entropy all
the way down to $10^{-6}$. It means **following that signal is memorisation**.

So the question Strategy 1 really poses is not *can the flow reduce the loss* (it can,
completely) but **how little can it be allowed to move before memorisation outweighs whatever
real structure it finds.**

---

## 2 · The default learning rate was the worst choice

| config | Δ vs. probe | selected epoch | moved |
| --- | --- | --- | --- |
| **main** (lr $10^{-3}$) | **+0.15 ± 0.85 pp** | 9 | 12.7% |
| lr $10^{-4}$ | **+1.27 ± 1.32 pp** | 28 | 28.6% |
| lr $10^{-5}$ | **+1.21 ± 0.91 pp** | 131 | 16.4% |
| displacement 0.001 / 0.01 / 0.1 | +0.70 / +0.64 / +0.29 pp | 107 / 140 / 105 | 4.6% / 2.5% / 0.9% |
| velocity 0.001 / 0.01 / 0.1 | +0.78 / +0.56 / +0.32 pp | 101 / 118 / 68 | 4.7% / 2.5% / 1.0% |

$10^{-3}$ is the value Stages 1 and 2 both use, and carrying it over was the natural choice —
it was right in Stage 2, where a flow trains from scratch against fixed targets. Here it is
actively harmful: it drives a 657k-788k-parameter network onto 470–1000 features so fast that
validation accuracy never rises, and the checkpoint rule salvages an epoch-9 model that has
barely moved. **Reducing the learning rate by 10× is worth about +1.1 pp**, which is larger
than anything the regularisation achieves.

Both regularisers behave exactly as designed — stronger penalty, less movement (4.6% → 0.9%) —
but constraining *how far* the flow moves is less effective than constraining *how fast* it
gets there.

---

## 3 · Remaining training loss anti-predicts the gain

This is the step's most interesting result, and it runs against the obvious reading of
section 1.

| combination | CE(train) at identity | gain at lr $10^{-5}$ |
| --- | --- | --- |
| ResNet-18 / DTD | 0.327 — **most** left to minimise | **+0.32 pp** — smallest gain |
| ResNet-18 / Aircraft | 0.031 | +1.09 pp |
| DINOv2 / Aircraft | 0.005 — **least** left to minimise | **+2.21 pp** — largest gain |

The combination with the most unminimised loss gains the *least*; the one whose objective is
already at zero gains the *most*. Whatever the flow is exploiting, **it is not the residue of
the classification objective.**

Displacement tells the same story from the other side: across all 81 runs, how far the flow
moved features correlates *positively* with the gain (Spearman $\rho = +0.54$, $p < 0.001$;
$+0.70$ on DINOv2/Aircraft). The flow has to actually move features to help — it just has to
get there slowly.

**What we do not claim.** With three combinations it is impossible to say what *does* explain
the ordering: DTD differs from the two Aircraft combinations in dataset, in class count
(47 vs 100) and in training-set size (470 vs 1000) simultaneously. Stage 2's headroom rule
earned its claim because nine cells lined up behind it; three do not support a rule here.

---

## 4 · Training and validation curves

`plots/training_curves.png` shows the mechanism directly, and is the clearest single figure in
this step:

- **Training accuracy is pinned at ~100% for all 200 epochs** — it starts there.
- **Training cross-entropy falls from ~0.005 to $10^{-6}$** — five orders of magnitude of
  optimisation, all of it on data already classified correctly.
- **Validation cross-entropy rises monotonically from epoch 0.** There is no regime in which
  both improve.
- Validation accuracy peaks within the first ~20 epochs and decays thereafter.

Checkpoint selection on validation accuracy is therefore not a refinement here, it is what
makes the method work at all. It is also Stage 1's own rule, which keeps the comparison against
the probe consistent.

---

## Design decisions

- **Validation-based checkpointing**, unlike Stage 2's fixed epoch budget. Stage 2 avoided it
  because its two objectives were on different scales and no single validation quantity could
  select fairly between them. Stage 3's strategies both produce directly comparable validation
  accuracies, so the fair rule is available — and it is Stage 1's rule, the baseline being
  compared against.
- **Gradient clipping at 1.0**, carried over from Stage 2, which found backpropagation through
  $T$ composed Euler steps can amplify gradients enough to diverge.
- **Velocities recovered from the trajectory** ($v_k = T(\hat z_{k+1} - \hat z_k)$) rather than
  by re-running the network, so the penalty is measured on exactly the states the loss saw.

## Outputs

| File | What it holds |
| --- | --- |
| `../../results/strategy1_runs.pt` | all 81 runs, every training curve, the diagnostic |
| `tables/strategy1_all_runs.csv` | one row per run |
| `tables/strategy1_config_summary.csv` | the configuration table above |
| `tables/strategy1_main_summary.csv` | per-combination, main configuration |
| `tables/strategy1_best_config_summary.csv` | per-combination, best configuration |
| `tables/strategy1_learning_rate_sweep.csv` | the learning-rate comparison |
| `tables/objective_diagnostic.csv` | per-run probe train/val accuracy and CE at identity |
| `plots/training_curves.png` | train vs. validation, accuracy and loss (required deliverable) |
| `plots/strategy1_results.png` | the diagnostic, and every configuration's Δ |
| `plots/strategy1_displacement.png` | displacement vs. gain, and selected-epoch distribution |

**Next:** step 03 trains Strategy 2 — classifier-guided targets with standard FM.
