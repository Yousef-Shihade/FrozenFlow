# Step 04 — Evaluation

The consolidated comparison, assembled from steps 01–03's saved results. **Nothing is
retrained here** — every number comes from the run tables those steps wrote.

For each dataset, three methods on the same encoder, the same k-shot subset, the same seeds and
the same pretrained classifier: the **Stage 1 linear probe**, **Strategy 1** (end-to-end
rolled-out classification), and **Strategy 2** (classifier-guided targets with standard FM).

---

## Classification results

Top-1 test accuracy, $K = 10$, $T = 4$, mean ± sample std over 3 seeds.

| Combination | Linear probe | Strategy 1 | ΔAcc | Strategy 2 | ΔAcc |
| --- | --- | --- | --- | --- | --- |
| ResNet-18 / DTD | 52.04% | 52.15 ± 1.15 | +0.11 | **53.24 ± 1.39** | **+1.21** |
| DINOv2 / FGVC-Aircraft | 51.54% | **54.13 ± 1.13** | **+2.59** | 53.73 ± 1.53 | +2.19 |
| ResNet-18 / FGVC-Aircraft \* | 27.87% | 28.97 ± 0.74 | +1.10 | **29.92 ± 0.60** | **+2.05** |

\* carried beyond the two main combinations.

**Every combination improves on average, under both objectives** — Strategy 1 by +1.27 pp
overall, Strategy 2 by +1.82 pp.

### How the reported configuration was chosen

Steps 02 and 03 swept 9 and 11 configurations. Picking the winner by **test** accuracy would
make the headline optimistic by construction, so the configuration is chosen by **mean
validation accuracy**, which appears in no reported figure.

| Strategy | Best by validation | Best by test | Agree? |
| --- | --- | --- | --- |
| Strategy 1 | `lr=1e-4` | `lr=1e-4` | yes |
| Strategy 2 | `refresh=10` | `refresh=10` | yes |

Validation and test independently pick the same configuration for both strategies, so the
numbers above are not a product of test-set selection.

---

## Per-seed detail

| Strategy | Combination | per-seed Δ (pp) | mean | all > 0 | *p* |
| --- | --- | --- | --- | --- | --- |
| 1 | ResNet-18 / DTD | −0.11, 0.00, +0.43 | +0.11 | **no** | 0.580 |
| 1 | DINOv2 / Aircraft | +1.38, +3.03, +3.36 | +2.59 | yes | 0.052 |
| 1 | ResNet-18 / Aircraft | +0.39, +0.63, +2.28 | +1.10 | yes | 0.205 |
| 2 | ResNet-18 / DTD | +0.74, +1.33, +1.54 | +1.21 | yes | 0.037 |
| 2 | DINOv2 / Aircraft | +1.32, +2.16, +3.09 | +2.19 | yes | 0.050 |
| 2 | ResNet-18 / Aircraft | +1.29, +1.89, +2.97 | +2.05 | yes | 0.053 |

Three seeds give a t-test very little power, so these *p*-values are reported for completeness
rather than leaned on. The more informative statement is how consistently the sign holds across
the **full** sweeps: **67/81 (83%)** of Strategy 1 runs and **91/99 (92%)** of Strategy 2 runs
improve on the probe, across every configuration tried.

The one genuinely weak cell is **Strategy 1 on DTD** (+0.11, one seed negative) — visible in
the training curves, where its validation accuracy falls from 52% to 42% over the run.

---

## Do the two strategies actually differ?

Holding the learning rate fixed isolates the objective from the tuning:

| lr | Strategy 1 | Strategy 2 | difference | *p* |
| --- | --- | --- | --- | --- |
| $10^{-3}$ | +0.15 | **+0.94** | **+0.79** | **0.048** |
| $10^{-4}$ | +1.27 | +1.40 | +0.13 | 0.802 |
| $10^{-5}$ | **+1.21** | +0.89 | −0.31 | 0.451 |

**On accuracy they are indistinguishable** — +0.13 pp at *p* = 0.80 — and they trade wins per
combination.

**On robustness they clearly differ, and significantly** (*p* = 0.048 at $10^{-3}$). The
training curves show the mechanism directly: Strategy 1's validation cross-entropy rises far
faster than Strategy 2's on all three combinations, so it has to be stopped much earlier
(selected epochs 4 / 66 / 35 against 101 / 117 / 198).

Why: Strategy 1 optimises the classification loss **directly**, so a large step memorises
training labels within a handful of epochs. Strategy 2 optimises an **MSE regression onto a
target vector** — labels reach it only through where that target was placed, so the same step
size cannot memorise them nearly as fast.

---

## What Stage 3 answers

**Does a flow-matching layer in front of a frozen linear classifier help?** Yes, but modestly:
**+0.1 to +2.6 pp**. For contrast, Stage 2 found up to **+22.3 pp** applying the same idea in
front of an *untrained* prototype classifier.

That contrast is the point — and Stage 2 predicted it. Stage 2's headroom result (FM gain
tracks *trained probe − untrained prototype*, ρ = 1.000) says the flow recovers roughly what a
trained classifier would have gained. In Stage 3 the classifier **is** trained, so that headroom
is already spent. What remains is whatever a nonlinear transform can add on top of an
already-fitted linear boundary: about a point.

**Does classifier-guided training beat end-to-end rolled-out training?** Not on accuracy. It is
substantially more robust to the learning rate.

**What limits both?** Step 02 measured it: the frozen probe already classifies its own k-shot
training set at 98.44–100.00%, so the signal both strategies chase is nearly exhausted and following
it further is memorisation. Both depend entirely on validation-accuracy checkpointing.

---

## Outputs

| File | What it holds |
| --- | --- |
| `../../results/stage3_evaluation.pt` | the consolidated comparison |
| `tables/main_comparison.csv` | the headline accuracy table |
| `tables/delta_vs_probe.csv` | ΔAcc per combination and strategy |
| `tables/per_seed_deltas.csv` | per-seed values and t-tests |
| `tables/configuration_ranking.csv` | every configuration by validation and test |
| `tables/matched_learning_rate.csv` | the strategy comparison at fixed lr |
| `plots/main_comparison.png` | three methods, and ΔAcc (core result) |
| `plots/training_curves_both_methods.png` | train/validation curves, both methods (core result) |
| `plots/matched_learning_rate.png` | objective vs. tuning |

**Next:** step 05 visualises what the flow does to the feature space.
