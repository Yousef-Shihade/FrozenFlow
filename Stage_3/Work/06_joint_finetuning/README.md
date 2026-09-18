# Step 06 — Jointly fine-tuning the classifier

Every step so far kept the classifier frozen. This step unfreezes it and trains it jointly
with the flow, comparing the result against both the frozen-classifier setting and the
original Stage 1 linear probe.

**45 runs**: 5 configurations × 3 combinations × 3 seeds, 13 minutes.

---

## Headline

**Unfreezing the classifier makes no measurable difference: −0.02 pp, *p* = 0.97.**

The classifier's extra freedom is available and simply goes unused. Keeping it frozen — which
is what makes every other number in Stage 3 interpretable — costs nothing.

---

## 1 · The confound, and the control that resolves it

If *flow + unfrozen classifier* beats the probe, there are two explanations: the flow is
helping, or **the classifier just got more training**. Stage 1 trained the probe for up to 200
epochs with validation checkpointing so it *should* already be at its ceiling — but "should" is
not a measurement.

So this step runs a **classifier-only control**: continue training the classifier from Stage 1's
weights, same optimiser, budget and checkpoint rule, **no flow at all**. Without it the
extension cannot be interpreted.

| Combination | classifier only | joint | **flow contributes** | frozen S1 | frozen S2 |
| --- | --- | --- | --- | --- | --- |
| ResNet-18 / DTD | +0.02 | +0.23 | **+0.21** | +0.11 | +1.21 |
| DINOv2 / Aircraft | +0.65 | +2.53 | **+1.88** | +2.59 | +2.19 |
| ResNet-18 / Aircraft | −0.01 | +0.97 | **+0.98** | +1.10 | +2.05 |
| **mean** | **+0.22** | **+1.24** | **+1.02** | +1.27 | +1.95 |

Continuing to train the classifier alone moves it **+0.22 pp** — which **independently confirms
Stage 1 left the probe at its ceiling**, rather than that being assumed. Everything the joint
runs gain is therefore attributable to the flow.

---

## 2 · Frozen vs. unfrozen, at matched objective

The `joint` configuration uses Strategy 1's end-to-end loss. Comparing it against "best frozen"
would compare it against Strategy 2, confounding *unfreezing* with *which objective*. Holding
the objective and learning rate fixed:

| Combination | frozen | unfrozen | difference |
| --- | --- | --- | --- |
| ResNet-18 / DTD | +0.11 | +0.23 | +0.12 |
| DINOv2 / Aircraft | +2.59 | +2.53 | −0.06 |
| ResNet-18 / Aircraft | +1.10 | +0.97 | −0.13 |
| **overall** | **+1.27** | **+1.24** | **−0.02 (*p* = 0.97)** |

**No effect.** The −0.71 pp that appears when comparing the joint runs against the *best* frozen
result is not a cost of unfreezing — it is the gap between the two **objectives**, since
Strategy 2 beats Strategy 1 on two of three combinations (step 04). Unfreezing neither closes
that gap nor opens one.

---

## 3 · The freedom is unused — three ways of seeing it

| config | Δ vs. probe | classifier drift | selected epoch |
| --- | --- | --- | --- |
| `clf_only` (control) | +0.22 ± 0.39 | 0.104 | 62 |
| `joint` | +1.24 ± 1.22 | 0.039 | 32 |
| `clf_lr=1e-5` | +1.23 ± 1.20 | **0.005** | 25 |
| `clf_lr=1e-3` | +1.20 ± 1.31 | **0.214** | 46 |
| `delayed=50` | +1.24 ± 1.27 | **0.004** | 25 |

`classifier_drift` is $\lVert W - W_{\text{Stage 1}}\rVert_F / \lVert W_{\text{Stage 1}}\rVert_F$
at the selected epoch.

1. **The classifier learning rate spans 100×** ($10^{-5}$ → $10^{-3}$) and every result lands
   between **+1.20 and +1.24 pp**.
2. **The resulting drift spans 50×** (0.004 → 0.214) with no corresponding change in accuracy.
3. **`delayed=50`, where the classifier barely moves at all** (drift 0.004), scores identically
   to letting it move freely from epoch 0.

A fourth observation is the most telling: the classifier drifts **more when trained alone**
(0.104) than when trained jointly with the flow (0.039). With a flow present, the flow absorbs
the adaptation and the classifier stays put — it is not that the classifier *cannot* move, it is
that once the flow is doing the work there is nothing left for it to do.

---

## 4 · Why this is a useful negative result

Stage 3's entire premise is that the classifier is **pretrained and then frozen**, and every
$\Delta$ reported in steps 02–05 depends on that: the flow at identity *is* the Stage 1 probe,
so any change is the flow's doing. This step is what licenses that framing. Had unfreezing
delivered a large gain, the frozen-classifier results would look like an artificial handicap.
It does not, so they do not.

---

## Design notes

- **`unfreeze()` builds a separate trainable `nn.Linear`** initialised from the frozen weights,
  rather than defeating `FrozenClassifier`'s guarantee. Verified in the notebook: it reproduces
  the classifier's outputs **exactly**, and the original stays frozen — so epoch 0 of every
  configuration here, control included, is the Stage 1 probe.
- **Separate learning rates for flow and classifier**, because they start at incomparable
  points: the classifier is already fitted, the flow starts at identity. One shared rate would
  be an arbitrary choice rather than a neutral one.
- **Regularisation was not swept** here — step 02 already measured it to be the weaker lever in
  this setting, and the three knobs that *are* swept all came back null.

## Outputs

| File | What it holds |
| --- | --- |
| `../../results/joint_finetuning_runs.pt` | all 45 runs, curves, attribution |
| `tables/joint_all_runs.csv` | one row per run |
| `tables/joint_config_summary.csv` | the configuration table above |
| `tables/attribution.csv` | control vs. joint, and the flow's contribution |
| `tables/frozen_vs_unfrozen_matched.csv` | the matched-objective comparison |
| `tables/joint_vs_frozen.csv` | joint against both frozen strategies |
| `tables/classifier_drift.csv` | weight drift per configuration and combination |
| `plots/joint_comparison.png` | all settings, and every configuration incl. the control |
| `plots/joint_curves.png` | validation accuracy — control vs. joint vs. delayed |

**Stage 3's experiments are complete.**
