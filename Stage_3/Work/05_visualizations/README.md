# Step 05 — Feature-space visualisation

The brief's third deliverable: for a readable subset of classes, visualise the original
features $z$ and the transported features $\hat z$ for **both** Stage 3 methods, using the same
test examples and class colours throughout, with the embedding computed **jointly** over the
sets being compared.

Steps 02 and 03 saved every run's metrics and curves but not the flow weights, so this step
retrains the six models it needs (seed 0, at the configurations step 04 selected on validation
accuracy) and **saves them** to `results/visualised_flow_weights.pt`, so nothing downstream has
to retrain again. 81 seconds.

> **Read the numbers in this step as subset diagnostics, not headline results.** Everything here
> is measured on **6 classes, one seed** (199–240 test images), chosen to make a 2-D picture
> readable. The headline accuracies are in step 04, on the full test split over 3 seeds.

---

## 1 · The required figure

`plots/feature_space_*.png` — three panels per combination (original, Strategy 1, Strategy 2),
same test images, same class colours, one PCA fitted on the **concatenation** of all three sets
so the panels are directly comparable rather than three independent rotations.

**PCA rather than t-SNE.** The brief permits either; PCA is used for the reason Stage 1 gave —
it is deterministic (no perplexity or seed to justify), and "compute the embedding jointly over
the sets being compared" has an unambiguous meaning for a linear method. Classes are picked by
an **even stride** through the sorted class list, not hand-picked.

Caveat worth stating: two components capture only **20.7–33.8%** of the variance, so these
pictures are read qualitatively. That is exactly why section 2 measures the geometry in the
full feature space instead.

### The clearest thing the pictures show

On DINOv2/Aircraft the two strategies reach almost the same accuracy by completely different
geometric routes:

| | displacement ‖ẑ−z‖ | Δ accuracy (6 classes) |
| --- | --- | --- |
| Strategy 1 | **34.94** | +1.01 pp |
| Strategy 2 | **4.60** | +1.51 pp |

Strategy 1 visibly **stretches the whole space** — the clusters fly apart, some by more than the
original cloud's own diameter. Strategy 2's panel is almost indistinguishable from the original.
**Equal-or-better result, roughly 8× less movement.** How far the flow moves features is not
what produces the gain.

---

## 2 · What actually changed, measured in the full feature space

A 2-D projection cannot say whether the *class structure* improved. Two quantities can, and
they disagree with each other in an informative way.

| Combination | Strategy | Δ acc (6 cls) | logit margin | Δ separability |
| --- | --- | --- | --- | --- |
| ResNet-18 / DTD | 1 | −1.25 | **0.997×** | +0.0046 |
| ResNet-18 / DTD | 2 | −0.42 | **1.92×** | +0.0445 |
| DINOv2 / Aircraft | 1 | +1.01 | 1.58× | **−0.0739** |
| DINOv2 / Aircraft | 2 | +1.51 | 1.64× | +0.0059 |
| ResNet-18 / Aircraft | 1 | +2.01 | 1.45× | +0.0429 |
| ResNet-18 / Aircraft | 2 | +3.52 | 1.62× | +0.0103 |

**Logit margin is the quantity that tracks the result.** The frozen classifier is linear, so its
decision is entirely the gap between the top logit and the runner-up. Five of six cases widen
that gap by **1.45–1.92×**. The single exception is **Strategy 1 on DTD at 0.997×** — no widening at
all, in fact a hair narrower — and that is precisely the combination where Strategy 1 delivered nothing on the
full test set (+0.11 pp, one seed negative). The diagnostic and the headline agree.

**Class separability does not track the result.** Own-minus-other cosine similarity — the
training-free measure from Stage 1 step 02 that predicted the whole Stage 1 ordering — moves
the *wrong way* where the gain is real: Strategy 1 on DINOv2/Aircraft **loses 0.074** of
separability while *gaining* accuracy.

This is the same lesson Stage 2 recorded from the other direction. There, flows contracted
features toward the right prototype without separating them from the wrong ones, and accuracy
did not follow. Here a flow degrades class-mean separability and accuracy improves anyway.
**Neither clustering in a projection nor class-mean geometry is the thing a linear classifier
actually consumes** — the margin is.

---

## 3 · Why Strategy 2 needs so little movement

Strategy 2 is trained toward targets built by descending the classification loss in feature
space, so its target vector already points along the direction that widens the margin. It moves
features a short distance in a useful direction. Strategy 1 has no such guidance — it
discovers a direction by backpropagating through the rollout, and the displacement it settles
on is far larger for the same benefit.

That also fits step 04's robustness finding: the strategy that moves less is the one whose
validation loss rises more slowly and which tolerates a 10× larger learning rate.

---

## Outputs

| File | What it holds |
| --- | --- |
| `../../results/visualised_flow_weights.pt` | the six trained flows, so they need not be retrained |
| `tables/visualised_models.csv` | the six models' accuracies (seed 0, full test split) |
| `tables/feature_geometry.csv` | accuracy, margin, separability, displacement per panel |
| `tables/geometry_change.csv` | each strategy's change against the original features |
| `plots/feature_space_resnet18_dtd.png` | the required three-panel figure |
| `plots/feature_space_dinov2_fgvcaircraft.png` | the required three-panel figure |
| `plots/feature_space_resnet18_fgvcaircraft.png` | the required three-panel figure |
| `plots/what_the_flow_changes.png` | accuracy, margin and separability change side by side |

**Next:** step 06, the optional extension — unfreezing the classifier and training it jointly
with the flow.
