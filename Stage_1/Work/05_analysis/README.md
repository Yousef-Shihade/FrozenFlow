# Step 5 — Analysis

**A Computer Vision Project — *Flow Matching as a Layer*** · Stage 1
University of Haifa

---

## Purpose

The notebook that pulls together everything the project's classification-baseline pipeline
needs to present and discuss. Nothing is measured here — it reads what steps 2–4 already
produced and assembles the five core results:

1. Accuracy table (every dataset × encoder × K × baseline)
2. Accuracy vs. training-set size, with error bars
3. Representative training/validation loss curves
4. Row-normalized confusion matrices, one per dataset
5. 2D feature visualization with prototypes overlaid

This completes Stage 1's classification-baseline pipeline: linear probe and image-derived
prototypes, on DTD and FGVC-Aircraft, with ResNet-18 on both and DINOv2 on Aircraft,
reproducible end to end from `Stage_1/Data/` through this notebook.

---

## Quick start

```text
1. Open  code/05_analysis.ipynb  in VS Code
2. Select the kernel:  Python (CVLAB Stage 1)
3. Run All
```

**Runtime: seconds.** Requires steps 2–4's saved results (`Stage_1/results/`) — nothing is
retrained or re-extracted.

---

## Headline results

Full-data accuracy, both baselines:

| Combination | Linear probe | Prototype | Gap |
| --- | --- | --- | --- |
| ResNet-18 / DTD | 62.82% | 58.83% | +3.99 pp |
| ResNet-18 / FGVC-Aircraft | 38.15% | 25.47% | +12.68 pp |
| DINOv2 / FGVC-Aircraft | 67.55% | 34.26% | **+33.29 pp** |

Full combined table (18 rows, all K × both methods) in `tables/combined_accuracy_table.csv`.

---

## Two decisions in the feature-visualization section

Four of the five results re-plot what steps 2–4 already computed. The feature visualization
is the one that needs genuinely new code, and two choices inside it decide whether the plot
answers anything at all.

### 1. The features must be normalized before the joint projection

Concatenating **raw** test features with **unit-norm** prototypes and projecting the mix
directly does not work:

```python
combined = np.concatenate([feats_sel, proto_sel], axis=0)  # feats_sel: norm ~24-50
                                                             # proto_sel: norm exactly 1
proj = TSNE(...).fit_transform(combined)
```

Step 2 measured ResNet-18/DINOv2 feature norms at roughly 24–50 — never close to the
prototypes' unit norm. Section 6a of this notebook demonstrates the effect directly: with
unnormalized features, all 9 prototype stars get crushed into a single corner of the plot
regardless of the actual class geometry (`plots/normalization_effect.png`, left panel).
L2-normalizing the test features before the joint fit — the same normalization the
prototypes themselves are built from — fixes it (right panel): each star lands inside or
near its own class's point cloud, which is the actual question this plot is supposed to let
a reader answer.

### 2. Class selection has to be systematic, not index-based

```python
SELECTED_CLASSES = list(range(10))   # "arbitrary but deterministic"
```

For FGVC-Aircraft, class indices are alphabetically sorted, so `range(10)` selects
`707-320, 727-200, 737-200, 737-300, 737-400, 737-500, 737-600, 737-700, 737-800, 737-900`
— eight of ten are 737 variants, close to the worst possible choice for a *readable* subset
of visually distinct classes.

We use `pick_diverse_classes` instead: an evenly spaced stride through the sorted class list
(`np.linspace(0, n-1, 9).astype(int)`), which selects nine genuinely different aircraft
families — `707-320, 747-300, A319, BAE 146-200, Cessna 525, DHC-8-300, Falcon 2000, MD-80,
Yak-42` — and nine visually distinct DTD textures. Deterministic and systematic rather than
either arbitrary or hand-picked.

One more decision alongside these: confusion-matrix predictions are loaded from step 3's
saved `linear_probe_predictions.pt` rather than retraining probes just to obtain them, which
removes any risk of the confusion matrix quietly describing a slightly different model from
the one in the accuracy table.

---

## Confusion matrices: readability at scale

DTD (47 classes) gets full axis tick labels — dense but legible. FGVC-Aircraft (100 classes)
does not: a 100×100 grid of text labels isn't readable at any figure size that also fits a
report page. Instead, `tables/top_confusions.csv` lists every off-diagonal cell sorted by
confusion rate, which is a more direct answer to "what are the main classification errors"
than asking a reader to scan a 10,000-cell heatmap for bright spots.

**What the top confusions actually show:**

| Dataset | Pattern |
| --- | --- |
| DTD | Confusions cluster among related pattern concepts — `dotted` ↔ `polka-dotted` (40.0% / 25.0%), `woven` → `braided` (20.0%), `grid` → `meshed` (20.0%) |
| FGVC-Aircraft | Confusions cluster **within manufacturer families** — `C-47` ↔ `DC-3` (48.5% / 44.1%, both classic twin-prop transports), `A340-200` → `A340-300` (38.2%), `BAE 146-300` ↔ `BAE 146-200` (35.3% / 30.3%) |

This is the same fine-grained, low-inter-class-distance structure that step 4 identified as
the reason prototypes struggle far more than the linear probe on Aircraft — visible here at
the level of specific class pairs rather than an aggregate accuracy number.

---

## Feature visualization: what the projections show

PCA rather than t-SNE, deliberately — it is deterministic (no perplexity/seed sensitivity to
justify) and "fit jointly on features and prototypes" has an unambiguous meaning for a linear
projection, keeping section 6 about the normalization step rather than about
projection-hyperparameter choices.

On FGVC-Aircraft, DINOv2's 9 classes visibly separate into tighter, more distinct clusters
than ResNet-18's — consistent with, not new evidence for, the accuracy and separability
numbers already established in steps 2–4. As with any 2D projection, this is read
qualitatively: it is not a performance measurement.

---

## Outputs

### Tables (`tables/`)

| File | Contents |
| --- | --- |
| `methodology.csv` | The 5 results and where each is produced |
| `combined_accuracy_table.csv` | Result 1 — 18 rows, both baselines |
| `top_confusions.csv` | Every off-diagonal confusion, sorted by rate (976 rows) |

### Plots (`plots/`)

| File | Shows |
| --- | --- |
| `accuracy_vs_k.png` | Result 2 — both baselines vs. K, error bars |
| `loss_curves.png` | Result 3 — representative 10-shot train/val loss, all 3 combinations |
| `confusion_matrices.png` | Result 4 — DTD (labeled) + Aircraft (unlabeled, see table) |
| `normalization_effect.png` | Before/after: why prototypes must be normalized before a joint projection |
| `feature_viz_dtd.png` | Result 5 — ResNet-18 / DTD, 9 classes + prototypes |
| `feature_viz_aircraft.png` | Result 5 — ResNet-18 vs. DINOv2, same 9 classes/images/colors |

---

## Stage 1 status

All five results are produced and all five notebooks execute cleanly end to end from
`Stage_1/Data/`. Reproducibility was verified directly: a full clean re-run of all 5
notebooks from a deleted `features/`/`results/` cache reproduced every saved number
**bit-for-bit identical** to the prior run — independent confirmation that nothing in the
pipeline depends on hidden state.
