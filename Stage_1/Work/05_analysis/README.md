# Step 5 — Analysis

**CVLAB Summer Project — *Flow Matching as a Layer*** · Stage 1
University of Haifa · Dr. Simon Korman

---

## Purpose

The notebook that pulls together everything `stage_1.pdf` says we must be prepared to
present and discuss. Nothing is measured here — it reads what steps 2–4 already produced
and assembles the five required deliverables:

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

## Two problems fixed from the original notebook

The original `05_analysis` notebook (archived in `_original_colab/`) built all five
deliverables reasonably, with two real bugs in the feature-visualization section — the one
deliverable that involves genuinely new code rather than re-plotting existing results.

### 1. Prototypes projected without normalizing the features first

Its `plot_feature_viz` concatenated **raw** test features with **unit-norm** prototypes and
projected the mix directly:

```python
combined = np.concatenate([feats_sel, proto_sel], axis=0)  # feats_sel: norm ~24-50
                                                             # proto_sel: norm exactly 1
proj = TSNE(...).fit_transform(combined)
```

Step 2 measured ResNet-18/DINOv2 feature norms at roughly 24–50 — never close to the
prototypes' unit norm. Section 6a of this notebook reproduces the effect directly: with
unnormalized features, all 9 prototype stars get crushed into a single corner of the plot
regardless of the actual class geometry (`plots/normalization_bug_demo.png`, left panel).
L2-normalizing the test features before the joint fit — the same normalization the
prototypes themselves are built from — fixes it (right panel): each star lands inside or
near its own class's point cloud, which is the actual question this deliverable is supposed
to let a reader answer.

### 2. Class selection produced 8 near-duplicate classes out of 10

```python
SELECTED_CLASSES = list(range(10))   # "arbitrary but deterministic"
```

For FGVC-Aircraft, class indices are alphabetically sorted, so `range(10)` selects
`707-320, 727-200, 737-200, 737-300, 737-400, 737-500, 737-600, 737-700, 737-800, 737-900`
— eight of ten are 737 variants, close to the worst possible choice for a *readable* subset
of visually distinct classes.

Fixed with `pick_diverse_classes`: an evenly spaced stride through the sorted class list
(`np.linspace(0, n-1, 9).astype(int)`), which selects nine genuinely different aircraft
families — `707-320, 747-300, A319, BAE 146-200, Cessna 525, DHC-8-300, Falcon 2000, MD-80,
Yak-42` — and nine visually distinct DTD textures. Deterministic and systematic rather than
either arbitrary or hand-picked.

A smaller change alongside these: confusion-matrix predictions are loaded from step 3's
saved `linear_probe_predictions.pt` instead of retraining two probes just to get them, which
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
projection, keeping section 6 about the normalization fix rather than about
projection-hyperparameter choices.

On FGVC-Aircraft, DINOv2's 9 classes visibly separate into tighter, more distinct clusters
than ResNet-18's — consistent with, not new evidence for, the accuracy and separability
numbers already established in steps 2–4. Per the assignment's explicit caveat, this is read
qualitatively: a 2D projection is not a performance measurement.

---

## Outputs

### Tables (`tables/`)

| File | Contents |
| --- | --- |
| `requirements_compliance.csv` | The 5 deliverables and where each is produced |
| `combined_accuracy_table.csv` | Deliverable 1 — 18 rows, both baselines |
| `top_confusions.csv` | Every off-diagonal confusion, sorted by rate (976 rows) |

### Plots (`plots/`)

| File | Shows |
| --- | --- |
| `accuracy_vs_k.png` | Deliverable 2 — both baselines vs. K, error bars |
| `loss_curves.png` | Deliverable 3 — representative 10-shot train/val loss, all 3 combinations |
| `confusion_matrices.png` | Deliverable 4 — DTD (labeled) + Aircraft (unlabeled, see table) |
| `normalization_bug_demo.png` | Before/after: why prototypes must be normalized before a joint projection |
| `feature_viz_dtd.png` | Deliverable 5 — ResNet-18 / DTD, 9 classes + prototypes |
| `feature_viz_aircraft.png` | Deliverable 5 — ResNet-18 vs. DINOv2, same 9 classes/images/colors |

---

## What changed from the original Colab notebook

| Area | Original | Now |
| --- | --- | --- |
| Platform | Colab + Drive | Local, package-backed |
| Confusion-matrix predictions | Retrained 2 probes to get them | Loaded from step 3's saved `linear_probe_predictions.pt` |
| Confusion-matrix readability | Unlabeled axes, heatmap only | DTD labeled directly; Aircraft gets a sorted top-confusions table |
| Feature-viz normalization | Raw features + unit-norm prototypes projected together | Features L2-normalized to match prototypes before the joint fit |
| Feature-viz class selection | `list(range(10))` — 8/10 near-duplicate 737 variants | Systematic diverse stride — 9 distinct families/textures |
| Projection method | t-SNE (stochastic, perplexity-dependent) | PCA (deterministic, unambiguous "joint fit") |
| std convention | — | `cvlab/evaluation.py`, `ddof=1`, same as steps 3–4 |

---

## Stage 1 status

All five deliverables are produced and all five notebooks execute cleanly end to end from
`Stage_1/Data/`. Two final-polish items are also closed out: step 2's cache-verification cell
now uses `cvlab.data.labels()` instead of `reference._labels` directly, and a full clean
re-run of all 5 notebooks from a deleted `features/`/`results/` cache reproduced every saved
number **bit-for-bit identical** to the prior run — independent confirmation that nothing in
the pipeline depends on hidden state.
