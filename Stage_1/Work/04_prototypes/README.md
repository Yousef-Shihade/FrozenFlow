# Step 4 — Image-Derived Class Prototypes

**A Computer Vision Project — *Flow Matching as a Layer*** · Stage 1
University of Haifa

---

## Purpose

The second baseline: image-derived prototypes, computed directly from class-mean features
rather than via zero-shot text prompts. No training: L2-normalize, average each class's
training features, re-normalize, classify by cosine similarity.

$$\mu_c = \text{normalize}\Big(\tfrac{1}{|S_c|}\sum_{i \in S_c}\text{normalize}(z_i)\Big)
\qquad \hat{y} = \arg\max_c \cos(z, \mu_c)$$

Runs on the **same cached features** and, for a given `(K, seed)`, the **same k-shot
subset** as the linear probe (`cvlab.data.make_kshot_subset` is shared by both). That
shared plumbing is what makes the step 3 vs step 4 comparison a statement about classifier
design, not an artifact of one baseline seeing different data.

---

## Quick start

```text
1. Open  code/04_prototypes.ipynb  in VS Code
2. Select the kernel:  Python (CVLAB Stage 1)
3. Run All
```

**Runtime: seconds.** Requires step 2's feature cache; sections 7–8 additionally require
step 3's outputs in `Stage_1/results/`.

---

## Results

| Dataset / encoder | 5-shot | 10-shot | full |
| --- | --- | --- | --- |
| ResNet-18 / DTD | 46.10 ± 1.89 | 51.67 ± 1.33 | 58.83 (1 run) |
| ResNet-18 / FGVC-Aircraft | 16.46 ± 0.39 | 20.19 ± 1.14 | 25.47 (1 run) |
| DINOv2 / FGVC-Aircraft | 23.79 ± 0.47 | 27.74 ± 0.98 | 34.26 (1 run) |

All 21 numbers match our first implementation's results exactly (e.g. ResNet-18/DTD
full: 58.83% here against 58.8% there). Prototype computation is deterministic given a
fixed subset, so this is an exact check rather than a within-noise comparison.

---

## Linear probe vs. prototypes

| Combination | probe − prototype (full-data) |
| --- | --- |
| ResNet-18 / DTD | +3.99 pp |
| ResNet-18 / FGVC-Aircraft | +12.68 pp |
| DINOv2 / FGVC-Aircraft | **+33.29 pp** |

**On DTD the two methods are close**, sometimes with prototypes slightly ahead at low K
(46.10 vs 44.86 at 5-shot). DTD's 47 texture classes are broad and visually homogeneous, so
a handful of examples already gives a representative class mean.

**On FGVC-Aircraft the probe wins by a wide, growing margin.** Step 2's separability
analysis predicted this before either classifier was trained: ResNet-18's separability
margin on Aircraft was 0.041 (features barely distinguish classes at all), so a *single*
mean vector per class is a poor summary — but the discriminative signal is still present in
the full feature, just not aligned with the direction the class mean happens to point in. A
trained linear layer can find the combination of dimensions that separates classes;
averaging cannot.

---

## The most useful finding: aggregate accuracy hides massive per-class unevenness

Section 6 asked what "25.4% accuracy" on Aircraft actually means at the class level.

| Combination | mean (= section 5's accuracy) | worst class | best class | classes below 50% |
| --- | --- | --- | --- | --- |
| ResNet-18 / DTD | 58.8% | 20.0% | 85.0% | 12 of 47 |
| ResNet-18 / FGVC-Aircraft | 25.4% | 0.0% | 91.2% | **85 of 100** |
| DINOv2 / FGVC-Aircraft | 34.2% | 0.0% | 100.0% | **70 of 100** |

**85 of 100 ResNet-18 Aircraft classes score below chance-level** own-prototype accuracy,
even though the aggregate figure (25.4%) sits far above the 1% random-guess rate. A handful
of visually distinctive variants — biplanes, large regional jets — carry the average, while
most classes are barely separated by their mean at all. Section 8 names the classes on both
sides: the top-5 gap on DINOv2/Aircraft (Global Express, Tu-154, Challenger 600, 777-300,
Falcon 2000) each lose 70–82 percentage points to the linear probe.

**One thing worth being explicit about:** the *mean* of this per-class metric is not new
information — averaged over all
test images it is arithmetically the same quantity as the accuracy already reported in
section 5, since "argmax similarity matches the true label" is the classification decision
itself. What is new is computing it *per class first*: the spread (0% to 100% on
DINOv2/Aircraft) is invisible in a single number and is the real explanation for why
prototypes lose so much ground on this dataset.

---

## Outputs

### Shared artefacts (`Stage_1/results/`) — consumed by step 5

| File | Contents |
| --- | --- |
| `prototype_runs.csv` | One row per run |
| `full_prototypes.pt` | Full-data prototype vectors per combination — needed for step 5's feature-visualization section |

### Tables (`tables/`)

`methodology.csv` · `run_grid.csv` · `accuracy_summary.csv` ·
`prototype_centrality.csv` · `probe_vs_prototype.csv` · `per_class_divergence.csv`

### Plots (`plots/`)

| File | Shows |
| --- | --- |
| `prototype_centrality.png` | Mean vs. min/max own-prototype accuracy per combination |
| `probe_vs_prototype.png` | Accuracy vs K, both baselines overlaid |
| `per_class_divergence.png` | Top-10 classes where the probe beats prototypes most |
| `similarity_margins.png` | Cosine-similarity confidence margins, correct predictions only |

---

## Method sanity check

Before running on real features, `compute_prototypes` is verified against a toy example
built by hand: two classes, each containing one unit-norm vector and one vector scaled 3–5×
in the same direction. If normalization were silently skipped, the larger-magnitude example
would pull the class mean off its true direction — an error that would still often "work
well enough" on real data, which is exactly why it needs an explicit check rather than relying on
downstream accuracy to catch it. The toy prototypes come out exactly `[1,0]` and `[0,1]`, as
expected.

---

## Next step

**Step 5 — Analysis.** Pulls together the headline results: the combined accuracy table
(this notebook's results plus step 3's), the accuracy-vs-K plot with the prototype baseline
included, the training curves (already produced in step 3), row-normalized confusion
matrices, and the 2D feature visualization with prototypes overlaid.

That last section is where step 2's feature-geometry finding becomes load-bearing:
ResNet-18 and DINOv2 features sit at norms of ~24–50, never close to 1, while every
prototype here is exactly unit-norm. Any projection that mixes the two without normalizing
first collapses the prototypes into a single point — a trap we can step around here
precisely because step 2 measured the norms rather than assuming them.
