# Step 2 — Feature Extraction

**A Computer Vision Project — *Flow Matching as a Layer*** · Stage 1
University of Haifa

---

## Purpose

This is the pivot point of Stage 1. The protocol here is strict: the encoders stay
**frozen**, run **exactly once**, and their outputs are **cached**, with all classifier
training done on the cache. After this notebook nothing in the project touches an image or
an encoder again — steps 3, 4 and 5 read plain tensors.

That constraint is not bookkeeping:

- The linear probe (step 3) and the prototype baseline (step 4) read the **same bytes**, so
  any accuracy gap between them comes from the classifier design rather than from one of
  them happening to get better features.
- A 27-run sweep becomes minutes instead of hours.
- Stage 2's Flow-Matching layer drops into exactly this position, on exactly these
  features — the only way its contribution can be isolated.

---

## Quick start

```text
1. Open  code/02_feature_extraction.ipynb  in VS Code
2. Select the kernel:  Python (CVLAB Stage 1)
3. Run All
```

**Runtime ~5 minutes** (4.7 min of extraction; allow ~8 on a cold file cache), GPU required
— or it falls back to CPU and takes considerably longer. Re-running is near-instant:
existing caches are reused unless `FORCE_REEXTRACT = True`.

Requires step 01 to have passed and the shared library to be installed
(`pip install -e Stage_1`).

---

## What gets extracted

| Encoder | Feature dim | Params | DTD | FGVC-Aircraft |
| --- | --- | --- | --- | --- |
| ResNet-18 (ImageNet-1K) | 512 | 11.18 M | yes | yes |
| DINOv2 ViT-S/14 | 384 | 22.06 M | — | yes |

Three (encoder, dataset) pairs × three splits = **9 cache files**, 25,640 image forward
passes, **45.3 MB** of cached features written to `Stage_1/features/`.

Cache files are self-describing: alongside the tensors each stores the encoder, dataset,
split, whether the banner was cropped, the exact transform, throughput, and library
versions. A cache whose provenance is unknown is a cache you cannot trust.

---

## Folder layout

```text
02_feature_extraction/
├── README.md                        <- you are here
├── code/
│   └── 02_feature_extraction.ipynb  <- the notebook (run this)
├── plots/                           <- 6 generated figures
└── tables/                          <- 8 generated CSVs
```

---

## Outputs

### Tables (`tables/`)

| File | Contents |
| --- | --- |
| `encoder_inventory.csv` | Both encoders: dims measured vs expected, params, trainable count, eval mode |
| `extraction_plan.csv` | The 9 files to produce, and which already exist |
| `extraction_log.csv` | Per-file wall-clock, throughput, peak VRAM |
| `cache_verification.csv` | 9 files reloaded from disk and checked against the dataset definitions |
| `feature_statistics.csv` | Norms, value ranges, sparsity per cache file |
| `pca_variance.csv` | Components needed for 50 / 90 / 95% of variance |
| `class_separability.csv` | Training-free class-structure diagnostic |
| `banner_ablation.csv` | Cropped vs un-cropped copyright banner |

### Plots (`plots/`)

| File | Shows |
| --- | --- |
| `preprocessing_pipeline.png` | One image of each dataset traced through all 4 preprocessing stages |
| `extraction_throughput.png` | Images/second and wall-clock per cache file |
| `feature_norms.png` | Norm distributions, value distributions, sparsity |
| `pca_explained_variance.png` | Cumulative explained variance per encoder |
| `class_separability.png` | Cosine to own vs other class centroids, and nearest-centroid accuracy |
| `banner_ablation.png` | Feature perturbation and task-level effect of the banner crop |

---

## Verification

Every cache file is reloaded **from disk** (not reused from memory) and checked on:
sample count, feature dimension, class count, absence of NaN/Inf, label range, and that the
cached labels are **identical, in order**, to the dataset's own labels.

That last check guards against the silent catastrophe of features and labels drifting out of
alignment — which would produce a model that trains happily and scores at chance, with no
error anywhere.

**Result: 9/9 PASS.**

---

## Findings

### 1. Why ResNet-18 struggles on Aircraft — measured before any training

The single most useful result in this notebook. For each test image, cosine similarity to
its own class centroid versus the mean similarity to all other class centroids:

| Combination | cos to own class | cos to other classes | **margin** | nearest-centroid |
| --- | --- | --- | --- | --- |
| ResNet-18 / DTD | 0.759 | 0.587 | **0.172** | 67.7% |
| ResNet-18 / FGVC-Aircraft | 0.903 | 0.862 | **0.041** | 40.4% |
| DINOv2 / FGVC-Aircraft | 0.745 | 0.325 | **0.420** | 44.6% |

In ResNet-18's feature space **every aircraft looks like every other aircraft** — mean
cosine similarity of 0.862 to the *wrong* classes. The representation has collapsed the
whole dataset into a narrow cone that roughly means "airliner", leaving a margin of 0.041
for any classifier to exploit. DINOv2 spreads the same 100 classes over a margin **ten times
wider**.

This is the mechanism behind the accuracy gap steps 3 and 4 will measure, established
without fitting anything. It also makes the section a regression test: if step 3 does not
rank the three combinations in this order, the bug is in the classifier code, not the
features.

> These accuracies are **diagnostics, not results**. The centroids are built from the test
> features themselves, so they are optimistically biased. Step 4 builds prototypes from the
> official *training* subsets and evaluates on the untouched test split; only those numbers
> are reportable. The *ranking* is what carries over.

### 2. Feature geometry — why normalisation is mandatory later

| Combination | mean L2 norm | % exactly zero | % negative |
| --- | --- | --- | --- |
| ResNet-18 / DTD | 23.6 | 7.3% | 0% |
| ResNet-18 / FGVC-Aircraft | 28.9 | 0.9% | 0% |
| DINOv2 / FGVC-Aircraft | 49.7 | 0.0% | 50.4% |

ResNet-18 features are post-ReLU: strictly non-negative and partly sparse. DINOv2's CLS
token is signed and dense. More importantly, **all of them sit far from unit norm** and the
two encoders differ by roughly 2×.

Consequence for step 5: any plot that puts raw features and unit-norm prototypes in the same
projection will show the prototypes collapsed into a single blob near the origin, regardless
of the data. Normalise first.

### 3. The banner crop makes no measurable difference — correcting step 01

Step 01 found the 20 px copyright banner and recommended cropping it with some confidence.
The ablation here tests that instead of assuming it:

| Measure | Value |
| --- | --- |
| Cosine similarity, cropped vs un-cropped feature of the same image | **0.982** |
| Mean relative L2 change | 19.2% |
| Nearest-centroid accuracy, banner kept | 32.00% |
| Nearest-centroid accuracy, banner cropped | 31.22% |
| Gap | 0.78 pp (≈7 images) vs a binomial SE of ±1.6 pp |

The feature *direction* — the only thing cosine classifiers and a learned linear map respond
to — is essentially unchanged, and the task-level difference is noise pointing in the
opposite direction from the one predicted.

**Step 01 was right that the banner exists and survives into the crop, but it is not
materially damaging results.** The crop is kept because feeding the encoder a photograph
rather than a photograph plus a copyright notice is more correct and costs nothing — but it
is hygiene, not an improvement. If asked "how much did that help?", the honest answer is
*"nothing measurable, and here is the ablation that shows it"*.

### 4. Intrinsic dimensionality

Components needed to explain 95% of test-feature variance: ResNet-18/DTD **231**,
ResNet-18/Aircraft **207**, DINOv2/Aircraft **104** (of 384). No representation is
degenerate, so poor accuracy in step 3 cannot be blamed on features that carry too little
information.

---

## Performance notes

The bottleneck is **CPU-side JPEG decode and resize, not the GPU**. Measured on FGVC-Aircraft:

| Stage (single-threaded microbenchmark) | Throughput |
| --- | --- |
| Reading files from disk | ~2,360 img/s |
| JPEG decode | ~112 img/s |
| Resize + crop transform | ~122 img/s |
| Decode + transform combined | ~58 img/s |

End-to-end with 4 workers and the GPU forward pass: **82–104 img/s**, total **4.7 minutes**
for all 25,640 images (ResNet-18 ~98–104 img/s, DINOv2 ~82 img/s). A first run on a cold OS
file cache is slower — roughly 55–92 img/s, about 7 minutes — since the 3.2 GB of JPEGs have
not been read before.

Peak VRAM was **0.46 GB** (ResNet-18) and **0.385 GB** (DINOv2) at batch size 64 — the 4 GB
card is almost empty, and raising the batch size would not help because the GPU is not the
constraint.

**`num_workers=4` is roughly 2× faster than 0** (58 vs 28 img/s at scale). This works only
because every transform, including `CropBottomBanner`, is defined at module level in the
installed `cvlab` package: Windows starts workers with the `spawn` method, which pickles the
dataset and its transform, and a class defined inside a function cannot be pickled. A
transform defined inside a function would therefore rule out workers entirely here.

Worker counts above 4 were slower (8 workers → 36 img/s) from spawn overhead and CPU
oversubscription.

---

## Next step

**Step 3 — Linear probe.** Train `s = Wz + b` on the cached features: 3 encoder-dataset
pairs × K ∈ {5, 10, full} × 3 seeds = 27 runs (AdamW, lr 1e-3, weight decay 1e-4, batch 64,
up to 200 epochs, checkpoint on best validation accuracy).

The k-shot sampler already exists in `cvlab.data.make_kshot_subset`, so step 4's prototype
baseline will draw **identical** subsets for a given `(K, seed)` — which is what makes the
two baselines directly comparable.

Expect step 3 to reproduce the separability ranking above. If it does not, suspect the
classifier code.
