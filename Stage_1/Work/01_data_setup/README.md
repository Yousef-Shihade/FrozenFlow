# Step 01 — Data setup and protocol verification

**A Computer Vision Project — *Flow Matching as a Layer*** · Stage 1
University of Haifa

---

## Purpose

Stage 1 exists to produce an *honest baseline* that Stages 2 and 3 will be measured
against. That comparison is only meaningful if the data protocol is followed exactly — so
this step's job is to make the protocol **verifiable rather than assumed**, before a single
feature is extracted.

A wrong DTD partition or an overlap between train and test would not crash anything. It
would quietly inflate every accuracy number in the project while still looking completely
reasonable. That is precisely the failure mode this notebook is built to catch.

---

## Quick start

```text
1. Open  code/01_data_setup.ipynb  in VS Code
2. Select the kernel:  Python (CVLAB Stage 1)
3. Run All
```

Requires the shared library to be installed once (see the root `README.md`):

```powershell
C:\cvlab_env\Scripts\python.exe -m pip install -e Stage_1
```

Runtime is roughly **1–2 minutes**, CPU only — no GPU needed at this step.
No network access: the notebook passes `download=False` and reads only local data.

---

## Folder layout

```text
01_data_setup/
├── README.md                 <- you are here
├── code/
│   └── 01_data_setup.ipynb   <- the notebook (run this)
├── plots/                    <- generated figures (.png)
└── tables/                   <- generated data (.csv)
```

Paths are **not** hardcoded. Everything resolves through the installed `cvlab` package
(`Stage_1/src/cvlab/`), which locates the project from its own file location. Moving or
renaming the project folder breaks nothing, and the notebook contains no `sys.path`
manipulation.

The notebook uses three modules:

| Import | Provides |
| --- | --- |
| `cvlab.paths` | `step_dirs()`, dataset roots, `describe()` |
| `cvlab.data` | `load_datasets()`, `image_files()`, `labels()`, `class_counts()`, `DATASET_SPECS` |
| `cvlab.plotting` | `apply_style()`, `ArtifactWriter`, shared colour maps |

---

## What the notebook checks

Section 4 runs **20 assertions** (10 per dataset) and fails loudly if any breaks.
All 20 currently pass. Results are saved to `tables/protocol_checks.csv`.

| Check | Why it matters |
| --- | --- |
| DTD has 47 classes, partition 1 | DTD ships **ten** partitions; this project fixes partition 1 throughout for a single, reproducible protocol |
| Aircraft has 100 classes, `variant` level | `variant` (100) is the fine-grained task — not `family` (70) or `manufacturer` (30) |
| Class list identical across all three splits | A mismatch would silently scramble label indices between splits |
| Split sizes match the official counts | Detects a truncated or partial extraction |
| **train/val, train/test, val/test are pairwise disjoint** | Splits are kept fully separate throughout; a leak inflates every reported accuracy invisibly |
| **Every referenced image exists on disk** | Split files are text and can reference images a partial extraction never wrote |
| **Every class has at least 10 training images** | The 10-shot setting silently requires this — better verified now than discovered in Step 3 |

The three checks in bold are the ones we treat as non-negotiable.

---

## Verified results

| Dataset | Classes | Train | Val | Test | Images/class | On disk |
| --- | --- | --- | --- | --- | --- | --- |
| DTD (partition 1) | 47 | 1,880 | 1,880 | 1,880 | exactly 40 | 5,640 JPEG · 0.59 GB |
| FGVC-Aircraft (variant) | 100 | 3,334 | 3,333 | 3,333 | 33–34 | 10,000 JPEG · 2.57 GB |

Training-set size at each K setting:

| Dataset | 5-shot | 10-shot | Full |
| --- | --- | --- | --- |
| DTD | 235 (12.5%) | 470 (25.0%) | 1,880 |
| FGVC-Aircraft | 500 (15.0%) | 1,000 (30.0%) | 3,334 |

---

## Outputs

### Tables (`tables/`)

| File | Contents |
| --- | --- |
| `environment.csv` | Python / PyTorch / CUDA / GPU versions — reproducibility record |
| `on_disk_inventory.csv` | JPEG counts and byte sizes found on disk vs. expected |
| `protocol_checks.csv` | The 20 assertions with PASS/FAIL and details |
| `split_summary.csv` | Images and classes per split |
| `class_distribution_dtd.csv` | Per-class image counts, all three DTD splits |
| `class_distribution_fgvc_aircraft.csv` | Per-class image counts, all three Aircraft splits |
| `class_balance_summary.csv` | min / max / mean / std of class sizes |
| `image_properties_sample.csv` | Width, height, aspect, mode for 1,500 sampled images per dataset |
| `image_properties_summary.csv` | Aggregated geometry statistics |
| `centre_crop_coverage.csv` | How much of each image survives the 256-resize + 224 centre crop |
| `kshot_feasibility.csv` | Exact training-set size for each K in {5, 10, full} |
| `_artifact_manifest.csv` | Index of everything produced |

### Plots (`plots/`)

| File | Shows |
| --- | --- |
| `split_sizes.png` | Official split sizes per dataset |
| `class_distribution_train.png` | Training images per class, with the 10-shot floor marked |
| `image_geometry.png` | Shorter-side and aspect-ratio distributions vs. the 256/224 targets |
| `centre_crop_coverage.png` | Fraction of each image surviving the centre crop |
| `aircraft_copyright_banner.png` | The burned-in 20 px banner and the recommended crop line |
| `samples_dtd.png` | 12 random DTD classes, one image each |
| `samples_aircraft.png` | 12 random Aircraft variants, one image each |
| `kshot_training_sizes.png` | Training-set size at each K setting (log scale) |

---

## Findings that change what Step 2 does

### 1. FGVC-Aircraft has a burned-in copyright banner

Every one of the 10,000 aircraft images carries a **20-pixel banner along the bottom
edge** with a copyright string rendered directly into the pixels — see
`plots/aircraft_copyright_banner.png`, where the zoomed strip shows text such as
`COPYRIGHT MICK BAJCAR` and `AIRLINERS.NET`. The 20 px figure is confirmed empirically:
the marked crop line lands exactly on the banner's top edge.

This is not noise. It is a consistent, high-contrast, text-shaped strip present in every
image, and with the standard resize-256 then crop-224 pipeline it survives into the crop,
so every aircraft feature vector would encode it. Because it is near-identical across
classes it mostly wastes representational capacity rather than actively misleading the
classifier, but removing it is free and strictly correct.

**Decision:** crop the bottom 20 px before resizing, in Step 2.

### 2. Source resolution is never the bottleneck

The shorter side is at least 300 px for DTD and 413 px for Aircraft, both comfortably
above the 256 resize target — so no image is being upscaled and the standard preprocessing
is not resolution-limited.

### 3. The centre crop discards a large, measurable part of every aircraft image

This is the most useful thing this notebook found, and it was not visible without
measuring. Aircraft photos cluster tightly at an aspect ratio of **1.47** — they are
nearly all the same wide landscape shape. After resizing the shorter side to 256, a 224
centre crop keeps only about **59% of the horizontal extent**; **all 1,500** sampled
aircraft images lose more than 25% of their long axis.

That matters specifically for this dataset: aircraft are long, thin, horizontally-extended
objects, and the details separating one *variant* from another — engine count and
placement, wingtip shape, tail geometry, fuselage length — sit near the horizontal
extremes. A centre crop is precisely the operation most likely to remove them. DTD is the
opposite case: textures are statistically uniform, so any crop is as representative as any
other (and it still loses 26% at the median, harmlessly).

**We keep the standard preprocessing anyway.** The preprocessing associated with each
pretrained checkpoint is kept unchanged, since deviating would break comparability both
with the pretrained encoders and with Stages 2–3. This is recorded as a **known, quantified
limitation** — and it is a strong, evidence-backed explanation for why Aircraft
accuracy is so much lower than DTD's.

### 4. Both datasets are near-balanced

DTD is exactly 40 images/class; Aircraft is 33–34 (std 0.48). So plain top-1 accuracy is a
fair metric and no class weighting is needed, now with evidence behind that choice.

---

## Dataset characteristics

`tables/kshot_feasibility.csv` quantifies the training-set growth across K settings: for
DTD the full training set is 8x the 5-shot set; for Aircraft it is 6.7x.

The pair gives a genuine easy-vs-hard contrast: DTD's 47 texture classes are broad and
visually distinct, while Aircraft's 100 variants differ by subtle details such as engine
placement and tail geometry.

---

## Next step

**Step 2 — Feature extraction.** Load the frozen encoders (ResNet-18 on both datasets,
DINOv2 ViT-S/14 on Aircraft), run every image through exactly once, and cache the feature
tensors to `Stage_1/features/`. That is the last step that touches images or the GPU in any
serious way — Steps 3–5 operate purely on cached vectors.

Carry forward into Step 2:

1. Crop the bottom 20 px from every FGVC-Aircraft image before the resize.
2. Define every transform at module level. Windows starts DataLoader workers with the
   `spawn` method, which pickles the dataset and its transform — a class defined *inside* a
   function cannot be pickled, which is what breaks `num_workers > 0` on Windows.
   Keeping the transforms in `cvlab.encoders` preserves them,
   measured in Step 2 to be worth roughly a 2x speed-up.
3. Batch size 64 rather than 128, to stay well inside 4 GB of VRAM.
