# Stage 1 — Classification Baselines with Frozen Encoders

**A Computer Vision Project — *Flow Matching as a Layer***
University of Haifa · 2026

Yousef Shihade · Mira Bitar

---

## Goal

Establish an honest, reproducible baseline before any Flow Matching is introduced in Stage
2/3. A pretrained encoder is frozen, its features are cached once, and two classifiers are
trained on top of those frozen features under an identical protocol:

1. A trained **linear probe**, $s = Wz + b$ — the reference for how much the frozen features
   can give when a classifier is actually fitted.
2. A training-free **prototype classifier** — one mean feature vector per class, classify by
   cosine similarity to the nearest prototype. This is the baseline Stage 2/3 build on top of.

No Flow Matching happens in Stage 1. Its entire purpose is a trustworthy point of comparison:
without it, any later improvement is indistinguishable from noise — which is why Stage 1 is
strict about official splits, fixed seeds, and cached features rather than re-extracting them
per experiment.

---

## Datasets

| Dataset | Classes | Train / Val / Test | Character |
| --- | --- | --- | --- |
| DTD (partition 1) | 47 | 1,880 / 1,880 / 1,880 | Broad, visually distinct textures — the "easy" case |
| FGVC-Aircraft (`variant`) | 100 | 3,334 / 3,333 / 3,333 | Near-identical airframes — the fine-grained "hard" case |

The pair gives a genuine easy-vs-hard contrast: DTD's 47 texture classes are broad and
visually distinct, while Aircraft's 100 variants differ by subtle details such as engine
placement and tail geometry.

**Frozen encoders** — ResNet-18 (ImageNet-1K, 512-d) on both datasets; DINOv2 ViT-S/14
(384-d CLS token) on Aircraft.

**Protocol** — $K \in \{5, 10, \text{full}\}$; subset seeds $\{0, 1, 2\}$ for 5/10-shot;
three classifier initialisation seeds for the full linear probe; top-1 accuracy on the
complete official test split, reported as mean ± sample std.

Both classifiers are trained and evaluated on the same frozen features and the same k-shot
subsets, so any accuracy gap between them comes from the classifier design rather than one
seeing different data.

---

## Pipeline

| Step | Does | Needs GPU | Runtime |
| --- | --- | --- | --- |
| 01 · Data setup | Verifies the protocol: splits, partitions, disjointness, K-shot feasibility | no | ~2 min |
| 02 · Feature extraction | Runs the frozen encoders once, caches features to `features/` | yes | ~5 min |
| 03 · Linear probe | 27 runs + a 6-run epoch-budget check | optional | ~4 min |
| 04 · Prototypes | 21 runs: nearest-class-mean on cached features | no | seconds |
| 05 · Analysis | Consolidated accuracy tables, plots, confusion matrices, feature visualization | no | seconds |

After Step 2, nothing touches the images or the encoders again — Steps 3–5 operate purely
on cached vectors, which is what makes the whole sweep cheap enough to re-run at will.

---

## Results

### Step 01 — Data setup ([details](Work/01_data_setup/README.md))

All 20 protocol assertions pass: partitions, annotation level, split sizes, pairwise
disjointness, on-disk presence, and K-shot feasibility.

Its most useful finding: the standard 256-resize + 224 centre crop keeps only **~59% of each
aircraft image's horizontal extent**, and aircraft variants are distinguished precisely by
details near the wingtips and tail. The standard preprocessing is kept regardless, but this
is a quantified, defensible explanation for the dataset's difficulty.

### Step 02 — Feature extraction ([details](Work/02_feature_extraction/README.md))

9 feature caches (45.4 MB) covering 25,640 forward passes, all verified. From here the
encoders are never loaded again.

Its most useful finding is a **training-free explanation of the ResNet-18 / DINOv2 gap**.
Mean cosine similarity of each test feature to its own class centroid vs. to all other class
centroids:

| Combination | own | other | margin |
| --- | --- | --- | --- |
| ResNet-18 / DTD | 0.759 | 0.587 | 0.172 |
| ResNet-18 / FGVC-Aircraft | 0.903 | 0.862 | **0.041** |
| DINOv2 / FGVC-Aircraft | 0.745 | 0.325 | **0.420** |

In ResNet-18's feature space every aircraft resembles every other aircraft, leaving almost
no margin for a classifier to exploit. DINOv2 spreads the same 100 classes over a margin ten
times wider. Steps 03–04 should reproduce this ranking; if they do not, the fault is in the
classifier code rather than the features.

Step 02 also **tested a step 01 recommendation rather than trusting it**: the 20 px
copyright-banner crop turns out to make no measurable difference (cosine 0.982; accuracy gap 0.78 pp against a ±1.55 pp
standard error). It is kept as correct hygiene, not as an improvement.

### Step 03 — Linear probe ([details](Work/03_linear_probe/README.md))

27 runs, using a standard configuration (AdamW, lr 1e-3, weight decay 1e-4, batch size 64,
up to 200 epochs, checkpoint on highest validation accuracy) unmodified. Results:

| Dataset / encoder | 5-shot | 10-shot | full |
| --- | --- | --- | --- |
| ResNet-18 / DTD | 44.86 ± 1.83 | 52.04 ± 1.16 | 62.82 ± 0.42 |
| ResNet-18 / FGVC-Aircraft | 20.76 ± 0.26 | 27.87 ± 1.33 | 38.15 ± 0.29 |
| DINOv2 / FGVC-Aircraft | 38.19 ± 2.49 | 51.54 ± 1.70 | 67.55 ± 0.28 |

Reproduces the step-2 separability ranking exactly, and **two independent implementations
of the protocol agree**: DTD (which gets no banner crop) matches to within 0.26 pp on every
K setting despite re-extracted features and a rewritten training loop — strong evidence
both implementations are correct.

A late-checkpoint pattern on Aircraft (some runs select an epoch near the 200-epoch budget)
initially looked like it might be costing accuracy. It was tested directly: tripling the
epoch budget moves test accuracy by at most 0.21 pp. The concern did not hold up, and the
notebook says so rather than quietly dropping it.

A separate experiment (30 runs, not part of the notebook) checked three adjustments —
L2-normalizing features, standardizing them, and a longer epoch budget — against the
baseline configuration. None gained more than **+0.96 pp anywhere**, and most made results
worse. The baseline configuration is already close to the ceiling available without
touching the frozen encoders themselves — which is expected, since Stage 1 is meant to be a
fair reference point for Stage 2/3's Flow-Matching layer, not a maximized number.

### Step 04 — Image-derived prototypes ([details](Work/04_prototypes/README.md))

21 runs, no training, seconds of compute. All 21 numbers match our first implementation's
results exactly (prototype computation is deterministic given a fixed subset).

| Dataset / encoder | 5-shot | 10-shot | full |
| --- | --- | --- | --- |
| ResNet-18 / DTD | 46.10 ± 1.89 | 51.67 ± 1.33 | 58.83 (1 run) |
| ResNet-18 / FGVC-Aircraft | 16.46 ± 0.39 | 20.19 ± 1.14 | 25.47 (1 run) |
| DINOv2 / FGVC-Aircraft | 23.79 ± 0.47 | 27.74 ± 0.98 | 34.26 (1 run) |

Close to the linear probe on DTD, far behind on Aircraft (probe wins by up to 33 pp) —
exactly what step 2's separability analysis predicted before either classifier was trained.

Its own finding: aggregate accuracy hides massive per-class unevenness. **85 of 100
ResNet-18/Aircraft classes score below chance-level** own-prototype accuracy even though the
overall figure (25.4%) sits far above random guessing — a handful of distinctive variants
carry the average while most classes are barely separated by their mean at all.

### Step 05 — Analysis ([details](Work/05_analysis/README.md))

Assembles the headline outputs from steps 2–4's saved results — nothing is retrained or
re-extracted. Full-data headline:

| Combination | Linear probe | Prototype | Gap |
| --- | --- | --- | --- |
| ResNet-18 / DTD | 62.82% | 58.83% | +3.99 pp |
| ResNet-18 / FGVC-Aircraft | 38.15% | 25.47% | +12.68 pp |
| DINOv2 / FGVC-Aircraft | 67.55% | 34.26% | **+33.29 pp** |

Its feature-visualization section turns on two decisions. First, prototypes are unit-norm by
construction, so projecting them jointly with **raw** test features (norm ~24–50 per step 2)
would collapse every prototype into one corner regardless of the actual geometry — the
features are L2-normalized first, demonstrated directly with a before/after comparison.
Second, classes are selected by a systematic stride through the sorted class list: picking
by raw index would have given 8 near-duplicate 737 variants out of 10 for FGVC-Aircraft.

The confusion-matrix analysis also shows *what* gets confused, not just that some things do:
DTD errors cluster among related pattern concepts (`dotted` ↔ `polka-dotted`, 40%/25%), while
Aircraft errors cluster within manufacturer families (`C-47` ↔ `DC-3`, 48.5%/44.1%) — the
same fine-grained structure step 4 identified as the reason prototypes lose so much ground
to the linear probe on that dataset.

### End-to-end reproducibility

Verified independently: `features/` and `results/` were deleted and all 5 notebooks re-run
from scratch, in order. Every saved result — all 27 linear-probe runs, all 21 prototype
runs, every test prediction, every prototype vector — came back **bit-for-bit identical** to
the prior run. Nothing in the pipeline depends on hidden state, execution order beyond the
documented step sequence, or anything not captured in `Data/` plus the code itself.

Stage 1 is complete and independently reproducible end to end.

---

## Folder layout

```text
Stage_1/
├── README.md                  this file
├── pyproject.toml             makes `cvlab` installable
├── Reports/
│   └── Stage1Report.pdf       the written report for this stage
├── Data/                      datasets (NOT in git, ~3.2 GB)
├── features/                  cached feature tensors (NOT in git, regenerable)
├── results/                   shared run outputs consumed across steps (~1.4 MB, committed)
├── src/cvlab/                 the shared library
│   ├── paths.py                project path resolution
│   ├── data.py                 dataset loading, k-shot sampling, protocol constants
│   ├── encoders.py             frozen encoders + checkpoint preprocessing
│   ├── features.py             extraction loop and the feature cache
│   ├── probe.py                the linear-probe training loop
│   ├── prototypes.py           image-derived class prototypes
│   ├── evaluation.py           mean/std aggregation, one ddof convention project-wide
│   └── plotting.py             shared plot style, artefact writers
└── Work/                      one folder per pipeline step
    ├── 01_data_setup/           DONE (8 plots, 12 tables)
    ├── 02_feature_extraction/   DONE (6 plots, 8 tables)
    ├── 03_linear_probe/         DONE (5 plots, 7 tables)
    ├── 04_prototypes/           DONE (4 plots, 6 tables)
    └── 05_analysis/             DONE (6 plots, 4 tables)
```

Each step folder holds `README.md` (what it does and found), `code/*.ipynb` (with outputs, as
reproducibility evidence), `plots/` and `tables/`.

### Why a package *and* notebooks

The library holds the **plumbing** — paths, dataset loading, encoders, classifiers, plot
style. The notebooks hold the **experiment** — what is measured, why, and what it means.

The split is not decoration. With `load_features()` and the k-shot sampler pasted into
three notebooks each, and the project root string repeated in all five, a fix applied to one
copy silently leaves the others wrong — precisely the class of error that produces
plausible-but-invalid numbers. Writing each piece of logic once also
means the linear probe and the prototype baseline provably train on *identical* k-shot
subsets, so any accuracy gap between them comes from the classifier design rather than a
luckier draw.

Each step keeps its own `plots/` and `tables/` so every figure sits beside the notebook
that produced it, and every number in the report is traceable to a CSV.

---

## Reproducing these results

### 1. Python environment

Requires Python 3.10+ and an NVIDIA GPU for Step 2 (Steps 1 and 3–5 run fine on CPU).

```powershell
# Create an environment OUTSIDE any cloud-synced folder - it is ~5 GB
python -m venv C:\cvlab_env

# PyTorch first, from the CUDA index. Plain PyPI wheels are CPU-only.
C:\cvlab_env\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 `
    --index-url https://download.pytorch.org/whl/cu124

C:\cvlab_env\Scripts\python.exe -m pip install -r requirements.txt
C:\cvlab_env\Scripts\python.exe -m pip install -e Stage_1

C:\cvlab_env\Scripts\python.exe -m ipykernel install --user --name cvlab `
    --display-name "Python (CVLAB Stage 1)"
```

Verify:

```powershell
C:\cvlab_env\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# -> 2.6.0+cu124 True
```

### 2. Datasets

Not in git. Download and extract into `Stage_1/Data/` so the layout is:

```text
Stage_1/Data/
├── dtd/dtd/{images,labels,imdb}                 <- dtd-r1.0.1.tar.gz  (~625 MB)
└── FGVC-Aircraft/fgvc-aircraft-2013b/data/      <- fgvc-aircraft-2013b.tar.gz  (~2.75 GB)
```

- DTD: <https://www.robots.ox.ac.uk/~vgg/data/dtd/>
- FGVC-Aircraft: <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>

The nesting is not arbitrary — torchvision appends its own sub-paths, and
`src/cvlab/paths.py` documents exactly why the two roots differ. Run notebook 01 to verify
the layout; it fails loudly if anything is wrong.

### 3. Run

Open any notebook under `Work/*/code/` in VS Code, select the kernel
**Python (CVLAB Stage 1)**, and Run All. Steps must run in order the first time, because
each consumes the previous step's artefacts.
