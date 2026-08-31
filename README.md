# Flow Matching as a Layer - A Computer Vision Project

**University of Haifa** · 2026

**Authors:** Yousef Shihade & Mira Bitar

---

## The project

The research question is whether a **Flow Matching** module can replace a standard
classifier layer and do better than the conventional alternative. Answering that requires
a trustworthy point of comparison, which is what Stage 1 builds; Stage 2 then inserts the
Flow Matching layer on top of it and measures the difference.

| Stage | Topic | Status |
| --- | --- | --- |
| **1** | **Classification baselines with frozen encoders** — linear probe + image-derived prototypes | **complete** |
| **2** | **Flow Matching as the *last* layer** (standard vs. rolled-out training) | **complete** |
| 3 | Flow Matching *before* the last layer | not started |
| 4 | Extensions — structured tasks, encoder fine-tuning | not started |

Stage 1 uses no Flow Matching at all. Its entire purpose is to establish an honest,
reproducible baseline: freeze a pretrained encoder, cache its features once, and train
only a small classifier on top. Without that, any Stage 2/3 improvement is
indistinguishable from noise — which is why Stage 1 is strict about official splits, fixed
seeds, and cached features.

---

## Stage 1 at a glance

**Datasets**

| Dataset | Classes | Train / Val / Test | Character |
| --- | --- | --- | --- |
| DTD (partition 1) | 47 | 1,880 / 1,880 / 1,880 | Broad, visually distinct textures — the "easy" case |
| FGVC-Aircraft (`variant`) | 100 | 3,334 / 3,333 / 3,333 | Near-identical airframes — the fine-grained "hard" case |

The pair gives a genuine easy-vs-hard contrast: DTD's 47 texture classes are broad and
visually distinct, while Aircraft's 100 variants differ by subtle details such as engine
placement and tail geometry.

**Frozen encoders** — ResNet-18 (ImageNet-1K, 512-d) on both datasets; DINOv2 ViT-S/14
(384-d CLS token) on Aircraft.

**Baselines** — a linear probe and image-derived class prototypes, trained and evaluated on
the same frozen features and the same k-shot subsets, so any accuracy gap between them
comes from the classifier design rather than from one seeing different data.

**Protocol** — K ∈ {5, 10, full}; subset seeds {0, 1, 2} for 5/10-shot; three classifier
initialisation seeds for the full linear probe; top-1 accuracy on the complete official
test split, reported as mean ± std.

---

## Stage 2 at a glance

Stage 2 inserts a **flow-matching layer** between the frozen feature and Stage 1's prototype
classifier. A small velocity network transports each feature toward the prototype of its
class; the transported feature is then classified with the *same* cosine rule as Stage 1.
Everything else — encoders, cached features, prototypes, k-shot subsets, seeds, test splits —
is reused unchanged, so any accuracy difference is attributable to the FM layer alone.

**Two training objectives are compared.** *Standard FM* supervises the velocity at random
points on the ideal straight path to the prototype. *Rolled-out FM* runs the full T-step Euler
rollout used at inference and supervises only where the feature ends up. Both are evaluated at
T = 4 and T = 12, over a 45-cell grid (3 encoder/dataset combinations × 3 training sizes ×
5 methods) built from **81 trained models**.

Headline results at full data, T = 4:

| Encoder / dataset | Prototype baseline | Standard FM | Rolled-out FM |
| --- | --- | --- | --- |
| ResNet-18 / DTD | 58.83 | **59.59** (+0.76) | 55.85 (−2.98) |
| ResNet-18 / FGVC-Aircraft | 25.47 | **31.04** (+5.57) | 25.64 (+0.17) |
| DINOv2 / FGVC-Aircraft | 34.26 | **56.52** (+22.26) | 52.05 (+17.79) |

Three findings worth stating up front:

- **The FM layer works, and works best where prototypes were weakest** — up to **+22.3 pp**.
  24 of 36 FM cells beat the baseline.
- **Rolled-out training does not beat standard FM**, winning only 4 of 18 head-to-head cells
  despite removing the train/inference mismatch it was designed to remove. It reaches the
  prototype *more* closely while raising similarity to competing prototypes just as much — a
  better *transport*, not a better *decision*.
- **One quantity predicts every cell.** The gain tracks the **headroom** Stage 1 left behind
  (trained linear probe minus prototype baseline) with Spearman ρ = 1.000 — and its 2.2 pp
  break-even explains exactly the two cells where the FM layer loses.

Full detail, including the L2-normalization ablation and every figure and table:
**[Stage_2/README.md](Stage_2/README.md)**.

---

## Repository structure

```text
ProjectInComputerVision/
├── README.md                  <- this file
├── requirements.txt
├── .gitignore
├── Stage_1/
│   ├── pyproject.toml         <- makes `cvlab` installable
│   ├── Data/                  <- datasets (NOT in git, ~3.2 GB)
│   ├── features/              <- cached feature tensors (NOT in git, regenerable)
│   ├── results/               <- shared run outputs consumed across steps (~1.4 MB, committed)
│   ├── src/cvlab/             <- the shared library
│   │   ├── paths.py           <- project path resolution
│   │   ├── data.py            <- dataset loading, k-shot sampling, protocol constants
│   │   ├── encoders.py        <- frozen encoders + checkpoint preprocessing
│   │   ├── features.py        <- extraction loop and the feature cache
│   │   ├── probe.py           <- the linear-probe training loop
│   │   ├── prototypes.py      <- image-derived class prototypes
│   │   ├── evaluation.py      <- mean/std aggregation, one ddof convention project-wide
│   │   └── plotting.py        <- shared plot style, artefact writers
│   └── Work/                  <- one folder per pipeline step
│       ├── 01_data_setup/           <- DONE (8 plots, 12 tables)
│       ├── 02_feature_extraction/   <- DONE (6 plots, 8 tables)
│       ├── 03_linear_probe/         <- DONE (5 plots, 7 tables)
│       ├── 04_prototypes/           <- DONE (4 plots, 6 tables)
│       └── 05_analysis/             <- DONE (6 plots, 4 tables)
└── Stage_2/
    ├── pyproject.toml         <- makes `cvlabfm` installable
    ├── docs/                  <- the Stage 2 brief (NOT in git)
    ├── results/               <- run tables, curves, prototypes, test predictions
    ├── src/cvlabfm/           <- the Flow Matching layer
    │   ├── flow.py            <- velocity network, Euler rollout, both training loops
    │   └── paths.py           <- Stage 2 paths; re-exports Stage 1's so they cannot drift
    ├── tests/                 <- the brief's four formulas, checked against the code
    ├── Reports&Demos/         <- Stage2Report.pdf, the written report
    └── Work/
        ├── 01_setup_prototypes/  <- DONE (2 plots, 6 tables)
        ├── 02_standard_fm/       <- DONE (5 plots, 5 tables)
        ├── 03_rolled_out_fm/     <- DONE (7 plots, 5 tables)
        ├── 04_evaluation/        <- DONE (5 plots, 7 tables)
        └── 05_visualizations/    <- DONE (9 plots, 4 tables)
```

Stage 2 reuses Stage 1's code directly — `cvlab.prototypes`, `cvlab.data`, `cvlab.features`,
`cvlab.evaluation` are all imported rather than reimplemented. `cvlabfm` holds only what did
not exist before: the velocity network, the Euler rollout, and the two training objectives.

Each step folder holds `README.md` (what it does and found), `code/*.ipynb`, `plots/` and
`tables/`. Stage 1's steps additionally keep `_original_colab/` — an earlier Colab-based
version of that step's notebook, preserved unmodified so the comparison tables in those
READMEs are verifiable rather than asserted.

### Why a package *and* notebooks

The library holds the **plumbing** — paths, dataset loading, encoders, classifiers, plot
style. The notebooks hold the **experiment** — what is measured, why, and what it means.

The split is not decoration. In an earlier draft, `load_features()` and the k-shot sampler
were pasted into three notebooks each and the project root string appeared in all five. A
fix applied to one copy silently leaves the others wrong, and that is precisely the class
of bug that produces plausible-but-invalid numbers. Writing each piece of logic once also
means the linear probe and the prototype baseline provably train on *identical* k-shot
subsets, so any accuracy gap between them comes from the classifier design rather than a
luckier draw.

Each step keeps its own `plots/` and `tables/` so every figure sits beside the notebook
that produced it, and every number in the report is traceable to a CSV.

---

## Setup

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
C:\cvlab_env\Scripts\python.exe -m pip install -e Stage_2

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
`Stage_1/src/cvlab/paths.py` documents exactly why the two roots differ. Run notebook 01
to verify the layout; it fails loudly if anything is wrong.

### 3. Run

Open any notebook under `Stage_1/Work/*/code/` or `Stage_2/Work/*/code/` in VS Code, select
the kernel **Python (CVLAB Stage 1)**, and Run All. Steps must run in order the first time,
because each consumes the previous step's artefacts — and all of Stage 2 depends on Stage 1's
feature cache and results, so Stage 1 must be run first.

To check the Stage 2 implementation against the brief's formulas:

```powershell
C:\cvlab_env\Scripts\python.exe -m pytest Stage_2/tests -q   # 21 passed
```

---

## Pipeline

### Stage 1

| Step | Does | Needs GPU | Runtime |
| --- | --- | --- | --- |
| 01 · Data setup | Verifies the protocol: splits, partitions, disjointness, K-shot feasibility | no | ~2 min |
| 02 · Feature extraction | Runs the frozen encoders once, caches features to `features/` | yes | ~5 min |
| 03 · Linear probe | 27 runs + a 6-run epoch-budget check | optional | ~4 min |
| 04 · Prototypes | 21 runs: nearest-class-mean on cached features | no | seconds |
| 05 · Analysis | Consolidated accuracy tables, plots, confusion matrices, feature visualization | no | seconds |

After Step 2, nothing touches the images or the encoders again — Steps 3–5 operate purely
on cached vectors, which is what makes the whole sweep cheap enough to re-run at will.

### Stage 2

| Step | Does | Needs GPU | Runtime |
| --- | --- | --- | --- |
| 01 · Setup & prototypes | Rebuilds Stage 1's baseline from cached features and verifies it exactly; settles the L2-normalization decision | no | seconds |
| 02 · Standard FM | 27 models on the ideal straight path, plus the normalization ablation | yes | ~6 min |
| 03 · Rolled-out FM | 54 models trained through the full T-step rollout (one per T) | yes | ~40 min |
| 04 · Evaluation | The 45-cell accuracy table, ΔAcc, accuracy-vs-K, training curves, headroom analysis | no | ~1 min |
| 05 · Visualizations | Feature-space comparisons, flow trajectories, per-step accuracy and distance | no | ~1 min |

Stage 2 never re-runs an encoder either — it consumes Stage 1's cached features throughout.

---

## Status — Stage 1

**All five steps are complete**, executed locally with outputs committed. Stage 1's
classification-baseline pipeline — linear probe and image-derived prototypes, on DTD and
FGVC-Aircraft, ResNet-18 on both and DINOv2 on Aircraft — is reproducible end to end from
`Stage_1/Data/`.

### Step 01 — Data setup ([details](Stage_1/Work/01_data_setup/README.md))

All 20 protocol assertions pass: partitions, annotation level, split sizes, pairwise
disjointness, on-disk presence, and K-shot feasibility.

Its most useful finding: the standard 256-resize + 224 centre crop keeps only **~59% of each
aircraft image's horizontal extent**, and aircraft variants are distinguished precisely by
details near the wingtips and tail. The standard preprocessing is kept regardless, but this
is a quantified, defensible explanation for the dataset's difficulty.

### Step 02 — Feature extraction ([details](Stage_1/Work/02_feature_extraction/README.md))

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
times wider. Steps 03–04 should reproduce this ranking; if they do not, the bug is in the
classifier code rather than the features.

Step 02 also **walked back a step 01 recommendation**: the 20 px copyright-banner crop turns
out to make no measurable difference (cosine 0.982; accuracy gap 0.78 pp against a ±1.55 pp
standard error). It is kept as correct hygiene, not as an improvement.

### Step 03 — Linear probe ([details](Stage_1/Work/03_linear_probe/README.md))

27 runs, using a standard configuration (AdamW, lr 1e-3, weight decay 1e-4, batch size 64,
up to 200 epochs, checkpoint on highest validation accuracy) unmodified. Results:

| Dataset / encoder | 5-shot | 10-shot | full |
| --- | --- | --- | --- |
| ResNet-18 / DTD | 44.86 ± 1.83 | 52.04 ± 1.16 | 62.82 ± 0.42 |
| ResNet-18 / FGVC-Aircraft | 20.76 ± 0.26 | 27.87 ± 1.33 | 38.15 ± 0.29 |
| DINOv2 / FGVC-Aircraft | 38.19 ± 2.49 | 51.54 ± 1.70 | 67.55 ± 0.28 |

Reproduces the step-2 separability ranking exactly, and **independently reproduces an
earlier draft's results**: DTD (which gets no banner crop) agrees to within 0.26 pp on every
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

### Step 04 — Image-derived prototypes ([details](Stage_1/Work/04_prototypes/README.md))

21 runs, no training, seconds of compute. All 21 numbers match an earlier draft's results
exactly (prototype computation is deterministic given a fixed subset).

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

### Step 05 — Analysis ([details](Stage_1/Work/05_analysis/README.md))

Assembles the headline outputs from steps 2–4's saved results — nothing is retrained or
re-extracted. Full-data headline:

| Combination | Linear probe | Prototype | Gap |
| --- | --- | --- | --- |
| ResNet-18 / DTD | 62.82% | 58.83% | +3.99 pp |
| ResNet-18 / FGVC-Aircraft | 38.15% | 25.47% | +12.68 pp |
| DINOv2 / FGVC-Aircraft | 67.55% | 34.26% | **+33.29 pp** |

Fixes two real bugs in an earlier draft's feature-visualization section. First, prototypes
(unit-norm by construction) were projected jointly with **raw** test features (norm ~24–50
per step 2), collapsing every prototype into one corner regardless of the actual geometry —
fixed by L2-normalizing the features first, demonstrated directly with a before/after
comparison. Second, its class selection (`list(range(10))`) picked 8 near-duplicate 737
variants out of 10 for FGVC-Aircraft; fixed with a systematic diverse stride through the
sorted class list.

The confusion-matrix analysis also shows *what* gets confused, not just that some things do:
DTD errors cluster among related pattern concepts (`dotted` ↔ `polka-dotted`, 40%/25%), while
Aircraft errors cluster within manufacturer families (`C-47` ↔ `DC-3`, 48.5%/44.1%) — the
same fine-grained structure step 4 identified as the reason prototypes lose so much ground
to the linear probe on that dataset. 

### Final polish pass 

Two items tracked from step 05 are now closed out:

- **Step 02's cache-verification cell** used `reference._labels` directly instead of
  `cvlab.data.labels()`, contradicting the module's own accessor design. Fixed.
- **Full end-to-end reproducibility**, verified independently: `features/` and `results/`
  were deleted and all 5 notebooks re-run from scratch, in order, on the (now-fixed) code.
  Every saved result — all 27 linear-probe runs, all 21 prototype runs, every test
  prediction, every prototype vector — came back **bit-for-bit identical** to the prior run.
  Nothing in the pipeline depends on hidden state, execution order beyond the documented
  step sequence, or anything not captured in `Stage_1/Data/` plus the code itself.

Stage 1 is complete and independently reproducible end to end.

---

## Status — Stage 2

**All five steps are complete**, with outputs committed and the written report in
`Stage_2/Reports&Demos/Stage2Report.pdf`. Per-step detail lives in each step's README; the
consolidated account is **[Stage_2/README.md](Stage_2/README.md)**.

Everything the stage set out to produce is in place — the accuracy table with ΔAcc and error
bars, training curves for both objectives, feature-space comparisons under a jointly fitted
projection, flow trajectories, accuracy and prototype distances at every intermediate Euler
step, and the learned flow run in reverse from the prototypes.

Three checks were added because the conclusions depend on them:

- **Stage 1 is reproduced before anything is built on it.** All 21 baseline runs recomputed
  through Stage 2's own code path, max |Δ| = **0.00e+00**; the prototype tensors are
  byte-identical to Stage 1's saved file.
- **The L2-normalization decision is ablated, not assumed.** Training the identical objective
  on raw features costs **15.8–20.3 pp**, and on both ResNet-18 combinations leaves the flow
  *worse than no flow at all*.
- **The implementation is checked against the brief's formulas.** A 21-test suite re-derives
  the Euler step, both losses, and the classification rule from the brief's wording and
  asserts the code agrees. The suite was validated by deliberately breaking the
  implementation and confirming it noticed.

Reproducibility was verified the same way as Stage 1: re-training all **81 models** from
scratch returned every accuracy and every loss curve **bit-for-bit identical**, and all 108
reported accuracies recompute from the stored per-example predictions to within 6e-06 pp.

Stage 2 is complete and independently reproducible end to end. **Stage 3 — moving the Flow Matching module *before* the last layer — is next.**
