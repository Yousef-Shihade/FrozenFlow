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

Two frozen encoders (ResNet-18 and DINOv2 ViT-S/14), two datasets chosen for a genuine
easy-vs-hard contrast (DTD's broad textures vs. FGVC-Aircraft's near-identical airframes),
and two classifiers built on the same cached features and the same k-shot subsets: a trained
**linear probe** and a training-free **prototype classifier**.

Full-data headline:

| Combination | Linear probe | Prototype | Gap |
| --- | --- | --- | --- |
| ResNet-18 / DTD | 62.82% | 58.83% | +3.99 pp |
| ResNet-18 / FGVC-Aircraft | 38.15% | 25.47% | +12.68 pp |
| DINOv2 / FGVC-Aircraft | 67.55% | 34.26% | **+33.29 pp** |

The prototype classifier is Stage 2/3's baseline — everything a Flow Matching layer is
measured against. The gap above tracks a **training-free separability measure** computed
straight from the cached features (own-class vs. other-class cosine similarity): wide margin
on DINOv2/Aircraft, almost none on ResNet-18/Aircraft, predicting exactly which combination
the linear probe would win big on before either classifier was trained.

Full detail, every step's findings, and reproduction instructions:
**[Stage_1/README.md](Stage_1/README.md)**.

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
├── README.md                  <- this file, the project-wide overview
├── requirements.txt
├── .gitignore
├── Stage_1/                   <- classification baselines with frozen encoders
│   └── README.md              <- full detail: datasets, protocol, per-step results, setup
└── Stage_2/                   <- Flow Matching as the last layer
    └── README.md              <- full detail: objectives, results, design decisions, setup
```

(Stage 3 — Flow Matching *before* the last layer — is not started yet; its own `Stage_3/`
folder and README will appear here once it begins.)

Each stage is a self-contained, independently installable package (`cvlab` for Stage 1,
`cvlabfm` for Stage 2) with its own `Work/` folder (one subfolder per pipeline step, each
holding a notebook, its plots, and its tables) and its own `README.md` covering that stage's
goal, results, folder layout, and how to reproduce it. Later stages reuse earlier stages'
code directly rather than reimplementing it — Stage 2 imports `cvlab.prototypes`,
`cvlab.data`, `cvlab.features` and `cvlab.evaluation` — so a fix applied once cannot drift
between stages.

See **[Stage_1/README.md](Stage_1/README.md)** and **[Stage_2/README.md](Stage_2/README.md)**
for each stage's full folder layout.

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

Not in git — downloaded and extracted into `Stage_1/Data/`. Full instructions (sources,
expected layout, why the nesting matters): **[Stage_1/README.md § Reproducing these
results](Stage_1/README.md#reproducing-these-results)**.

### 3. Run

Open any notebook under `Stage_1/Work/*/code/` or `Stage_2/Work/*/code/` in VS Code, select
the kernel **Python (CVLAB Stage 1)**, and Run All. Steps must run in order the first time,
because each consumes the previous step's artefacts — and all of Stage 2 depends on Stage 1's
feature cache and results, so Stage 1 must be run first.

```powershell
C:\cvlab_env\Scripts\python.exe -m pytest Stage_2/tests -q   # 21 passed
```

checks the Stage 2 implementation against the brief's formulas.

---

## Pipeline

Stage 1 turns raw images into cached feature vectors once, then trains and evaluates both
classifiers purely on those cached vectors — no encoder is ever re-run after step 2. Stage 2
consumes Stage 1's cached features throughout and never touches an encoder either. Full
per-step tables (what each step does, GPU requirement, runtime):
**[Stage_1/README.md § Pipeline](Stage_1/README.md#pipeline)** ·
**[Stage_2/README.md § Experimental grid](Stage_2/README.md#experimental-grid)**.

---

## Status

### Stage 1

**All five steps are complete**, executed locally with outputs committed. The
classification-baseline pipeline — linear probe and image-derived prototypes, on DTD and
FGVC-Aircraft, ResNet-18 on both and DINOv2 on Aircraft — is reproducible end to end from
`Stage_1/Data/`. Per-step findings, including two bugs caught and fixed in an earlier draft's
feature-visualization step, live in **[Stage_1/README.md](Stage_1/README.md)**.

Reproducibility was verified directly: `features/` and `results/` were deleted and all 5
notebooks re-run from scratch, in order. Every saved result — all 27 linear-probe runs, all
21 prototype runs, every prototype vector — came back **bit-for-bit identical** to the prior
run.

Stage 1 is complete and independently reproducible end to end.

### Stage 2

**All five steps are complete**, with outputs committed and the written report in
`Stage_2/Reports/Stage2Report.pdf`. Per-step detail lives in each step's README; the
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

Stage 2 is complete and independently reproducible end to end. **Stage 3 — moving the Flow
Matching module *before* the last layer — is next.**
