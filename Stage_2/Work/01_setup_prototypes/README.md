# Step 1 — Setup & Class Prototypes

**A Computer Vision Project — *Flow Matching as a Layer*** · Stage 2
University of Haifa

---

## Purpose

Prepare everything the Flow Matching layer needs, and **prove the Stage 1 prototype baseline
is reproduced exactly** before anything is built on top of it.

Stage 2 inserts a learned transport step between the frozen feature $z$ and the prototype
classifier. That comparison is only meaningful if the starting point is bit-for-bit the same
baseline Stage 1 reported — so this step rebuilds all 21 prototype sets from the cached
features and checks every resulting accuracy against `Stage_1/results/prototype_runs.csv`.

---

## Nothing is re-extracted or retrained here

| | |
| --- | --- |
| Encoders run | **none** — no forward passes, no GPU work |
| Datasets loaded | **none** — no images are read |
| Stage 1 files touched | **read-only** |

The two inputs are Stage 1 artefacts that already exist on disk:

- `Stage_1/features/*.pt` — the 9 cached feature files
- `Stage_1/results/prototype_runs.csv` — the baseline accuracies to verify against

Prototype construction is deterministic given a fixed subset, so rebuilding is arithmetic on
tensors already in memory: **0.09 s** to load 25,640 feature vectors, **0.30 s** to build all
21 prototype sets.

---

## Quick start

```text
1. Open  code/01_setup_prototypes.ipynb  in VS Code
2. Select the kernel:  Python (CVLAB Stage 1)
3. Run All
```

**Runtime: under 5 seconds.** Requires Stage 1's feature cache and results; requires both
packages installed editable (`pip install -e Stage_1`, `pip install -e Stage_2`).

---

## The design decision: L2-normalize before the flow

This is the one choice Stage 2 has to make before writing any training code, and it is made
here rather than assumed later.

Class prototypes are **unit-norm by construction**. Raw encoder features are not — they sit
at norms of 24–50. The flow target is $u_i = p_{y_i} - z_i$, so when
$\lVert z_i \rVert \approx 30$ and $\lVert p_{y_i} \rVert = 1$, the prototype is lost in the
rounding and $u_i \approx -z_i$.

Quantified by the alignment between the target and "just cancel the feature",
$a = \cos(u,\, -z)$:

| Encoder / dataset | $\lVert z \rVert$ | $\lVert p - z \rVert$ | $a$ **raw** | $a$ **normalized** |
| --- | --- | --- | --- | --- |
| ResNet-18 / DTD | 23.61 ± 5.24 | 22.86 | **0.9995** | 0.3427 |
| ResNet-18 / FGVC-Aircraft | 28.98 ± 2.62 | 28.08 | **0.9999** | 0.2166 |
| DINOv2 / FGVC-Aircraft | 49.64 ± 0.77 | 48.90 | **0.9999** | 0.3499 |

On raw features the supervised direction is **99.99 % aligned with $-z$** — it carries
essentially no class information, and a velocity network trained on it would learn "shrink
toward the origin" rather than "move toward the right class". After normalization the
alignment drops to 0.22–0.35, so the prototype genuinely shapes the target.

**This is not a deviation from Stage 1.** `cvlab.prototypes` already L2-normalizes before
averaging and before the cosine rule, so the baseline is untouched; and cosine classification
is scale-invariant, so the reported accuracy is unaffected either way. Every subsequent step
operates on $\hat z = z / \lVert z \rVert$.

**The argument above is not left as an argument.** Step 2 ablates it — retraining the same
objective with `normalize=False` — and the alignment figures predict the outcome well: raw
features cost **15.8–20.3 pp**, and on both ResNet-18 combinations the flow ends up *worse
than no flow at all*. See step 2, "Was L2 normalization actually necessary?".

---

## Why 21 prototype sets and not 27

| $K$ | Sets per combination | Why |
| --- | --- | --- |
| 5 | 3 | the seed selects *which images* form the subset |
| 10 | 3 | same |
| full | **1** | nothing left to subsample; prototype computation is deterministic |

$3 \text{ combos} \times 7 = 21$.

This matters downstream: FM training at $K = \text{full}$ still runs three seeds, but there
the seed varies only the *network* initialisation and batch order — all three share the same
prototype targets. Full-data sets are therefore keyed `seed=NA`.

---

## Verification result

Every prototype set classifies the **complete official test split** through Stage 1's own
`run_prototype_classifier`, and the result is compared against what Stage 1 recorded.

> **max |Δ| across all 21 runs: `0.00e+00` pp — exact match.**

Because both prototype construction and the cosine rule are deterministic, this is an *exact*
check rather than a within-noise comparison. The notebook `assert`s it and halts Stage 2 if it
ever drifts.

### The baseline Stage 2 has to beat

| Encoder / dataset | $n_\text{train}$ (5 / 10 / full) | 5-shot | 10-shot | full |
| --- | --- | --- | --- | --- |
| ResNet-18 / DTD | 235 / 470 / 1880 | 46.10 ± 1.89 | 51.67 ± 1.33 | 58.83 |
| ResNet-18 / FGVC-Aircraft | 500 / 1000 / 3334 | 16.46 ± 0.39 | 20.19 ± 1.14 | 25.47 |
| DINOv2 / FGVC-Aircraft | 500 / 1000 / 3334 | 23.79 ± 0.47 | 27.74 ± 0.98 | 34.26 |

---

## Integrity checks

Beyond the baseline reproduction, the notebook asserts rather than eyeballs:

- all 9 bundles are `float32` features / `int64` labels, finite, with labels in range and
  matching lengths — a silent dtype or shape drift would poison every FM result downstream;
- every one of the 21 prototype sets is exactly unit-norm (`atol=1e-5`) — the invariant the
  flow target depends on;
- exactly one Stage 1 row matches each run, so a missing or duplicated baseline entry fails
  loudly instead of being averaged away.

---

## Outputs

### Shared artefact (`Stage_2/results/`) — consumed by steps 2–5

| File | Contents |
| --- | --- |
| `prototypes.pt` | 21 unit-norm prototype sets, keyed `{encoder}__{dataset}__K{K}__seed{seed\|NA}`, plus provenance metadata (3.1 MB) |

The k-shot subsets are deliberately **not** cached. They are regenerated on demand by
`cvlab.data.make_kshot_subset`, which is deterministic; caching them would create a second
source of truth that could silently drift from Stage 1's.

### Tables (`tables/`)

`environment.csv` · `feature_inventory.csv` · `feature_geometry.csv` · `run_grid.csv` ·
`prototype_verification.csv` · `methodology.csv`

### Plots (`plots/`)

| File | Shows |
| --- | --- |
| `feature_norm_mismatch.png` | Why the flow operates on normalized features: endpoint scales, target magnitude, and the $\cos(u, -z)$ alignment |
| `baseline_reproduction.png` | All 21 runs on the $y = x$ line, and the Stage 1 baseline the FM layer must improve on |

---

## Next step

**Step 2 — Standard FM training.** Implements the velocity network $v_\theta(z, t)$ and the
standard flow-matching objective

$$\mathcal{L}_\mathrm{FM} = \lVert v_\theta(z_t, t) - u_i \rVert_2^2,
\qquad z_t = (1-t)\hat z_i + t\,p_{y_i}, \qquad u_i = p_{y_i} - \hat z_i$$

trained on the L2-normalized features and the prototypes saved here. One model per
`(combination, K, seed)` — 27 models — each evaluated at both $T = 4$ and $T = 12$, since
standard FM training does not depend on the number of inference steps.
