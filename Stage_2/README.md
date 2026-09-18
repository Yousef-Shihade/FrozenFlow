# Stage 2 — Flow Matching to Class Prototypes

**A Computer Vision Project — *Flow Matching as a Layer***
University of Haifa · 2026

Yousef Shihade · Mira Bitar

**Full report:** [Stage_2/Reports/Stage2Report.pdf](Reports/Stage2Report.pdf)

---

## Goal

Insert a **flow-matching (FM) layer** between the frozen image feature and the prototype
classifier from Stage 1. Given a frozen feature $z_i$, a velocity network $v_\theta(z, t)$
transports it toward the fixed prototype $p_{y_i}$ of its class; the transported feature is
then classified with the *same* cosine-similarity rule as Stage 1.

Everything downstream of the encoder is reused unchanged — same datasets, encoders, cached
features, prototypes, k-shot subsets, seeds, and test splits — so any accuracy difference is
attributable to the FM layer alone.

Three questions to answer:

1. Does FM-based classification beat the Stage 1 prototype baseline?
2. Does **rolled-out** training beat **standard** FM training?
3. How does the learned transformation change the geometry of the feature space?

---

## The two training objectives

| | **Standard FM** | **Rolled-out FM** |
| --- | --- | --- |
| Sampling | $t \sim \mathcal{U}(0,1)$, $z_t = (1-t)z_i + t\,p_{y_i}$ | full $T$-step Euler rollout from $\hat z_0 = z_i$ |
| Target | velocity $u_i = p_{y_i} - z_i$ | final position $p_{y_i}$ |
| Loss | $\mathcal{L}_\mathrm{FM} = \lVert v_\theta(z_t, t) - u_i \rVert_2^2$ | $\mathcal{L}_\mathrm{roll} = \lVert \hat z_T - p_{y_i} \rVert_2^2$ |
| Backprop through | one velocity prediction | the complete sequence of $T$ predictions |
| Train/inference match | **mismatched** — trained on the ideal straight path, evaluated on its own rollout | **matched** — trains on exactly the states inference produces |

Inference is identical for both: $T$ Euler steps,
$\hat z_{k+1} = \hat z_k + \tfrac{1}{T} v_\theta(\hat z_k, \tfrac{k}{T})$, then classify
$\hat z_T$ by cosine similarity to the prototypes.

**Standard FM training is $T$-independent** — one trained model is evaluated at both
$T = 4$ and $T = 12$. **Rolled-out training is not** — since $T$ is part of the rollout that is
backpropagated through, it needs a separate model per $T$.

---

## Results

| Encoder / dataset | $K$ | Baseline | Standard FM ($T{=}4$) | Rolled-out FM ($T{=}4$) |
| --- | --- | --- | --- | --- |
| ResNet-18 / DTD | 5 | 46.10 | 44.47 (−1.63) | 40.99 (−5.11) |
| ResNet-18 / DTD | 10 | 51.67 | 50.11 (−1.56) | 44.77 (−6.90) |
| ResNet-18 / DTD | full | 58.83 | **59.59 (+0.76)** | 55.85 (−2.98) |
| ResNet-18 / Aircraft | 5 | 16.46 | 18.81 (+2.35) | 16.77 (+0.31) |
| ResNet-18 / Aircraft | 10 | 20.19 | 23.68 (+3.49) | 20.34 (+0.15) |
| ResNet-18 / Aircraft | full | 25.47 | **31.04 (+5.57)** | 25.64 (+0.17) |
| DINOv2 / Aircraft | 5 | 23.79 | 33.30 (+9.51) | **34.13 (+10.34)** |
| DINOv2 / Aircraft | 10 | 27.74 | 43.27 (+15.53) | **45.42 (+17.68)** |
| DINOv2 / Aircraft | full | 34.26 | **56.52 (+22.26)** | 52.05 (+17.79) |

**Five findings:**

1. **The FM layer works, and works best where prototypes were weakest** — up to **+22.3 pp**
   on DINOv2/Aircraft. 24 of 36 FM cells beat the baseline.
2. **Removing the train/inference mismatch does not help.** Rolled-out FM wins only 4 of 18
   head-to-head cells, despite training on exactly the states inference produces. It reaches
   the prototype *more* closely (on test data, so it is not overfitting) while raising
   similarity to competing prototypes just as much — a better *transport*, not a better
   *decision*.
3. **The number of Euler steps barely matters** — under 1 pp between $T = 4$ and $T = 12$
   everywhere, with inconsistent sign. For rolled-out FM, where $T$ is part of training, 3×
   the training cost buys nothing measurable.
4. **Rolled-out flows take meaningless detours.** Classifying the intermediate state at each
   step, rolled-out accuracy *drops* mid-rollout — on ResNet-18/Aircraft to 9 pp *below* doing
   nothing at all — before recovering at the final step. Only $\hat z_T$ is supervised, so
   nothing constrains the path.
5. **One quantity predicts every cell.** Plotting the gain against the **headroom** Stage 1
   left behind — the trained linear probe minus the prototype baseline — orders all nine cells
   exactly (Spearman $\rho = 1.000$, $R^2 = 0.988$). The FM layer recovers about **70 % of
   whatever a trained classifier would have gained**, minus a fixed ~1.6 pp cost, while
   training no classifier at all.

**On DTD the FM layer is net harmful** at $K \in \{5, 10\}$ (−1.3 to −1.6 pp for standard FM,
worse for rolled-out) — and finding 5 says why. The fit breaks even at 2.2 pp of headroom, and
those two cells are the only ones below it: DTD's prototype baseline is already level with the
trained probe, so there is nothing for a transport to recover.

---

## Design decision: L2-normalize before the flow

Stage 1 prototypes are **unit-norm by construction**, but raw encoder features have norms of
**~24–50** (measured in Stage 1, step 2). Feeding raw features to the flow would make
$u_i = p_{y_i} - z_i$ almost entirely the $-z_i$ term, so the network would spend its
capacity learning "shrink toward the origin" rather than "move toward the right class".

**Both $z$ and $p$ are therefore L2-normalized before the flow**, putting them on the same
unit sphere. This is not a deviation from Stage 1: `cvlab.prototypes` already normalizes
before averaging and before the cosine rule, so the baseline is untouched and the comparison
stays exact. Cosine classification is scale-invariant, so this changes nothing about how the
final accuracy is measured.

**This is ablated, not assumed** (step 2, section 6b). Training the identical objective on raw
features costs **15.8–20.3 pp** at $K = \text{full}$, $T = 4$:

| Encoder / dataset | L2-normalized | raw | cost | raw vs. baseline |
| --- | --- | --- | --- | --- |
| ResNet-18 / DTD | 58.88 | 38.56 | **−20.3 pp** | −20.3 — *below* the baseline |
| ResNet-18 / Aircraft | 31.53 | 15.78 | **−15.8 pp** | −9.7 — *below* the baseline |
| DINOv2 / Aircraft | 56.56 | 37.92 | **−18.6 pp** | +3.7 — still above it |

On both ResNet-18 combinations the flow trained on raw features ends up **worse than doing
nothing at all** — it destroys accuracy rather than merely failing to add any. DINOv2 is the
exception: raw features still beat the baseline there, but give up 18.6 of the 22.3 pp that
normalization delivers.

The mechanism is visible in the loss. With unit-norm inputs the objective settles at
0.08–0.20; with raw inputs at **27–111**, because $u_i = p_{y_i} - z_i$ then has magnitude
$\approx \lVert z_i \rVert \approx 24\text{–}50$ and the network is asked to regress a vector
two orders of magnitude larger whose direction is 99.99 % determined by $-z_i$.

---

## Gradient clipping

Backpropagating through $T$ composed Euler steps can amplify gradients: without clipping the
rolled-out objective diverged in 1 of 54 runs, while standard FM never did in 27.
Gradient-norm clipping at **1.0** is applied to **both** objectives — not only the one that
needed it — so they remain configuration-identical and the comparison still isolates the
objective. It is inert for standard FM (every cell moved by ≤ 0.25 pp), which is the evidence
that it grants no advantage. The unclipped run is preserved in `results/` as evidence.

---

## Experimental grid

3 encoder/dataset combinations × 3 training sizes $K \in \{5, 10, \text{full}\}$ × 3 seeds.

| Method | Models trained per (combo, K, seed) | Evaluated at |
| --- | --- | --- |
| Stage 1 prototype baseline | 0 — reused from Stage 1 | — |
| Standard FM | 1 | $T = 4$ and $T = 12$ |
| Rolled-out FM | 2 (one per $T$) | its own $T$ |

**81 trained models**, filling a **45-cell** results table (3 combos × 3 K × 5 methods).

Reported as top-1 accuracy on the complete official test split, plus
$\Delta\mathrm{Acc} = \mathrm{Acc}_\mathrm{FM} - \mathrm{Acc}_\mathrm{baseline}$ against the
corresponding Stage 1 cell.

---

## Velocity network

A small MLP, deliberately not tuned: 2 hidden layers of width 512, SiLU activations, scalar
$t$ concatenated to the input feature, output dimension equal to the feature dimension. The
architecture and main training choices are held **fixed** across standard and rolled-out
training so the comparison isolates the objective.

---

## Folder layout

```text
Stage_2/
├── README.md                  this file
├── pyproject.toml             makes `cvlabfm` installable
├── src/cvlabfm/               FM layer, training loops, rollout — the reusable code
├── tests/                     the objective and rollout formulas, checked against the implementation
├── Work/
│   ├── 01_setup_prototypes/   load Stage 1 features, rebuild prototypes, verify vs Stage 1
│   ├── 02_standard_fm/        standard FM training
│   ├── 03_rolled_out_fm/      rolled-out FM training
│   ├── 04_evaluation/         accuracy table, ΔAcc, accuracy-vs-K, training curves
│   └── 05_visualizations/     feature-space comparison, flow trajectories
├── results/                   shared artefacts consumed across notebooks
│                               (the two *_models.pt weight files are NOT in git, ~26 MB
│                               combined, regenerable by re-running steps 2 and 3 — everything
│                               needed to check the results is tracked instead)
└── Reports/                   Stage2Report.pdf — the written report
```

Each `Work/` step follows the Stage 1 convention: `code/` (the notebook), `plots/`
(figures), `tables/` (the CSVs behind every claim), and a `README.md`.

---

## Reproducing these results

```powershell
C:\cvlab_env\Scripts\python.exe -m pip install -e Stage_1          # cvlab: encoders, features, prototypes, evaluation
C:\cvlab_env\Scripts\python.exe -m pip install -e Stage_2          # cvlabfm: the flow-matching layer
C:\cvlab_env\Scripts\python.exe -m pytest Stage_2/tests -q
```

Then run the five notebooks in `Work/` in order, selecting the `Python (CVLAB Stage 1)`
kernel. Steps 01, 04 and 05 take seconds; step 02 takes minutes; step 03 trains 54 models and
takes roughly 40 minutes on a single GPU.

Stage 1's cached features are required and are not tracked here (see `Stage_1/README`);
`torch` must come from the PyTorch CUDA index rather than PyPI, as noted in `pyproject.toml`.
Every run is seeded — re-training all 81 models reproduces every accuracy and every loss curve
bit-for-bit.

---

## Does the code compute the formulas it claims to?

Every result here rests on that being true, and it is unusually easy to break without
noticing: a loss averaged over the feature dimension, a rollout whose last step lands on
$t = 1$, or an interpolation normalized on one side only would each still train, still
converge, and still produce a plausible accuracy table.

`tests/test_flow.py` re-derives each of the four formulas from their stated form and asserts
the implementation agrees — deliberately written the slow, literal way, so the check is
independent of the code it checks.

```powershell
C:\cvlab_env\Scripts\python.exe -m pytest Stage_2/tests -q      # 21 passed
```

The suite itself was checked by breaking the implementation five ways on purpose. Three
breakages were caught by several tests each. The other two turned out not to be breakages at
all, which is worth recording:

- Removing the L2 normalization inside the classifier changes nothing — dividing a row by a
  positive scalar cannot reorder that row's similarity scores.
- Swapping cosine for **nearest prototype by Euclidean distance** also changes nothing, but
  only because the prototypes are unit-norm: expanding the square gives
  $\arg\min_c \lVert z - p_c \rVert^2 = \arg\max_c \, z \cdot p_c$ once every
  $\lVert p_c \rVert$ is equal.

---

## What this stage produced

| Output | Produced in |
| --- | --- |
| Accuracy over the full grid, $\Delta\mathrm{Acc}$ against the baseline, and accuracy vs. $K$ with error bars | step 4 |
| Training-loss curves for both objectives | step 4 |
| Feature space before and after transport — original, standard FM, rolled-out — one jointly fitted projection | step 5 |
| Flow trajectories: intermediate states, original feature, transported feature, prototype | step 5 |
| Accuracy at every intermediate Euler step of the rollout | step 5 |
| Distance from samples to prototypes at intermediate flow times | step 5 |
| The learned flow run in reverse, starting from the prototypes | step 5 |

Beyond those, three checks were added because the conclusions depend on them: the Stage 1
baseline is reproduced **exactly** before anything is built on it (step 1), the L2
normalization decision is **ablated** rather than assumed (step 2), and the reason rolled-out
training underperforms is established by **ruling out** underfitting and overfitting rather
than asserted (steps 3 and 5).

---

## Dependency on Stage 1

| Reused | Source |
| --- | --- |
| Cached frozen features (9 files) | `Stage_1/features/` |
| Prototype construction, L2 normalization | `cvlab.prototypes` |
| Balanced k-shot subsets (same seeds → same images) | `cvlab.data.make_kshot_subset` |
| Baseline accuracies to compare against | `Stage_1/results/prototype_runs.csv` |

Stage 1 is **not modified**. Step 1 re-derives the Stage 1 prototype numbers from scratch and
checks them against `prototype_runs.csv` before anything is built on top.
