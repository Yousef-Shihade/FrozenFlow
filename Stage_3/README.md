# Stage 3 — Flow Matching Before a Frozen Linear Classifier

**A Computer Vision Project — *Flow Matching as a Layer***
University of Haifa · 2026

Yousef Shihade · Mira Bitar

**Full report:** [Stage_3/Reports/Stage3Report.pdf](Reports/Stage3Report.pdf)

---

## Goal

Insert a **flow-matching (FM) layer** between the frozen image feature and the **linear
classifier trained in Stage 1**, then hold that classifier fixed:

$$z \;\xrightarrow{\ \text{FM},\ T \text{ Euler steps}\ }\; \hat z \;\xrightarrow{\ \text{frozen } W,\,b\ }\; s$$

The encoder is frozen, and now the classifier is frozen too. The **only** trainable thing in
the system is the flow. The question is whether a learned transformation of the features can
make them better handled by a linear classifier that has already been fitted and can no longer
adapt.

The flow is initialised so that $v(z,t) = 0$, which makes $\hat z = z$ exactly. **Before
training, the system *is* the Stage 1 linear probe** — so every number reported here is a
difference caused by training the flow, and by nothing else.

Three questions to answer:

1. Does a flow in front of a **frozen, already-trained** classifier help?
2. Does **classifier-guided** FM training beat **end-to-end rolled-out** training?
3. What does the learned transformation change about the feature space?

---

## How Stage 3 differs from Stage 2

Both stages put a flow in front of a classifier, but the classifier is not the same kind of
object, and that changes almost everything.

| | Stage 2 | Stage 3 |
| --- | --- | --- |
| Classifier | class prototypes, **untrained** | linear probe, **trained** (Stage 1) |
| Decision rule | cosine similarity — scale-**invariant** | affine $Wz+b$ — **not** scale-invariant |
| Features | **L2-normalized** | **raw** (norms 24–50) |
| FM target | a fixed prototype, known in advance | no fixed target — built from the classifier |
| Baseline to beat | the prototype classifier | the trained probe — a far higher bar |
| Result | up to **+22.3 pp** | **+0.1 to +2.6 pp** |

That last row is not a disappointment, it is the prediction Stage 2 made. Stage 2 found the FM
gain tracked *trained probe − untrained prototype* with $\rho = 1.000$: the flow recovers
roughly what training the classifier would have gained. **In Stage 3 the classifier is already
trained, so that headroom is spent before the flow starts.** What remains is whatever a
nonlinear transform can add on top of an already-fitted linear boundary — about a point.

---

## The two training strategies

Both train the same velocity network through the same $T=4$ Euler rollout. They differ in how
the frozen classifier is used to produce a learning signal.

### Strategy 1 — end-to-end rolled-out classification

Roll $z$ all the way to $\hat z$, classify it, and backpropagate the classification loss
**through every Euler step**:

$$\mathcal{L}_\mathrm{cls} = \mathrm{CE}(W\hat z + b,\; y)$$

Only the flow's parameters are updated. The classifier is the loss.

### Strategy 2 — classifier-guided targets, standard FM

Use the classifier to **build a target**, then train the flow to it with an ordinary FM update:

1. Roll $z$ through the current flow to get $\hat z$.
2. Compute the classification loss of the frozen classifier on $\hat z$.
3. Build an improved $\hat z'$ by descending that loss **in feature space**:
   $\hat z' = \hat z - \eta\,\nabla_{\hat z}\,\mathrm{CE}$ (unit-normalized, so $\eta$ is the
   distance actually travelled).
4. Source $z$, target $\hat z'$ — **detached**, so this is a genuine FM update and not a second
   end-to-end objective.
5. Standard FM: $t\sim\mathcal{U}(0,1)$, $z_t=(1-t)z+t\hat z'$, $u=\hat z'-z$,
   $\mathcal{L}_\mathrm{FM}=\mathrm{MSE}(v(z_t,t),u)$ — mean over the feature dimension, not
   on the same numeric scale as Stage 2's summed loss.
6. Recompute the targets every `refresh_every` epochs, as the flow changes.

The classifier says **where to go**; the standard FM machinery does the **going**. The gradient
never passes through the rollout.

---

## Results

Top-1 test accuracy, $K = 10$, $T = 4$, mean ± sample std over seeds $\{0,1,2\}$.
Configurations selected on **validation** accuracy, which appears in no reported figure.

| Combination | Linear probe | Strategy 1 | ΔAcc | Strategy 2 | ΔAcc |
| --- | --- | --- | --- | --- | --- |
| ResNet-18 / DTD | 52.04% | 52.15 ± 1.15 | +0.11 | **53.24 ± 1.39** | **+1.21** |
| DINOv2 / FGVC-Aircraft | 51.54% | **54.13 ± 1.13** | **+2.59** | 53.73 ± 1.53 | +2.19 |
| ResNet-18 / FGVC-Aircraft \* | 27.87% | 28.97 ± 0.74 | +1.10 | **29.92 ± 0.60** | **+2.05** |

\* beyond the two main combinations — included because Stage 1 measured its
features as the most entangled (own-vs-other cosine margin 0.041), making it the hardest test.

**Every combination improves under both objectives.** Across the full sweeps, **67/81 (83%)**
of Strategy 1 runs and **91/99 (92%)** of Strategy 2 runs beat the frozen probe.

### Do the two strategies actually differ?

Strategy 2's main configuration already used $\mathrm{lr}=10^{-4}$, carrying step 02's finding;
Strategy 1's used $10^{-3}$. Comparing those two would compare *tuning*, not *objective*. With
the learning rate held fixed:

| lr | Strategy 1 | Strategy 2 | difference | *p* |
| --- | --- | --- | --- | --- |
| $10^{-3}$ | +0.15 | **+0.94** | **+0.79** | **0.048** |
| $10^{-4}$ | +1.27 | +1.40 | +0.13 | 0.802 |
| $10^{-5}$ | **+1.21** | +0.89 | −0.31 | 0.451 |

- **On accuracy: indistinguishable** — +0.13 pp at $p = 0.80$. They trade wins per combination.
- **On robustness: Strategy 2 is clearly better**, and significantly so ($p = 0.048$).

The mechanism is visible in the curves. Strategy 1 optimises the classification loss
**directly**, so a large step memorises training labels within a handful of epochs (selected
epochs 4 / 66 / 35). Strategy 2 optimises an **MSE regression onto a target vector** — labels
reach it only through where that target was placed — so the same step size cannot memorise them
nearly as fast (selected epochs 101 / 117 / 198).

### What limits both

The frozen probe already classifies **its own k-shot training set** at 98.44–100.00% (CE as low as
0.005). Strategy 1's loss is computed on exactly that set, so the objective is nearly exhausted
before training starts, and following it further is memorisation. Both strategies depend
entirely on **validation-accuracy checkpointing** — Stage 1's own rule — to stop in time.

---

## Three findings worth singling out

**The default learning rate was the worst setting tested.** $10^{-3}$ is what Stages 1 and 2
both use, and carrying it over was the natural choice. Here it is actively harmful: dropping it
$10\times$ is worth **+1.1 pp**, more than any regularisation achieves. Constraining how *fast*
the flow moves beats constraining how *far*.

**Refreshing the guided targets less often is better.** Strategy 2 recomputes its targets
as the flow changes. Every epoch gives +1.40 pp; every 10 epochs **+1.82 pp**. Chasing
a target that moves with the flow chasing it is worse than letting it go slightly stale.

**Movement is not what produces the gain.** On DINOv2/Aircraft the two strategies reach nearly
the same accuracy by completely different routes — Strategy 1 displaces features by 34.94,
Strategy 2 by **4.60**, roughly $8\times$ less, for an equal-or-better result.

---

## What the flow actually changes

A 2-D projection cannot say whether class structure improved, so two quantities are measured in
the **full** feature space. They disagree, informatively.

| Combination | Strategy | Δ acc (6 cls) | logit margin | Δ separability |
| --- | --- | --- | --- | --- |
| ResNet-18 / DTD | 1 | −1.25 | **0.997×** | +0.0046 |
| ResNet-18 / DTD | 2 | −0.42 | **1.92×** | +0.0445 |
| DINOv2 / Aircraft | 1 | +1.01 | 1.58× | **−0.0739** |
| DINOv2 / Aircraft | 2 | +1.51 | 1.64× | +0.0059 |
| ResNet-18 / Aircraft | 1 | +2.01 | 1.45× | +0.0429 |
| ResNet-18 / Aircraft | 2 | +3.52 | 1.62× | +0.0103 |

**Logit margin tracks the result.** The frozen classifier is linear, so its decision is entirely
the gap between the top logit and the runner-up. Five of six cases widen it by 1.45–1.92×. The
one exception — Strategy 1 on DTD at **0.997×**, no widening at all — is exactly the cell where
Strategy 1 delivered nothing on the full test set (+0.11 pp). The diagnostic and the headline
agree.

**Class separability does not.** Own-minus-other cosine similarity, the training-free measure
that predicted the entire Stage 1 ordering, moves the *wrong way* where the gain is real:
Strategy 1 on DINOv2/Aircraft **loses** 0.074 of separability while *gaining* accuracy.

Stage 2 recorded the same lesson from the other direction — flows there contracted features
toward the right prototype without separating them from the wrong ones, and accuracy did not
follow. **Neither clustering in a projection nor class-mean geometry is what a linear classifier
consumes. The margin is.**

---

## Unfreezing the classifier

**Unfreezing makes no measurable difference: −0.02 pp, $p = 0.97$** (at matched objective and
learning rate — comparing against the *best* frozen result would confound unfreezing with which
objective was used).

The step also runs a **classifier-only control**, beyond unfreezing itself: continue
training the classifier from Stage 1's weights with no flow at all. Without it, "joint training
helped" cannot be distinguished from "the classifier simply got more training".

| Combination | classifier only | joint | **flow contributes** |
| --- | --- | --- | --- |
| ResNet-18 / DTD | +0.02 | +0.23 | **+0.21** |
| DINOv2 / Aircraft | +0.65 | +2.53 | **+1.88** |
| ResNet-18 / Aircraft | −0.01 | +0.97 | **+0.98** |
| **mean** | **+0.22** | **+1.24** | **+1.02** |

The control moves the classifier **+0.22 pp**, which independently confirms Stage 1 left the
probe at its ceiling rather than assuming it. The classifier learning rate spans $100\times$ and
the resulting weight drift spans $50\times$, with **no** corresponding change in accuracy.

Most telling: the classifier drifts **more** when trained alone (0.104) than jointly with the
flow (0.039). Once the flow is doing the work, there is nothing left for the classifier to do.

This is a useful negative result. Stage 3's entire premise is a frozen classifier; had
unfreezing delivered a large gain, the frozen results would look like an artificial handicap.
It does not, so they do not.

---

## Protocol

Fixed in step 01 and used unchanged throughout.

- **$T = 4$** Euler steps. Stage 2 measured $T=4$ against $T=12$ across its whole grid and
  found them within 1 pp with inconsistent sign, so we use the cheaper one — and it is
  meaningfully cheaper here, because Strategy 1 backpropagates through every step.
- **$K = 10$**, seeds $\{0,1,2\}$ — the same k-shot subsets as Stages 1 and 2, same images.
- **Raw features, no L2 normalization** (see below).
- **Velocity network**: Stage 2's design unchanged — 2 × 512 hidden, SiLU, $t$ concatenated —
  with the output layer zeroed for identity initialisation.
- **Validation-accuracy checkpointing**, which is Stage 1's own rule.
- **Gradient clipping at 1.0**, carried from Stage 2, which found backpropagation through $T$
  composed Euler steps can amplify gradients enough to diverge.

### Design decision: Stage 3 does *not* L2-normalize

This is the one thing Stage 3 inherits differently from Stage 2, so it is measured rather than
argued. Stage 2 normalized because its targets were prototypes, unit-norm by construction.
Stage 3 must not, because **its frozen classifier was trained on raw features**.

| Encoder / dataset | mean ‖z‖ | raw | L2-normalized | cost |
| --- | --- | --- | --- | --- |
| ResNet-18 / DTD | 23.7 | 52.04% | 47.55% | **−4.49 pp** |
| DINOv2 / FGVC-Aircraft | 49.7 | 51.54% | 46.19% | **−5.34 pp** |
| ResNet-18 / FGVC-Aircraft | 28.9 | 27.87% | 24.44% | **−3.43 pp** |

Not a collapse — normalizing preserves each feature's *direction*, where most class information
lives — but a free 3.4–5.3 pp loss everywhere.

The DINOv2 row is the sharpest form of the argument. Its norms are already tightly clustered
(46.9–52.0), so normalizing is close to dividing every feature by the same constant — and it
still costs the most. $W(z/\lVert z\rVert)+b$ rescales $Wz$ but leaves $b$ untouched, so the
decision boundary moves relative to the data. **Unlike Stage 2's cosine rule, this classifier is
not scale-invariant.**

---

## Pipeline

Each step's `README.md` holds its full reasoning, its tables and its figures.

### Step 01 — Setup and the frozen classifier ([details](Work/01_setup_classifier/README.md))

No flow is trained. Establishes the four things everything downstream stands on, and
**measures** each rather than assuming it.

Stage 1 saved accuracies, curves and predictions but **not the probe's weights** — the trained
layer went out of scope. Stage 3 needs the classifier itself, so it carries its own
`cvlab3.probe.train_frozen_classifier`, which reproduces Stage 1's loop and additionally returns
the weights. `ProbeConfig` is **imported** from Stage 1, so the hyperparameters keep exactly one
definition in the project.

| Check | Result |
| --- | --- |
| Re-derived probes vs. Stage 1's published accuracies | **max \|Δ\| = 0.00 pp** |
| Test predictions, per image | **identical** — 0 of 16,920 differ |
| Identity init: max \|$\hat z - z$\| | **0.0** — bit-for-bit |
| Accuracy at initialisation vs. the probe | **Δ = 0.0 pp** |

`FrozenClassifier` makes "frozen" mechanical rather than a promise: `requires_grad=False` at
construction, and `.train()` is overridden to refuse to leave eval mode, so a parent module
calling `.train()` cannot silently unfreeze it.

### Step 02 — Strategy 1 ([details](Work/02_end_to_end/README.md))

**81 runs** = 9 configurations × 3 combinations × 3 seeds, ~32 min. Learning-rate sweep plus
both regularisers we implemented (displacement and velocity magnitude, three strengths
each).

Contains the step's most interesting result: **remaining training loss anti-predicts the gain.**
The combination with the most unminimised loss (DTD, CE 0.327) gains the least (+0.32 pp); the
one whose objective is essentially at zero (DINOv2, CE 0.005) gains the most (+2.21 pp).
Whatever the flow exploits, it is not the residue of the classification objective. Displacement
correlates *positively* with the gain across all 81 runs ($\rho = +0.54$, $p < 0.001$).

### Step 03 — Strategy 2 ([details](Work/03_classifier_guided/README.md))

**99 runs** = 11 configurations × 3 combinations × 3 seeds, ~16 min. All four knobs Strategy 2
exposes: step size $\eta$, number of target steps, refresh frequency, and whether the update is
normalized.

On two of three combinations the targets start at **100% accuracy** and stay there — the probe
already classifies its own k-shot set perfectly, so there is no *correctness* for the guidance
to add. What the gradient still does is collapse target cross-entropy toward zero. Strategy 2 is
therefore a **margin-increasing transform**: it moves features so the classifier is more certain
about answers it already gives. That this still generalises is the interesting part.

Normalizing the guidance step matters more than any other choice (+0.16/+0.46 pp unnormalized
against +1.4 to +1.8 normalized), and the reason is measured: the raw gradient norm differs
$100\times$ across combinations, so one nominal step size means three very different effective
steps.

### Step 04 — Evaluation ([details](Work/04_evaluation/README.md))

The consolidated comparison. **Nothing is retrained here** — every number comes from the
run tables steps 02 and 03 wrote, so the reported comparison cannot quietly differ from the
experiments.

Configurations are chosen by **validation** accuracy. Validation and test independently pick the
same configuration for both strategies, so the headline is not a product of test-set selection.

### Step 05 — Feature-space visualisation ([details](Work/05_visualizations/README.md))

Three panels per combination — original, Strategy 1, Strategy 2 — on the **same test images**
with the **same class colours**, and one PCA fitted on the **concatenation** of all three sets,
so the panels are directly comparable rather than three independent rotations. Classes are
chosen by an even stride through the sorted class list, not hand-picked.

PCA rather than t-SNE: deterministic, with no perplexity or seed to justify, and "compute the
embedding jointly" has an unambiguous meaning for a linear method. Two components capture only
20.7–33.8% of the variance, which is why the geometry is *also* measured in the full feature
space rather than read off the pictures.

### Step 06 — Joint fine-tuning ([details](Work/06_joint_finetuning/README.md))

**45 runs** = 5 configurations × 3 combinations × 3 seeds, ~13 min. A further experiment —
unfreezing the classifier — plus the classifier-only control that makes it interpretable.

Deliberately last: unfreezing breaks the "only the flow changed" guarantee, so everything that
depends on that guarantee is finished and recorded before it is broken.

---

## Folder layout

```text
Stage_3/
├── README.md                      this file
├── pyproject.toml                 the cvlab3 package
├── Reports/
│   └── Stage3Report.pdf           the written report for this stage
├── src/cvlab3/
│   ├── classifier.py              FrozenClassifier, identity_flow, protocol constants
│   ├── probe.py                   Stage 1's probe loop + the weights it did not return
│   ├── training.py                Strategy 1 — end-to-end rolled-out
│   ├── guided.py                  Strategy 2 — classifier-guided targets
│   ├── joint.py                   unfreezing the classifier, and its control
│   └── paths.py                   where each step reads and writes
├── tests/
│   ├── test_probe_equivalence.py  11 tests — the duplicated probe loop matches Stage 1
│   └── test_stage3_formulas.py    20 tests — the stage's own formulas vs. the implementation
├── Work/
│   ├── 01_setup_classifier/       recover and freeze the probe, verify identity init
│   ├── 02_end_to_end/             Strategy 1 + regularisation sweep
│   ├── 03_classifier_guided/      Strategy 2 + the four-knob sweep
│   ├── 04_evaluation/             the consolidated comparison — retrains nothing
│   ├── 05_visualizations/         joint PCA, geometry in the full feature space
│   └── 06_joint_finetuning/       unfreezing the classifier + classifier-only control
└── results/                       artefacts shared across notebooks
```

Each `Work/` step follows the Stage 1 and 2 convention: `code/` (the notebook, with outputs),
`plots/` (figures), `tables/` (the CSVs behind every claim), a `README.md`, and an
`_artifact_manifest.csv` listing exactly what that step wrote.

---

## Reproducing these results

```powershell
C:\cvlab_env\Scripts\python.exe -m pip install -e Stage_1          # cvlab:   encoders, features, k-shot subsets, evaluation
C:\cvlab_env\Scripts\python.exe -m pip install -e Stage_2          # cvlabfm: the velocity network and Euler integrator
C:\cvlab_env\Scripts\python.exe -m pip install -e Stage_3          # cvlab3:  the frozen classifier and both strategies
C:\cvlab_env\Scripts\python.exe -m pytest Stage_2/tests Stage_3/tests -q      # 52 passed
```

Then run the six notebooks in `Work/` in order, selecting the `Python (CVLAB Stage 1)` kernel.
Steps 01, 04 and 05 take seconds to a couple of minutes; steps 02, 03 and 06 train 81, 99 and 45
models respectively.

Stage 1's cached features are required and are not tracked (see `Stage_1/README.md`). `torch`
must come from the PyTorch CUDA index rather than PyPI, as noted in `pyproject.toml`.

`results/visualised_flow_weights.pt` (~18 MB) is not tracked either — regenerate it by running
notebook 05. Everything needed to *check* those models is tracked instead: their accuracies,
their curves and the figures they produced.

---

## Does the code compute its own formulas?

Every result rests on that being true, and it is easy to break without noticing: a rollout whose
last step lands on $t = 1$, an FM interpolation with source and target swapped, or a velocity
recovery that drops the factor of $T$ would each still train, still converge, and still produce
a plausible accuracy table.

```powershell
C:\cvlab_env\Scripts\python.exe -m pytest Stage_3/tests -q      # 31 passed
```

`test_stage3_formulas.py` re-derives each formula from its stated form and asserts the
implementation agrees. `test_probe_equivalence.py` runs Stage 1's probe loop and Stage 3's side
by side and asserts they agree on accuracy, predictions, selected epoch, and **every** training
curve — so the duplicated loop cannot drift from Stage 1 silently.

The suite was itself validated by breaking the implementation **eight** ways on purpose. The
first run caught only five. The three misses were a real flaw: those tests re-implemented the
formula *inside the test file* rather than calling the code, so they could not fail. After
extracting `fm_pair()` and making `rollout_with_velocities()` public, **all eight are caught**.

---

## What this stage produced

| Output | Produced in |
| --- | --- |
| Top-1 test accuracy for the probe and both strategies, on both datasets, with ΔAcc | step 4 |
| Training and validation curves for both strategies | steps 2, 3, 4 |
| Feature space before and after — original, Strategy 1, Strategy 2 — one jointly fitted PCA | step 5 |
| Strategy 1 vs. Strategy 2 at matched learning rate, with significance | steps 3, 4 |
| Regularisation sweep: displacement and velocity penalties, three strengths each | step 2 |
| Guided-target sweep: step size, number of steps, refresh frequency, normalization | step 3 |
| Logit margin and class separability, measured in the full feature space | step 5 |
| Joint fine-tuning against frozen and against a classifier-only control | step 6 |

Beyond those, four checks were added because the conclusions depend on them: the Stage 1
classifier is reproduced **exactly** before anything is built on it (step 1), the
no-normalization decision is **ablated** rather than assumed (step 1), the strategy comparison
is made at **matched learning rate** rather than between differently-tuned defaults (steps 3
and 4), and the joint extension is interpreted against a **control** that separates the flow's
contribution from further classifier training (step 6).

---

## Dependency on Stages 1 and 2

| Reused | Source |
| --- | --- |
| Cached frozen features | `Stage_1/features/` |
| Balanced k-shot subsets (same seeds → same images) | `cvlab.data.make_kshot_subset` |
| Probe hyperparameters and evaluation | `cvlab.probe.ProbeConfig`, `cvlab.evaluation` |
| Baseline accuracies to compare against | `Stage_1/results/linear_probe_runs.csv` |
| Velocity network and Euler integrator | `cvlabfm.flow` |
| Plot styling and artefact manifests | `cvlab.plotting` |

**Stages 1 and 2 are not modified.** Stage 3 needed two things neither stage exposed — the
probe's trained weights, and a zero-initialised readout — and both were added **inside
`cvlab3`** rather than by editing published code. Step 01 then re-derives the Stage 1 numbers
from scratch and checks them before anything is built on top.
