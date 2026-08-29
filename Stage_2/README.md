# Stage 2 — Flow Matching as a Layer

**A Computer Vision Project — *Flow Matching as a Layer*** · Stage 2
University of Haifa

---

## Goal

Add a Flow Matching (FM) layer on top of Stage 1's prototype-based classifier.
Given a frozen image feature, a small learned network transports it toward its
class's fixed prototype (from Stage 1, unchanged) before classification. Two
training regimes are compared:

- **Standard FM** — supervise the velocity directly at random points along the
  straight-line path from feature to prototype.
- **Rolled-out FM** — simulate the actual multi-step inference procedure
  during training, and optimize only where the feature ends up.

Everything reuses Stage 1's exact cached features, prototypes, k-shot subsets,
and seeds — required for the comparison to be fair. Full spec: `stage_2.pdf`.

---

## Structure

```text
Stage_2/
├── stage_2.pdf                  the assignment
├── Work/
│   ├── 01_velocity_network/     the VelocityNetwork + FM math, validated on synthetic data
│   ├── 02_standard_fm/          the 27-run standard-FM training sweep
│   ├── 03_classification_eval/  FM vs. Stage 1 baseline, both training regimes, accuracy-vs-K
│   ├── 04_feature_space_viz/    before/after feature-space viz + flow trajectories + reverse flow
│   ├── 05_rolled_out_fm/        the 54-run rolled-out-FM training sweep
│   └── 06_training_curves/      per-epoch loss curves, both training regimes
└── results/                     everything the scripts above produce (see below)
```

## Quick start

Environment: same `C:\cvlab_env` Stage 1 uses (see the repo root README's Setup
section) — nothing additional to install. Requires Stage 1's cached features
in `Stage_1/features/` (see Stage 1's own README for how those are produced).

Run in this order:

```text
C:\cvlab_env\Scripts\python.exe Stage_2/Work/01_velocity_network/test_velocity_network.py
C:\cvlab_env\Scripts\python.exe Stage_2/Work/02_standard_fm/train_standard_fm.py
C:\cvlab_env\Scripts\python.exe Stage_2/Work/05_rolled_out_fm/train_rolled_out_fm.py
C:\cvlab_env\Scripts\python.exe Stage_2/Work/06_training_curves/training_curves.py
C:\cvlab_env\Scripts\python.exe Stage_2/Work/03_classification_eval/classification_eval.py
C:\cvlab_env\Scripts\python.exe Stage_2/Work/03_classification_eval/accuracy_vs_k.py
C:\cvlab_env\Scripts\python.exe Stage_2/Work/04_feature_space_viz/feature_space_viz.py
C:\cvlab_env\Scripts\python.exe Stage_2/Work/04_feature_space_viz/flow_trajectories.py
```

**Runtime, CPU (no NVIDIA GPU used or needed for this stage):** standard-FM
sweep ~20 min, rolled-out sweep ~53 min, everything else under a few minutes.

**Note on `results/`:** the 81 trained checkpoints (`flow_matching_models/`,
`rolled_out_fm_models/`, ~230 MB combined) are *not* committed — regenerable
from the two training scripts above, same convention as Stage 1's excluded
`Data/`/`features/`. Their `sweep_summary.json` files (the actual numeric
results) are tracked.

---

## Headline results

Top-1 test accuracy, ΔAcc vs. the Stage 1 prototype baseline (pp, mean over 3 seeds):

| Combo | K | standard T4 | standard T12 | rolled-out T4 | rolled-out T12 |
|---|---|---|---|---|---|
| DINOv2/Aircraft | 5 | +6.83 | +12.60 | −5.39 | −4.87 |
| | 10 | +12.18 | +17.84 | +2.65 | +0.64 |
| | full | +17.19 | **+22.24** | +5.98 | +3.61 |
| ResNet-18/Aircraft | 5 | −0.19 | +1.40 | −6.09 | −5.96 |
| | 10 | +0.96 | +3.04 | −3.25 | −4.83 |
| | full | +3.39 | +5.64 | −5.54 | −2.77 |
| ResNet-18/DTD | 5 | −16.05 | −8.81 | −24.02 | −26.05 |
| | 10 | −16.03 | −5.05 | −27.39 | −26.61 |
| | full | −12.32 | −1.68 | −28.49 | **−30.32** |

---

## Key findings

**1. Standard FM's benefit tracks Stage 1's separability margins, not just
baseline weakness.** DINOv2/Aircraft (Stage 1 margin 0.420) gains the most
(+22.2pp); ResNet-18/Aircraft (margin 0.041, the *worst*) gains little
(+5.6pp); ResNet-18/DTD, where the baseline was already near Stage 1's
linear-probe ceiling, is actively hurt (−1.7pp). A small learned network can
sharpen separation that's already present in the frozen features — it can't
manufacture separation that isn't there.

**2. Rolled-out FM underperforms standard FM in all 18 of 18 tested
configurations** — a fully consistent result across the entire grid. It is
*not* uniformly worse than doing nothing (DINOv2/Aircraft at K=10/full still
beats the baseline, +0.6 to +6.0pp), just far less than standard FM achieves
at the same settings.

**3. The mechanism, confirmed by direct measurement, not inferred:**
rolled-out's objective (`‖ẑ_T − prototype‖²`, checked only at the final step)
has a cheap shortcut standard FM's per-step supervision doesn't — pulling
every point toward a shared central region lowers average distance without
routing each point to its own correct target. Measured mean pairwise cosine
similarity between *different* classes' transported features is 2-4x higher
for rolled-out than standard FM in every checkpoint checked, and its
same-class-minus-different-class separation gap is smaller every time,
directly tracking the accuracy gap. Visually confirmed too: the feature-space
plot's rolled-out panel shows the 9 example classes contracted into one
dense, largely unseparated region (mean feature norm 49.8 raw → 14.9 after
standard FM → 8.5 after rolled-out).

**4. More Euler steps (T=12 vs. T=4) helps standard FM cleanly and
consistently, by margins well outside seed variance, everywhere** — a
textbook numerical-integration result (less discretization error). The same
is *not* true for rolled-out: T=4 vs. T=12 is close to a coin flip (5 of 9
combos favor T=4, 4 favor T=12), and most gaps are within seed noise. This
makes sense once you recall rolled-out bakes T into training itself — T=4 and
T=12 rolled-out models are two independently-trained solutions, not the same
model integrated more carefully.

**5. Forward flow trajectories show FM does not "arrive" at the target
prototype.** Cosine similarity to the *correct* prototype actually falls
during transport (~0.8 → ~0.4 across several real examples), yet
classification against the full class set stays correct. This is expected
for flow matching, not a bug: the network learns a population-level velocity
field shared across many training pairs, not a per-point direct route.
Classification only needs relative similarity to improve, not absolute
similarity to the true target — the transport can move a point's direction
away from every prototype while moving away from the wrong ones faster.

**6. Reverse flow (starting from a prototype, integrating with `-v_theta`) is
close to a no-op.** A prototype already sits at its class's real-feature
centroid by construction (cosine ~0.98-0.99), and the learned field has
little residual signal right where training already converged it to
near-zero — so reversing it barely moves the point (cosine drifts to
~0.94-0.96, not toward anything more representative).

---

## Implementation notes worth knowing upfront

- **Runs entirely on CPU** — this machine has no NVIDIA GPU. Confirmed via
  direct timing before each sweep that this doesn't matter for FM's small
  MLP (unlike Stage 1's feature extraction, which does need one).
- **`Stage_1/src/cvlab/paths.py` was modified** for this stage: Stage-1-root
  detection previously required both `Data/` and `Work/` to exist, which
  incorrectly rejected a valid cached-features-only checkout. Relaxed to
  only require `Work/` + `src/cvlab/`. Backward-compatible.
- **The standard-FM training sweep was rebuilt once.** An early version ran
  as a one-off terminal command and was never saved as a file — no
  reviewable code existed for how those checkpoints were produced. It was
  rebuilt as the current `train_standard_fm.py` and verified to reproduce
  the full-data (K=full) results bit-for-bit; the 5-shot/10-shot checkpoints
  from the original couldn't be exactly reconstructed (the original's exact
  subset-sampling RNG sequence was lost) but fall within normal seed-to-seed
  variance. All results in this README are from the rebuilt, reviewable
  version.

---

## Further detail

Each `Work/*/` script's own docstring documents its exact method and
conventions in more depth than this file.
