# Notes for Mira — Stage 2 setup & decisions log

Working solo on Stage 2 while you're prepping for your exam. This file tracks every
change to shared code, every bug caught, and every open question — so you're not
walking in cold when you review this, and so nothing gets explained from memory
under time pressure later. Updated as we go, not written after the fact.

---

## Environment

New machine, nothing pre-existing — `C:\cvlab_env` was built from scratch following
the root README's Setup section exactly (same PyTorch 2.6.0+cu124 / torchvision
0.21.0 versions, same editable install, same kernel name).

**No NVIDIA GPU on this machine** (Intel integrated graphics only — confirmed via
`nvidia-smi` not being found at all, not a driver issue). Everything for Stage 2 runs
on CPU. This is expected to be fine: the velocity network is a small MLP operating on
already-cached feature vectors, not raw images — much closer in cost to Stage 1's
linear probe (explicitly CPU-friendly per our own README) than to feature extraction.
Confirmed via direct timing before committing to any full sweep, not assumed.

Raw datasets (`Data/`) were never transferred to this machine — only the 9 cached
feature files from `Stage_1/features/`, verified against Step 02's README numbers
(25,640 total images, correct shapes/dtypes for both encoders) before any Stage 2
work began.

---

## Change to shared code: `Stage_1/src/cvlab/paths.py`

**What changed:** Stage-1-root detection previously required *both* `Data/` and
`Work/` to exist before recognizing a valid checkout. Relaxed to only require `Work/`
+ `src/cvlab/`.

**Why:** a cached-features-only checkout (this machine's exact situation — no raw
datasets, only `features/`) was being incorrectly rejected as "not a valid Stage 1
root," even though nothing that only reads cached features actually needs `Data/` to
exist.

**Why it's safe:** backward-compatible — any machine that still has `Data/` present
(yours) also has `Work/` and `src/cvlab/`, so this doesn't change behavior there, only
relaxes an overly strict check. Dataset-path resolution for code that genuinely needs
raw images is untouched.

Suggested commit message: `relax Stage-1-root detection to not require raw Data/,
since cached-feature-only checkouts are valid`

---

## New module: `Stage_1/src/cvlab/flow_matching.py`

Implements, per `Stage_2/stage_2.pdf`:
- `VelocityNetwork` — MLP, 2 hidden layers width 512, SiLU activations, feature+scalar-t
  input, feature-dim output
- Standard FM training loss (interpolate toward prototype, MSE against target velocity)
- Euler-step inference (T configurable)

**Validated on synthetic data before touching real features** (3 fake well-separated
classes, dim 16): mean distance to correct prototype went from 5.90 before transport
to 0.33 after, 100% of points landed nearest their correct prototype. Confirms the
mechanism itself is implemented correctly — this is a code-correctness check on an
easy toy case, not a preview of real-data performance.

---

## Bug caught during the 27-run standard FM sweep (caught before results were reported)

First pass: `torch.manual_seed(seed)` wasn't being applied before model
initialization specifically for the K=full runs — meaning those 9 runs weren't
actually reproducible yet, which breaks Stage 1's own seeding convention. Caught
before results were shared, not after. Full sweep was re-run from scratch rather than
patching only the affected runs.

---

## Standard FM sweep — 27/27 runs complete

All runs: `load_features()` → `compute_prototypes()` (Stage 1's exact k-shot subsets
and seeds, not recomputed) → train `VelocityNetwork` with standard FM loss. 200
epochs, AdamW, lr 1e-3. Total wall-clock: 19.6 minutes on CPU.

All 27 final losses checked for stability: no NaN, no outlier seeds, tight agreement
(<15% spread) across the 3 seeds within every (encoder, dataset, K) group.

**Open question, currently being checked:** DINOv2/Aircraft's loss values run
3-4x higher than ResNet-18's across every K setting. Leading hypothesis: this tracks
Stage 1's own finding that DINOv2's raw cached features have roughly double the L2
norm of ResNet-18's — if training features are used raw (unnormalized) while
prototypes are unit-norm, the velocity target's scale would differ by encoder for
reasons unrelated to task difficulty. Verifying directly (checking actual norms of z_i
vs. prototype in the interpolation) before deciding whether this needs a fix or is
just an explainable artifact of raw-scale loss values.

---

## New script: `Stage_2/Work/02_standard_fm/train_standard_fm.py`

The 27-run standard FM sweep, previously run from a scratch script that was never
saved, is now a reviewable file. Same recipe: `load_features()` → k-shot subset via
`cvlab.data.make_kshot_subset` (Stage 1's function, same seeds — not reimplemented) →
`compute_prototypes()` → train `VelocityNetwork` with `flow_matching_loss`, full-batch,
200 epochs, AdamW lr 1e-3, CPU. Writes the 27 checkpoints + `sweep_summary.json` in the
same on-disk format as before. `--dry-run` prints the grid.

Seed convention is unchanged and now explicit in code: `torch.manual_seed(seed)`
immediately before model construction in **every** run (the fix for the bug above),
with the k-shot draw (NumPy) additionally keyed by `seed` for K∈{5,10}.

**Ran it — 27/27 complete, 3.9 min on CPU** (faster than the archived 19.6 min; the
old run just hit more CPU contention — the per-run `elapsed_s` was ~2-5x higher across
the board). No NaN, seed spread 1.2-7.2% per group. Compared against the pre-regen
backup:

| | archived vs regenerated |
| --- | --- |
| **K=full (9 runs)** | **bit-identical** — same weights, same loss to 6 dp. Confirms the core recipe (arch, loss, optimiser, full-batch, 200 steps, seed→init order) is exactly what produced the originals. |
| **K=5/10 (18 runs)** | Losses land within each group's seed spread, but **weights differ** (max \|Δw\| ~0.3). The archived K=5/10 runs followed a different RNG path during training that the lost scratch script's details can't reconstruct — tried numpy-subset + N torch pre-draws and a torch-`randperm` subset, none reproduce the archived weights or loss. |

Net: the regenerated sweep is the one to carry forward — it's the one with a script
behind it, it reproduces K=full exactly, and its K=5/10 checkpoints are built the
sanctioned way (`cvlab.data.make_kshot_subset`, same seeds) and are internally
consistent with the eval below, which rebuilds prototypes from that same function.
The pre-regen `.pt` + json backup is in the session scratchpad if we ever need it.

---

## OPEN QUESTION FROM CLAUDE.md — RESOLVED

**Does the classification code L2-normalize `z_hat_T` before cosine similarity, matching
Stage 1's `classify_by_prototype`?** As of now: **yes, by construction.**
`cvlab.flow_matching.euler_inference` returns the transported feature raw (no
normalize, no classify — that part was simply never written). The new
`Stage_2/Work/03_classification_eval/classification_eval.py` adds it:

```python
def classify_by_cosine(features, prototypes):     # Stage 1's run_prototype_classifier, verbatim
    features_normed = l2_normalize(features)      # <-- the normalize in question
    similarities   = features_normed @ prototypes.T
    return similarities.argmax(dim=1)
```

`l2_normalize` is imported from `cvlab.prototypes` (not re-implemented), and the FM
path and the baseline are scored by this exact same function, so `Delta_Acc` is
apples-to-apples. A guard raises if `prototypes` isn't unit-norm.

One nuance worth knowing: because `compute_prototypes` already returns unit-norm rows,
normalizing the *query* doesn't actually change the arg-max on its own (scaling a row
by a positive scalar can't change which prototype it's most aligned with). Where the
convention really bites is metric choice: a Euclidean `torch.cdist` nearest-prototype
(like the one in `01_velocity_network/test_velocity_network.py`) is fine on unit-norm
toy data but wrong on real features — raw norms are 14-50 while every prototype is
norm 1, and on real data the FM-transported features come out at norm ~16 (DINOv2)
rather than ~1, so they never actually arrive at the prototype. `classify_by_cosine`
sidesteps all of that. The toy test asserts it matches `run_prototype_classifier`
exactly and is scale-invariant.

---

## Change to shared code: `Stage_1/src/cvlab/flow_matching.py` — added `euler_inference_trajectory`

**What changed:** new function `euler_inference_trajectory(model, starting_features, steps)`
alongside the existing `euler_inference`. Same Euler stepping, but it records each
state instead of overwriting, returning a `(steps + 1, N, D)` tensor:
`trajectory[0]` is the input unchanged, `trajectory[-1]` is exactly what
`euler_inference` returns, and the rows between are the intermediate steps — the
raw material for the flow-trajectory visualization deliverable.

**Why it's safe:** purely additive. `euler_inference` is byte-for-byte untouched;
`__all__` gains one name. Nothing that exists today changes behaviour.

**Verified** in `Stage_2/Work/01_velocity_network/test_velocity_network.py` (extended,
still passes): for steps ∈ {4, 100} the trajectory has exactly steps+1 states,
`trajectory[0]` equals the input (`torch.equal`), and `trajectory[-1]` equals
`euler_inference(model, x, steps)` (`torch.equal`).

Suggested commit message: `add euler_inference_trajectory: Euler transport that keeps
every intermediate state, for flow-path visualization`

---

## New scripts: `Stage_2/Work/03_classification_eval/`

- `classification_eval.py` — `evaluate(model, test_x, test_y, prototypes, steps)`
  returns `acc_fm`, `acc_baseline` (Stage 1 prototype baseline on the same raw test
  features), `delta_acc`, and before/after feature norms. `main()` sweeps all 27
  checkpoints at T∈{4,12}, rebuilding prototypes from each checkpoint's
  `(encoder, dataset, K, seed)` via `make_kshot_subset` + `compute_prototypes`.
  Writes `Stage_2/results/classification_eval/eval_runs.csv` + `eval_summary.json`
  (per-run rows + seed-aggregated Delta_Acc).
- `test_classification_eval.py` — synthetic toy test, **run first and passes**:
  `classify_by_cosine` == `l2_normalize→dot→argmax`, == `run_prototype_classifier`
  predictions exactly, scale-invariant, rejects non-unit prototypes; FM transport on
  the toy gives acc_fm 100% vs baseline 98.4% (Delta_Acc +1.56 pp) and pulls feature
  norm from ~4.2 to ~1.0.

**Ran the real eval — 54 runs, 10.6 s.** `acc_baseline` reproduces Step 04's prototype
numbers (DINOv2/Aircraft full 34.26, ResNet-18/DTD full 58.83, ResNet-18/Aircraft full
25.47 — all exact), which is the correctness check on `classify_by_cosine`. Headline
Delta_Acc (mean ± std over seeds, percentage points):

| encoder / dataset | K | T=4 | T=12 |
| --- | --- | --- | --- |
| DINOv2 / Aircraft | 5 | +6.83 ± 1.01 | +12.60 ± 0.11 |
| DINOv2 / Aircraft | 10 | +12.18 ± 1.33 | +17.84 ± 0.64 |
| DINOv2 / Aircraft | full | +17.19 ± 2.17 | +22.24 ± 0.14 |
| ResNet-18 / Aircraft | 5 | −0.19 ± 0.62 | +1.40 ± 0.20 |
| ResNet-18 / Aircraft | 10 | +0.96 ± 0.71 | +3.04 ± 1.00 |
| ResNet-18 / Aircraft | full | +3.39 ± 0.89 | +5.64 ± 0.45 |
| ResNet-18 / DTD | 5 | −16.05 ± 2.26 | −8.81 ± 1.76 |
| ResNet-18 / DTD | 10 | −16.03 ± 2.40 | −5.05 ± 1.71 |
| ResNet-18 / DTD | full | −12.32 ± 1.91 | −1.68 ± 0.67 |

Reading (for us to discuss, not a conclusion yet): standard FM helps a lot on
DINOv2/Aircraft, is mildly positive on ResNet-18/Aircraft, and *hurts* on
ResNet-18/DTD — least badly at full K and T=12, where it nearly returns to baseline.
T=12 beats T=4 everywhere.

---

## New script: `Stage_2/Work/04_feature_space_viz/feature_space_viz.py`

Before/after feature-space visualization, built to line up with Stage 1's own
(`Stage_1/Work/05_analysis` section 6 — the ResNet-18 vs DINOv2 Aircraft figure):

- **Same class subset**: `pick_diverse_classes` is Stage 1's helper verbatim
  (`np.linspace` stride through the sorted class list). Aircraft → the same 9:
  707-320, 747-300, A319, BAE 146-200, Cessna 525, DHC-8-300, Falcon 2000, MD-80,
  Yak-42. **Same test images**: every test-split image of those 9 classes, no
  subsampling (299 for DINOv2/Aircraft).
- **Same style**: `cvlab.plotting.apply_style` + `ArtifactWriter`, tab10
  colour-per-class, prototypes as black-edged stars — `plot_projection` is Stage 1's
  function unchanged.
- **Same normalize-before-projecting lesson** (Stage 1 section 6a): raw features,
  FM-transported features, and prototypes are all `l2_normalize`d before the fit.
- **New**: one PCA fit *jointly* across original + transported + prototypes, so both
  panels share one projection and one axis scale (`set_xlim/ylim` + equal aspect) —
  directly comparable. Panel 2 = same features after `euler_inference` through one
  standard-FM checkpoint (default DINOv2/Aircraft/full/seed0, T=12).

**Ran it.** `Stage_2/results/feature_space_viz/feature_space_viz.png` +
`feature_space_viz_classes.csv`. Transport pulls the DINOv2 feature norm 49.8 → 14.9;
2D PCA explained variance 15.0% (low — read qualitatively, as Stage 1 says). Left
panel: diffuse, overlapping clouds, prototypes at the rim. Right panel: transported
points contract inward toward the prototype region. Stars sit in identical positions
on both panels (prototypes are the fixed targets, not transported) — a good visual
check that the shared projection is really shared.

**Third panel** ("after rolled-out FM") — the `views` dict in `main()` has a marked
spot; add one line transporting `raw_sel` through a rolled-out checkpoint and the
joint PCA / shared axes / legend / layout all adapt to `len(views)`.

Only reads shared cvlab code, no changes to it.

---

## New script: `Stage_2/Work/04_feature_space_viz/flow_trajectories.py`

The professor's "visualize a path from a sample" and "opposite direction, -v"
asks. Small multiples (a single 2D projection can't carry several paths at this
explained-variance level); every panel uses `feature_space_viz.py`'s recipe --
L2-normalize onto the unit sphere, one PCA fit jointly across that panel's points
(its trajectory + the 9 prototypes + the class cloud). T=12, same checkpoint
default (DINOv2/Aircraft/full/seed0).

- **`flow_trajectories_forward.png`** -- 4 test images (one per class), transported
  with `euler_inference_trajectory`. Dot at every Euler step; quarter points
  labelled; **t=0.5** as an open diamond + bold callout; original feature (circle,
  t=0) and transported feature (square, t=1) black-edged; own prototype a big
  coloured star, siblings small grey stars; dotted connector t=1 -> own prototype.
- **`flow_trajectories_reverse.png`** -- start at a class prototype (t=1), integrate
  `w <- w - dt * v(w, t)` for 12 steps to t=0, over that class's real test
  features. `reverse_trajectory_from_prototype` is local to this script (time runs
  1 -> 0; `euler_inference_trajectory` is the forward-only library function).

**What the figures actually show (worth discussing with the professor):**

- *Forward:* the path leaves the class cloud but **does not go to the prototype**.
  cos(state, own prototype) *falls* along the path -- Cessna 525 0.83->0.46,
  DHC-8-300 0.85->0.40, Falcon 2000 0.78->0.36, 707-320 0.73->0.38 -- yet all
  four are still argmax-correct over 100 classes. Standard FM at T=12 is improving
  classification by shrinking the *sibling* prototypes' similarity faster than the
  true one's (relative separation), not by transporting features onto their
  prototype. The square sits nowhere near the star, and that gap is real, not a
  plotting artefact. (Consistent with feature_space_viz: transported norm 49.8 ->
  14.9, i.e. Euler undershoots the prototype badly.)
- *Reverse:* the prototype already sits at the centre of the real-data cloud
  (cos to the mean real test feature 0.98 / 0.99). Integrating -v_theta moves it a
  small amount and cos *dips* slightly (0.98->0.94, 0.99->0.96) -- it stays inside
  the data region but doesn't "trace back" to anything better than where it
  started. On this checkpoint the reverse flow is close to a no-op.

Reads shared cvlab code only (uses the new `euler_inference_trajectory`); no
changes to it.

---

## Change to shared code: `Stage_1/src/cvlab/flow_matching.py` — added `rolled_out_loss`

**What changed:** new function `rolled_out_loss(model, source_features, prototypes,
labels, steps)` alongside `flow_matching_loss`, implementing the spec's `L_roll`:
run the full `steps`-step Euler transport from each source feature, then
`mean((z_hat_T - prototype)**2)`, with the autograd graph kept through the whole
chain. No `@torch.no_grad()` (it must build the graph); no internal randomness, so
a rolled-out run is fully deterministic given its seed.

**Reduction choice:** `torch.mean` over batch *and* feature dim, same as
`flow_matching_loss` — the spec writes `L_roll = ||z_hat_T - p||^2_2` per sample,
but matching `flow_matching_loss`'s reduction keeps the two objectives on the same
scale for a fair comparison, which the spec explicitly asks for ("keep the main
training choices fixed").

**Why it's safe:** purely additive; `flow_matching_loss` / `euler_inference` /
`euler_inference_trajectory` untouched; `__all__` gains one name.

Suggested commit message: `add rolled_out_loss: backprop-through-Euler-chain FM
objective (Stage 2 spec L_roll)`

---

## Timing check before the rolled-out sweep (same discipline as the standard-FM check)

Measured per-epoch (40 epochs after 5 warmup), extrapolated to 200-epoch runs, CPU / 8 threads:

| config | n_train | dim | T=4 / run | T=12 / run |
| --- | --- | --- | --- | --- |
| dinov2 / Aircraft / full *(the config to time, per the ask)* | 3334 | 384 | ~41 s | ~149 s |
| **resnet18 / Aircraft / full** *(actually the worst — dim 512)* | 3334 | 512 | ~74 s | **~218 s** (saw 405 s under sustained load) |
| resnet18 / DTD / K5 *(cheapest)* | 235 | 512 | ~8 s | ~24 s |

Standard FM at full-data is ~16 s/run, so rolled-out T=12 is ~9-14x per run, T=4 ~2.5-4.5x.
**Extrapolated 54-run total ≈ 50-90 min** on CPU (revised up after seeing 405 s for
resnet18/Aircraft/full/T12 during the curves run — sustained load roughly doubles
the cold-start numbers). Worst single run ~3.5-7 min. Well within reason → ran it.

---

## New script: `Stage_2/Work/06_training_curves/training_curves.py` (deliverable 2)

The standard-FM sweep only stored the final loss, so there were no curves. This
recomputes the full per-epoch history (fresh, not from checkpoints) for standard FM
and rolled-out FM (T=4 and T=12) on one representative full-data run per combo
(K=full, seed 0), reusing `flow_matching_loss` / `rolled_out_loss`, same
architecture / optimiser / lr / epochs / seed convention. Outputs
`Stage_2/results/training_curves/training_curves.{png,json}`; log-y, one panel per
combo, plus an automated stability read-out (finite / net-decrease / no-divergence
/ flattening).

**Ran it — all 9 curves STABLE** (`[FDNL]` on every one). Final train loss:

| combo | standard FM | rolled-out T=4 | rolled-out T=12 |
| --- | --- | --- | --- |
| resnet18 / DTD | 1.07 -> 0.235 | 1.08 -> 0.086 | 1.08 -> 0.066 |
| resnet18 / Aircraft | 1.56 -> 0.190 | 1.56 -> 0.097 | 1.56 -> 0.061 |
| dinov2 / Aircraft | 6.24 -> 0.762 | 6.25 -> 0.210 | 6.25 -> 0.201 |

Notes: standard and rolled-out losses measure *different* quantities
(`||v_pred - v_target||^2` vs `||z_hat_T - p||^2`), so the columns aren't directly
comparable — but each objective converges cleanly. Rolled-out shows occasional
single-epoch spikes (visible ~epoch 70/140/180 for resnet18/Aircraft) that recover
immediately — a full-batch backprop-through-Euler-chain artefact, not divergence;
the stability check still passes. T=12 reaches a slightly lower final loss than T=4
for its objective, as expected (more steps to shape the endpoint). 15.5 min total.

---

## New script: `Stage_2/Work/05_rolled_out_fm/train_rolled_out_fm.py`

The 54-run rolled-out sweep, same pattern as `02_standard_fm/train_standard_fm.py`:
same COMBOS / K_VALUES / SEEDS / EPOCHS / lr / seed convention, only the loss
differs (`rolled_out_loss` at that run's T). Grid = 27 x T∈{4,12} = 54. `--dry-run`
prints the grid. Writes `Stage_2/results/rolled_out_fm_models/` — kept **separate**
from the standard-FM checkpoints because a rolled-out model is only valid at its
training T. Checkpoint dict adds `"T"` and `"training": "rolled_out"`.

**Ran it — 54/54, 53.0 min on CPU** (estimate was 50–90 min). All final losses
finite, decreasing with T (T12 < T4 in every group), seed spreads mostly <15%
(a couple of K10 configs at 17–29% — small subsets are noisier). Results:
`Stage_2/results/rolled_out_fm_models/sweep_summary.json` + 54 checkpoints.

---

## classification_eval.py updated for rolled-out (done)

`Stage_2/Work/03_classification_eval/classification_eval.py` now evaluates both
`flow_matching_models/` (standard, at every T∈{4,12}) and `rolled_out_fm_models/`
(rolled-out, at **only** `ckpt["T"]` — `steps_for(ckpt)`). `classify_by_cosine` /
`evaluate` unchanged. New `training` column ("standard" / "rolled_out") on every
run row and aggregated group. Write guard: refuses to write unless 27 standard +
54 rolled-out present and nothing skipped (`--allow-partial` / `--no-write` to
override). Toy test still passes.

**Ran the full combined eval — 108 run rows.** `Stage_2/results/classification_eval/
eval_{runs.csv,summary.json}`. Delta_Acc (mean ± std over 3 seeds, pp):

| enc / dataset | K | std T4 | std T12 | **roll T4** | **roll T12** |
| --- | --- | --- | --- | --- | --- |
| DINOv2 / Aircraft | 5 | +6.83 | +12.60 | −5.39 ±4.0 | −4.87 ±3.7 |
| DINOv2 / Aircraft | 10 | +12.18 | +17.84 | +2.65 | +0.64 |
| DINOv2 / Aircraft | full | +17.19 | **+22.24** | +5.98 | +3.61 |
| ResNet-18 / Aircraft | 5 | −0.19 | +1.40 | −6.09 | −5.96 |
| ResNet-18 / Aircraft | 10 | +0.96 | +3.04 | −3.25 | −4.83 |
| ResNet-18 / Aircraft | full | +3.39 | +5.64 | −5.54 | −2.77 |
| ResNet-18 / DTD | 5 | −16.05 | −8.81 | −24.02 | −26.05 |
| ResNet-18 / DTD | 10 | −16.03 | −5.05 | −27.39 | −26.61 |
| ResNet-18 / DTD | full | −12.32 | −1.68 | −28.49 | **−30.32** |

**Rolled-out FM underperforms standard FM in every cell.** Where standard helps
most (DINOv2/Aircraft/full/T12 +22.2) rolled-out manages +3.6; where standard is
neutral (ResNet/Aircraft) rolled-out is −3 to −6; where standard is bad
(ResNet/DTD) rolled-out is −24 to −30. And for rolled-out, **more steps = worse**
(T4 ≥ T12 in most cells) — the opposite of standard FM.

### Why: rolled-out collapses classes toward a shared region

For 3 strongly-negative rolled-out checkpoints, mean pairwise cosine between test
features of **different** classes (L2-normalized as in the classify rule),
compared to standard FM and the raw baseline at the same T:

| checkpoint (eval T) | view | cos, **diff-class** | cos, same-class | sep gap | mean \|z\| | test acc |
| --- | --- | --- | --- | --- | --- | --- |
| resnet18/DTD/full (T=4) | raw baseline | 0.445 | 0.566 | 0.121 | 23.8 | 58.8% |
| | standard FM | 0.013 | 0.046 | 0.033 | 9.6 | 46.0% |
| | **rolled-out FM** | **0.057** | 0.083 | **0.027** | 7.3 | 30.0% |
| resnet18/DTD/K10 (T=12) | raw baseline | 0.445 | 0.566 | 0.121 | 23.8 | 50.7% |
| | standard FM | 0.016 | 0.071 | 0.055 | 10.7 | 45.3% |
| | **rolled-out FM** | **0.044** | 0.086 | **0.042** | 7.7 | 26.0% |
| resnet18/Aircraft/K5 (T=12) | raw baseline | 0.779 | 0.811 | 0.032 | 28.9 | 16.6% |
| | standard FM | 0.034 | 0.088 | 0.054 | 9.4 | 18.1% |
| | **rolled-out FM** | **0.078** | 0.112 | **0.034** | 6.3 | 10.5% |

("sep gap" = same-class − diff-class mean cosine.)

- Rolled-out's diff-class cosine is **2–4× standard FM's** every time — different
  classes' transported features sit closer together.
- Rolled-out's separation gap is **smaller than standard FM's** every time — classes
  less distinguishable, tracking the lower accuracy.
- Rolled-out contracts hardest (mean \|z\| ~6–8 vs standard ~9–11 vs raw ~24–29).

`Lroll = mean(||z_hat_T − p||^2)` is minimised *on average* most cheaply by pulling
every class into a common neighbourhood of the prototype cloud, not by routing
each feature to its own correct target. Standard FM supervises the per-point
velocity toward each feature's own prototype, so it keeps more inter-class
direction. Caveat: **both** FMs cut diff-class cosine massively vs the raw baseline
(0.44–0.78 → 0.01–0.08) — raw ResNet features are pathologically co-linear; the
gap that matters for the negative Delta_Acc is rolled-out vs standard.

---

## feature_space_viz.py — third panel added

`feature_space_viz.py` now auto-adds an "after rolled-out FM" panel when the
matching rolled-out checkpoint exists (`rolled_out_fm_models/{enc}__{slug}__K{k}__T{steps}__seed{seed}.pt`
— same enc/ds/K/seed, T = `--steps`). Default run = DINOv2/Aircraft/full/seed0,
T=12. The joint PCA, shared axes, colours and layout all key off the `views` dict,
so nothing else changed. `results/feature_space_viz/feature_space_viz.png` is now
3 panels: original / after standard FM / after rolled-out FM.

The rolled-out panel is the collapse table made visual — the 9 classes contract
into one tight, colour-mixed ball (feature norm 49.8 → **8.5**, vs standard FM's
14.9). Prototype stars are in identical positions across all 3 panels (not
transported).

---

## New script: `Stage_2/Work/03_classification_eval/accuracy_vs_k.py` (deliverable 1 plot)

Reads `eval_summary.json` (nothing re-evaluated), plots top-1 test accuracy vs K:
3 subplots (one per encoder/dataset), 5 series each — baseline, standard-FM T=4/T=12,
rolled-out-FM T=4/T=12 — error bars = seed std. Colour = method (black baseline,
blue standard, orange rolled-out); line style = T (dashed T=4, solid T=12).
Follows Stage 1's `05_analysis` accuracy-vs-K styling.
`results/classification_eval/accuracy_vs_k.png`.

Reads it off the plot at a glance: DINOv2/Aircraft — standard FM (blue) well above
baseline (black), rolled-out (orange) below standard and only crossing baseline at
K≥10. ResNet/DTD — baseline on top, standard just under it, rolled-out far below.
ResNet/Aircraft — standard above baseline at K≥10, rolled-out below throughout.

---

## Not yet started

- Delta_Acc *table* formatted for the report/slides (numbers all in
  `eval_summary.json`; the accuracy-vs-K plot already covers the visual)
- Optional: rerun `flow_trajectories.py` with a rolled-out checkpoint to compare paths
- Write-up pulling the 4 deliverables together
