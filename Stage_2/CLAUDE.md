# Stage 2 — Context for Claude Code

## Project
"Flow Matching as a Layer" — 4-stage CV project, University of Haifa (Yousef Shihade & Mira Bitar).
Research question: can a Flow Matching module replace a standard classifier layer and outperform it?
Stage 1 (complete) built the frozen-encoder baseline this depends on. Stage 2 (in progress) adds
an FM layer on top of Stage 1's exact features/prototypes/subsets/seeds — see Stage_2/stage_2.pdf
for the full spec.

## Current status (update this section as work progresses)
- Environment: C:\cvlab_env, PyTorch 2.6.0+cu124, CPU-only (no NVIDIA GPU on this machine — Intel
  integrated graphics only, confirmed via nvidia-smi not found). All FM training runs on CPU;
  confirmed feasible via direct timing before running full sweeps.
- Stage 1 cached features (9 .pt files, Stage_1/features/) transferred from partner's machine and
  verified against Step 02's README numbers (25,640 total images, correct shapes/dtypes).
- cvlab/paths.py modified: Stage-1-root detection relaxed to not require Data/ (only Work/ +
  src/cvlab/), since this machine only has cached features, not raw datasets. Backward-compatible.
  Documented in Stage_2/NOTES_FOR_MIRA.md.
- New module Stage_1/src/cvlab/flow_matching.py: VelocityNetwork (MLP, 2 hidden layers width 512,
  SiLU, feature+scalar-t input, feature-dim output), standard FM training loss, Euler-step
  inference. Validated on synthetic toy data (3 well-separated fake classes, dim 16): mean distance
  to correct prototype went 5.90 -> 0.33, 100% correct nearest-prototype rate.
  euler_inference_trajectory added (additive; euler_inference untouched): same Euler stepping,
  returns all (T+1, N, D) states — trajectory[0]==input, trajectory[-1]==euler_inference output.
  For the flow-trajectory deliverable. Toy-tested in Work/01_velocity_network/.
  rolled_out_loss added (additive): spec's L_roll — full T-step Euler transport from source
  feature, mean((z_hat_T - prototype)**2), graph kept through the chain. torch.mean reduction
  matching flow_matching_loss (fair-comparison). Deterministic given seed (no internal rand).
- Standard FM 27-run sweep COMPLETE (3 encoder/dataset combos x K in {5,10,full} x 3 seeds,
  Stage 1's exact k-shot subsets/seeds, reused not recomputed).
  Results: Stage_2/results/flow_matching_models/sweep_summary.json + 27 checkpoints.
  All 27 final losses checked: stable, no NaN, tight seed agreement (1.2-7.2% per group).
- Sweep is now a reviewable script: Stage_2/Work/02_standard_fm/train_standard_fm.py
  (full-batch, 200 epochs, AdamW lr 1e-3, CPU; --dry-run prints the grid). Re-ran it to
  regenerate all 27 checkpoints + sweep_summary.json (3.9 min on CPU). K=full runs
  reproduce the pre-regen artifacts BIT-IDENTICALLY (weights + loss); K=5/10 losses stay
  within each group's seed spread but weights differ — the original K=5/10 scratch run
  used an unrecoverable RNG path. Carrying forward the regenerated set (has a script,
  reproduces K=full exactly, K=5/10 built via cvlab.data.make_kshot_subset). Pre-regen
  backup saved in the session scratchpad. See Stage_2/NOTES_FOR_MIRA.md.
- Rolled-out sweep COMPLETE: Stage_2/Work/05_rolled_out_fm/train_rolled_out_fm.py -> 54 ckpts
  + sweep_summary.json in Stage_2/results/rolled_out_fm_models/ (separate dir; rolled-out only
  valid at its training T). 53 min on CPU (timed first: est 50-90 min). All losses finite,
  decreasing with T, seed spreads mostly <15%. rolled_out_loss lives in cvlab.flow_matching.
- Training curves DONE (deliverable 2): Stage_2/Work/06_training_curves/training_curves.py ->
  results/training_curves/training_curves.{png,json}. Standard + rolled-out T4/T12, one
  full-data run per combo (seed 0), fresh per-epoch recompute. ALL 9 STABLE (finite /
  decreasing / no-divergence / flattening). Rolled-out has occasional 1-epoch spikes that
  recover — backprop-through-Euler-chain artifact, not divergence.
- Classification eval COMPLETE (standard + rolled-out): Stage_2/Work/03_classification_eval/
  classification_eval.py updated — standard ckpts at both T, rolled-out at ckpt["T"] only,
  new `training` column, write-guard needs 27+54. 108 run rows in eval_{runs.csv,summary.json}.
  acc_baseline reproduces Step 04 exactly. Delta_Acc: standard FM positive for DINOv2/Aircraft
  (+7..+22), mildly + for ResNet/Aircraft, negative ResNet/DTD; T12>T4. ROLLED-OUT
  UNDERPERFORMS STANDARD IN EVERY CELL — DINOv2/Aircraft/full only +3.6 (T12) vs standard +22.2;
  ResNet/DTD −24..−30. For rolled-out, more steps = worse (T4 ≥ T12). Cause: rolled-out
  collapses classes toward a shared region — inter-class cosine 2-4x standard FM's, smaller
  intra-vs-inter gap, mean|z| ~6-8 vs standard ~9-11. Table in NOTES_FOR_MIRA.md.
- Accuracy-vs-K plot DONE (deliverable 1): Stage_2/Work/03_classification_eval/accuracy_vs_k.py
  -> results/classification_eval/accuracy_vs_k.png. 3 subplots (per encoder/dataset), 5 series
  (baseline / standard-T4/T12 / rolled-T4/T12), error bars = seed std, reads eval_summary.json.
- Feature-space viz DONE with all 3 panels: Stage_2/Work/04_feature_space_viz/feature_space_viz.py
  -> results/feature_space_viz/feature_space_viz.png. original / after-standard-FM / after-
  rolled-out-FM (dinov2/Aircraft/full/seed0, T=12). Same 9 classes/images/colors as Stage 1;
  one jointly-fit PCA, shared axes. Rolled-out panel visibly collapses to a tight ball
  (norm -> 8.5 vs standard 14.9). Third panel auto-added when the rolled-out ckpt exists.
- IMPORTANT CONVENTION CONFIRMED: Stage 1 prototypes are L2-normalized (unit norm). Training
  features z_i are used RAW (not normalized) in the FM interpolation z_t=(1-t)z_i+t*prototype.
  This is why DINOv2 losses run 3-4x higher than ResNet-18's (DINOv2 raw feature norm ~48-50 vs
  ResNet-18's ~14-28) — NOT a bug, just means loss values aren't comparable ACROSS encoders.
  Assignment says "transport the frozen image feature," consistent with using it raw.

## OPEN QUESTION — RESOLVED (2026-08-27)
Does the classification code L2-normalize z_hat_T before cosine similarity against prototypes,
matching Stage 1's classify_by_prototype? Now: YES, by construction. euler_inference returns
z_hat_T raw (the classify step simply didn't exist yet). classification_eval.classify_by_cosine
is Stage 1's run_prototype_classifier inner step verbatim: l2_normalize(features) @ prototypes.T,
argmax, with l2_normalize imported from cvlab.prototypes. FM path and baseline use the same
function. Nuance: since compute_prototypes returns unit-norm rows, normalizing the query alone
can't change the argmax — the convention that actually matters is cosine vs Euclidean (cdist on
raw features is wrong: raw norms 14-50, transported ~16 for DINOv2, prototypes norm 1).
Verified: acc_baseline from the eval reproduces Step 04's prototype accuracies exactly.

## Remaining
- Delta_Acc *table* for the report/slides (all numbers are in eval_summary.json + the
  accuracy_vs_k plot; just needs formatting into a table if the report wants one).
- Optional: rerun flow_trajectories.py with a rolled-out checkpoint to compare paths.
- Write-up / slides pulling the 4 deliverables together.

## Done — deliverable notes kept for the findings
- Feature-space visualization: DONE, 3 panels (original / standard FM / rolled-out FM) —
  Stage_2/Work/04_feature_space_viz/feature_space_viz.py -> results/feature_space_viz/
  feature_space_viz.png. Same 9 Aircraft classes / test images / colors / star-prototypes as
  Stage 1's feature-viz (reuses pick_diverse_classes + plot_projection). Raw + transported +
  prototypes all L2-normalized, ONE PCA fit jointly across all views, panels share projection
  + axis scale. Third panel auto-added when the matching rolled-out ckpt (same enc/ds/K/seed,
  T=steps) exists. Default DINOv2/Aircraft/full/seed0, T=12. Rolled-out panel collapses to a
  tight ball (norm 49.8 -> 8.5 vs standard's 14.9).
- Flow trajectories + reverse flow: DONE for standard FM — Stage_2/Work/04_feature_space_viz/
  flow_trajectories.py -> results/feature_space_viz/flow_trajectories_{forward,reverse}.png.
  Small multiples, per-panel jointly-fit PCA (feature_space_viz recipe), T=12. Forward: 4 test
  images via euler_inference_trajectory, every Euler step + t=0.5 diamond/callout + t=0 circle
  + t=1 square + own/sibling prototypes + dotted t1->prototype gap. Reverse: local
  reverse_trajectory_from_prototype (w -= dt*v(w,t), time 1->0) from each prototype over the
  class cloud.
  FINDING: forward flow does NOT land on the prototype — cos(state, own prototype) FALLS along
  the path (~0.8 -> ~0.4) while argmax/100 stays correct. Standard FM at T=12 improves
  classification via RELATIVE separation (siblings drop faster), not by transporting onto the
  prototype (transported norm 49.8->14.9, big undershoot). Reverse flow is ~a no-op: prototype
  already sits at the real-data centroid (cos 0.98-0.99), -v_theta dips it slightly (->0.94-0.96).

## Professor's live guidance (from Stage 1 presentation, maps to the above)
- "Visualize before and after adding the layer" -> the feature-space visualization requirement
- "Visualize a path from a sample" -> the flow trajectory requirement
- "Opposite direction, -v" -> the optional reverse-flow extension
- "Compare at t=0.5, not just t=1" -> show intermediate states in visualizations, not just endpoints
  (does not change the classification protocol itself, which evaluates at the final T-step output)

## Timeline & working context
5 days total for Stage 2, working solo (partner has an upcoming exam and will review afterward —
keep Stage_2/NOTES_FOR_MIRA.md updated with every shared-code change and bug caught, in the same
tone as this file). User also has a personal exam the week after the Stage 2 presentation, so some
of the 5 days include exam study — don't assume every day is fully available.

## Working conventions established so far (follow these)
- Never recompute Stage 1's k-shot subsets or prototypes — reuse cvlab's existing functions
  (data.py, prototypes.py) directly, same seeds, same convention.
- Before running a full sweep, time the single most expensive configuration first and give an
  honest extrapolated estimate — don't assume all configs cost the same as the cheapest one.
- After any claim of "done" or "verified," show the actual raw output/file contents, not a
  summary description. Summaries have repeatedly not matched what actually happened in this
  project so far — always ask for and check literal output.
- Document every change to shared Stage 1 code (not just new Stage 2 code) in NOTES_FOR_MIRA.md,
  with what changed, why, and why it's safe/backward-compatible.
