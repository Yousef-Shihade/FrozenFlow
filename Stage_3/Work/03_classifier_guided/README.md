# Step 03 — Strategy 2: classifier-guided targets with standard FM

Instead of differentiating the classification loss through the whole rollout, the frozen
classifier is used to **construct an explicit target** for each feature, and the flow is
trained toward it with an ordinary Flow Matching update:

1. Roll $z$ through the current flow to get $\hat z$.
2. Compute the classification loss of the frozen classifier on $\hat z$.
3. Build $\hat z'$ by taking gradient steps on $\hat z$ in feature space.
4. Source $z$, target $\hat z'$.
5. Standard FM update: $t\sim\mathcal{U}(0,1)$, $z_t=(1-t)z+t\hat z'$, $u=\hat z'-z$,
   $\mathcal{L}_\mathrm{FM}=\lVert v(z_t,t)-u\rVert^2$.
6. Recompute the targets as the flow changes.

**99 runs**: 11 configurations × 3 combinations × 3 seeds, 16 minutes of training.

---

## Headline

| Encoder / dataset | Frozen probe | Strategy 2 (main) | Δ |
| --- | --- | --- | --- |
| ResNet-18 / DTD | 52.04% | 53.07% | **+1.03 ± 0.75 pp** |
| DINOv2 / FGVC-Aircraft | 51.54% | 52.85% | **+1.31 ± 0.65 pp** |
| ResNet-18 / FGVC-Aircraft | 27.87% | 29.72% | **+1.85 ± 1.00 pp** |

**91 of 99 runs (92%) improve on the frozen probe, and all 11 configurations are positive.**
Best configuration: `refresh=10` at **+1.82 pp**.

---

## 1 · What the guidance actually produces

The flow can never beat the targets it is aimed at, so those are measured first.

| Encoder / dataset | base acc | base CE | target acc (η=1) | target CE (η=1) |
| --- | --- | --- | --- | --- |
| ResNet-18 / DTD | 96.38% | 0.594 | 99.15% | 0.316 |
| DINOv2 / Aircraft | **100.00%** | 0.005 | **100.00%** | 0.001 |
| ResNet-18 / Aircraft | **100.00%** | 0.022 | **100.00%** | 0.001 |

**On two of three combinations the targets start at 100% accuracy and stay there.** The probe
already classifies its own K-shot training set perfectly (step 02), so there is no
*correctness* for the guidance to add. What the gradient still does is collapse the target
cross-entropy toward zero — it pushes features into **higher-confidence** regions.

So on those combinations Strategy 2 is not "move features so the classifier gets them right".
It is a **margin-increasing transform**: move features so the classifier is more certain about
answers it already gives. That it still generalises — +1.3 to +1.9 pp — is the interesting
part.

---

## 2 · The sweep

The brief names exactly these four knobs. Each is varied around the main configuration
(η = 1, one step, refresh every epoch, normalised, lr $10^{-4}$).

| config | Δ vs. probe | selected epoch | moved |
| --- | --- | --- | --- |
| **main** | +1.40 ± 0.79 pp | 119 | 17.1% |
| η = 0.5 | +1.53 ± 0.66 pp | 109 | 12.0% |
| η = 2 | +1.54 ± 0.72 pp | 91 | 19.5% |
| η = 4 | +1.18 ± 0.83 pp | 103 | 34.0% |
| 3 target steps | +1.12 ± 0.84 pp | 83 | 29.7% |
| **refresh every 10** | **+1.82 ± 0.80 pp** | 116 | 11.3% |
| refresh every 50 | +1.59 ± 0.76 pp | 143 | 6.1% |
| unnormalised, η = 10³ | +0.16 ± 0.24 pp | 99 | 2.1% |
| unnormalised, η = 10⁴ | +0.46 ± 0.46 pp | 133 | 4.3% |
| lr $10^{-3}$ | +0.94 ± 0.71 pp | 17 | 10.8% |
| lr $10^{-5}$ | +0.89 ± 0.80 pp | 144 | 8.4% |

**Refreshing the targets *less* often is better.** The brief's step 6 says to recompute targets
as the flow changes; recomputing every epoch gives +1.40 pp, every 10 epochs **+1.82 pp**, every
50 epochs +1.59 pp. Recomputing constantly means chasing a target that moves with the flow
chasing it. Letting it go stale for a few epochs makes the objective closer to stationary, and
it trains better.

**Normalising the guidance step matters more than any other choice** — +0.16/+0.46 pp
unnormalised against +1.4 to +1.8 normalised. The reason is measured: the raw gradient norm
differs **100×** across combinations (7.4e-4 on DTD, 7.1e-6 on DINOv2), so one nominal step
size means three very different effective steps.

**Step size is otherwise forgiving** (η ∈ [0.5, 2] all within 0.15 pp), and **more target steps
is worse** — 3 steps moves features 30% of ‖z‖ and loses 0.3 pp against 1 step.

---

## 3 · Against Strategy 1 — indistinguishable on accuracy

Strategy 2's main configuration already uses lr $10^{-4}$, because step 02 measured $10^{-3}$
to be harmful and that lesson was carried in. Strategy 1's main was $10^{-3}$. Comparing the
two "main" configurations would compare Strategy 2 *with* step 02's finding against Strategy 1
*without* it — a difference in tuning, not in objective. So the learning rate is held fixed:

| lr | Strategy 1 | Strategy 2 | difference |
| --- | --- | --- | --- |
| $10^{-3}$ | +0.15 pp | **+0.94 pp** | **+0.79** |
| $10^{-4}$ | +1.27 pp | +1.40 pp | +0.13 (*p* = 0.80) |
| $10^{-5}$ | **+1.21 pp** | +0.89 pp | −0.31 |

Per combination at $10^{-4}$ they trade wins: Strategy 2 is better on DTD (+0.92) and
ResNet-18/Aircraft (+0.75), Strategy 1 is clearly better on DINOv2/Aircraft (−1.28). Even
best-configuration against best-configuration — selected after the fact on *both* sides, so the
comparison flatters both — the gap is +0.55 pp at *p* = 0.30.

**The answer to the brief's second question is therefore: no, classifier-guided training does
not beat end-to-end rolled-out training on accuracy.**

### Where Strategy 2 *is* better: robustness

At lr $10^{-3}$ Strategy 1 collapses to +0.15 pp while Strategy 2 holds +0.94 pp, and there is
a mechanism for it. Strategy 1 optimises the classification loss **directly**, so a large step
lets it memorise training labels within a handful of epochs (median selected epoch: 9).
Strategy 2 optimises an **MSE regression onto a target vector** — the labels enter only through
where that target was placed, so the same step size cannot memorise them nearly as fast.

| | configs positive | runs improving |
| --- | --- | --- |
| Strategy 1 | 9 / 9 | 67 / 81 (83%) |
| Strategy 2 | **11 / 11** | **91 / 99 (92%)** |

---

## Design decisions

- **Normalised guidance step by default**, so η is the distance actually travelled and is
  comparable to ‖z‖ ≈ 24–50 across combinations. The unnormalised variant is run anyway, and
  measured to be worse (section 2).
- **lr $10^{-4}$ as the main setting**, carrying step 02's finding rather than re-deriving it —
  and section 3 confirms the lesson transfers.
- **Targets are detached** before the FM update, which is what makes it an ordinary Flow
  Matching step rather than a second end-to-end objective.
- **Validation-accuracy checkpointing**, matching step 02 and Stage 1, so both strategies are
  selected by the same rule.

## Outputs

| File | What it holds |
| --- | --- |
| `../../results/strategy2_runs.pt` | all 99 runs, every training curve, the target diagnostic |
| `tables/strategy2_all_runs.csv` | one row per run |
| `tables/strategy2_config_summary.csv` | the sweep table above |
| `tables/strategy2_main_summary.csv` | per-combination, main configuration |
| `tables/target_quality.csv` | target accuracy and CE at each η |
| `tables/strategy_comparison_matched_lr.csv` | Strategy 1 vs 2 at matched learning rate |
| `tables/matched_learning_rate_comparison.csv` | the three-row lr table above |
| `plots/training_curves.png` | train/val/target curves (required deliverable) |
| `plots/strategy2_results.png` | every configuration, and the matched-lr comparison |

**Next:** step 04 consolidates both strategies into the comparison the brief requires.
