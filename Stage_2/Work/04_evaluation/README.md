# Step 04 — Evaluation

**A Computer Vision Project — *Flow Matching as a Layer*** · Stage 2
University of Haifa

---

## Purpose

Assemble the first two of Stage 2's four results from the results steps 2 and 3
already saved.

**Nothing is retrained and no rollout is re-run.** Every number here is read off disk, which
is what guarantees the table, the plots and the per-run CSVs all describe exactly the same
models — a confusion matrix or a plot built from a re-trained model would silently describe a
*different* one.

| Result | Produced |
| --- | --- |
| 1. Classification results — all methods, $\Delta\mathrm{Acc}$, accuracy-vs-$K$ with error bars | **here** |
| 2. Training curves — both objectives, verifying stable training | **here** |
| 3. Feature-space visualisations | step 5 |
| 4. Flow trajectories | step 5 |

---

## Quick start

```text
1. Open  code/04_evaluation.ipynb  in VS Code
2. Select the kernel:  Python (CVLAB Stage 1)
3. Run All
```

**Runtime: seconds.** Requires steps 1–3's artefacts in `Stage_2/results/`.

---

## The complete accuracy table

Top-1 accuracy on the complete official test split, mean ± std over 3 seeds.
**45 cells** = 3 encoder/dataset combinations × 3 training sizes × 5 methods.

| Encoder / dataset | $K$ | Baseline | std FM $T{=}4$ | std FM $T{=}12$ | roll FM $T{=}4$ | roll FM $T{=}12$ |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet-18 / DTD | 5 | 46.10 ± 1.89 | 44.47 ± 1.05 | 44.79 ± 1.01 | 40.99 ± 1.60 | 40.87 ± 1.46 |
| ResNet-18 / DTD | 10 | 51.67 ± 1.33 | 50.11 ± 0.86 | 50.34 ± 0.66 | 44.77 ± 1.58 | 45.32 ± 1.23 |
| ResNet-18 / DTD | full | 58.83 *(1 run)* | **59.59 ± 0.61** | 59.56 ± 0.68 | 55.85 ± 0.47 | 55.82 ± 0.32 |
| ResNet-18 / Aircraft | 5 | 16.46 ± 0.39 | 18.81 ± 1.01 | **18.94 ± 1.02** | 16.77 ± 0.84 | 17.03 ± 0.87 |
| ResNet-18 / Aircraft | 10 | 20.19 ± 1.14 | 23.68 ± 0.39 | **23.91 ± 0.32** | 20.34 ± 0.53 | 20.09 ± 0.24 |
| ResNet-18 / Aircraft | full | 25.47 *(1 run)* | **31.04 ± 0.93** | 30.97 ± 0.56 | 25.64 ± 1.72 | 25.28 ± 1.69 |
| DINOv2 / Aircraft | 5 | 23.79 ± 0.47 | 33.30 ± 0.68 | 32.82 ± 0.57 | **34.13 ± 0.59** | 33.86 ± 0.58 |
| DINOv2 / Aircraft | 10 | 27.74 ± 0.98 | 43.27 ± 1.09 | 42.48 ± 1.16 | **45.42 ± 1.10** | 45.35 ± 0.83 |
| DINOv2 / Aircraft | full | 34.26 *(1 run)* | **56.52 ± 0.27** | 55.70 ± 0.31 | 52.05 ± 0.91 | 52.08 ± 0.82 |

Bold marks the best FM cell in each row **when it beats the baseline**; three rows have no
bold because no flow configuration does.

The Stage 1 full-data baseline is a single deterministic run — no subsampling left to vary —
so it carries no spread, and is labelled rather than shown with a meaningless `0.00`.

Every mean ± std here is produced by Stage 1's `cvlab.evaluation.summarize_runs`, which is the
one place in the project that decides between the sample and population standard deviation
(it uses `ddof=1`). This table restates baseline runs that Stage 1 already published, so
aggregating them independently here would risk quoting a different spread for an identical
experiment. Section 2 asserts that all 36 FM cells match steps 2–3 and all 9 baseline cells
match Stage 1's own table — the notebook fails rather than publishing a table that disagrees
with the notebooks it summarises.

### Change relative to the baseline

| Encoder / dataset | $K$ | std $T{=}4$ | std $T{=}12$ | roll $T{=}4$ | roll $T{=}12$ |
| --- | --- | --- | --- | --- | --- |
| ResNet-18 / DTD | 5 | −1.63 | −1.31 | −5.11 | −5.23 |
| ResNet-18 / DTD | 10 | −1.56 | −1.33 | **−6.90** | −6.35 |
| ResNet-18 / DTD | full | +0.76 | +0.73 | −2.98 | −3.01 |
| ResNet-18 / Aircraft | 5 | +2.35 | +2.48 | +0.31 | +0.57 |
| ResNet-18 / Aircraft | 10 | +3.49 | +3.72 | +0.15 | −0.10 |
| ResNet-18 / Aircraft | full | +5.57 | +5.50 | +0.17 | −0.19 |
| DINOv2 / Aircraft | 5 | +9.51 | +9.03 | +10.34 | +10.07 |
| DINOv2 / Aircraft | 10 | +15.53 | +14.74 | +17.68 | +17.61 |
| DINOv2 / Aircraft | full | **+22.26** | +21.44 | +17.79 | +17.82 |

**24 of 36 FM cells improve on the baseline**, ranging from **+22.26 pp** to **−6.90 pp**.

---

## What the results say

### 1. Standard FM is the better objective overall

It wins 14 of 18 head-to-head cells against rolled-out training. Rolled-out only leads on
DINOv2/Aircraft at $K \in \{5, 10\}$ — the strongest encoder in the lowest-data settings.
Step 3 establishes *why*: rolled-out reaches the prototype more closely (on test data too, so
it is not overfitting) but raises similarity to competing prototypes at the same time, so the
transport improves while the decision does not.

### 2. The number of Euler steps barely matters

Across all 18 FM cells the largest $T = 4$ vs $T = 12$ difference is under 1 pp, with an
inconsistent sign. For standard FM this is expected — training never rolls out, so more steps
just integrate the same field more finely. For rolled-out FM it is more notable, since $T$ is
part of training there: a 3× increase in training cost buys nothing measurable.

### 3. Within every combination, the gain grows with $K$

Monotonic in all three, without exception: DTD $-1.3 \to -1.3 \to +0.8$, ResNet-18/Aircraft
$+2.5 \to +3.7 \to +5.6$, DINOv2/Aircraft $+10.3 \to +17.7 \to +22.3$.

The FM layer is the only component with parameters to fit; the prototype baseline is a fixed
average and cannot exploit extra data the same way. This is what turns DTD's $-1.6$ pp deficit
at $K = 5$ into a small positive at full data.

### 4. What does *not* predict the gain

Two plausible explanations, both ruled out by the data:

- **Separability.** Stage 1 measured a separability margin per combination before any training.
  It does not order the gains: DINOv2/Aircraft has the widest margin (0.420) and the largest
  gain, but ResNet-18/DTD has the second-widest (0.172) and the *smallest*, while
  ResNet-18/Aircraft has the narrowest (0.041) and sits in between.
- **Baseline headroom.** "FM helps most where the baseline was weakest" is tempting and false:
  ResNet-18/Aircraft has the weakest baselines of all (16–25 %) yet gains only 2.5–5.6 pp,
  while DINOv2/Aircraft starts higher (24–34 %) and gains 10–22 pp.

The encoder matters more than either. DINOv2's features carry structure a learned transport
can exploit that ResNet-18's, on this dataset, do not.

---

## Training curves and convergence

Representative curves for both objectives at $K = 10$ — the middle setting, noisier than full
data but less erratic than 5-shot.

**The two rows must not be compared to each other.** Standard FM measures velocity error at a
randomly sampled time; rolled-out measures endpoint distance after $T$ steps. Different
quantities on different scales, so they are drawn on separate axes rather than overlaid.

| Objective | Models | Final loss | Loss reduction | Diverged |
| --- | --- | --- | --- | --- |
| Standard FM | 27 | 0.078 – 0.214 | 53.9 – 71.5 % | **0** |
| Rolled-out FM | 54 | 0.022 – 0.089 | 79.4 – 92.9 % | **0** |

All **81 models** converge; none diverges. Rolled-out reaches a much lower final loss and a
much larger reduction — but against its own objective, which is not the one standard FM is
optimising, and not the one accuracy measures.

**Context worth carrying forward:** those zero divergences hold *with* gradient clipping.
Without it, rolled-out training diverged in 1 of 54 runs while standard FM never did in 27
(step 3, `clipping_evidence.png`). Clipping is applied identically to both objectives and is
inert for standard FM — every cell moved by ≤ 0.25 pp — so it costs the comparison nothing.

---

## What predicts the gain: headroom to a trained probe

The gains range from −6.9 pp to +22.3 pp. Separability does not order them, and $K$ orders
them only *within* a combination. One quantity Stage 1 already measured orders **all nine
cells at once** — the **headroom** the untrained baseline left behind:

$$\mathrm{headroom} = \mathrm{Acc}_\mathrm{linear\,probe} - \mathrm{Acc}_\mathrm{prototype}$$

| Encoder / dataset | $K$ | prototype | linear probe | headroom | standard-FM gain | recovered |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet-18 / DTD | 5 | 46.10 | 44.86 | **−1.24** | −1.63 | — |
| ResNet-18 / DTD | 10 | 51.67 | 52.04 | **+0.37** | −1.56 | — |
| ResNet-18 / DTD | full | 58.83 | 62.82 | +3.99 | +0.76 | 19 % |
| ResNet-18 / Aircraft | 5 | 16.46 | 20.76 | +4.30 | +2.35 | 55 % |
| ResNet-18 / Aircraft | 10 | 20.19 | 27.87 | +7.68 | +3.49 | 45 % |
| ResNet-18 / Aircraft | full | 25.47 | 38.15 | +12.68 | +5.57 | 44 % |
| DINOv2 / Aircraft | 5 | 23.79 | 38.19 | +14.40 | +9.51 | 66 % |
| DINOv2 / Aircraft | 10 | 27.74 | 51.54 | +23.80 | +15.53 | 65 % |
| DINOv2 / Aircraft | full | 34.26 | 67.55 | **+33.29** | **+22.26** | 67 % |

**Spearman $\rho = 1.000$** — the nine cells rank identically by headroom and by gain — with
$R^2 = 0.988$ on the fit $\text{gain} = 0.71 \times \text{headroom} - 1.56$.

**It predicts the failures, which is the stronger test.** The fit crosses zero at a headroom
of **2.2 pp**. Exactly two cells fall below it — ResNet-18/DTD at $K = 5$ and $K = 10$, where
the prototype baseline is already level with (at $K = 5$, *ahead* of) the trained probe — and
those are exactly the two cells where the FM layer loses. With no gap to recover, the
transport is pure overhead.

**Stated plainly:** the FM layer recovers roughly **70 % of whatever a trained linear probe
would have gained** over the prototype baseline, minus a fixed ~1.6 pp cost, while keeping the
prototype structure and training no classifier at all.

Two caveats worth stating rather than leaving implicit: nine cells is a small sample, and the
linear probe is a *reference point*, not a mechanism — this says the flow captures most of
what a trained classifier would, not why.

---

## Outputs

### Shared artefact (`Stage_2/results/`)

| File | Contents |
| --- | --- |
| `all_runs_combined.csv` | Every run from all three methods in one long-format table — 129 rows, the single source for step 5 and the report |

### Tables (`tables/`)

`accuracy_table_full.csv` (machine-readable, separate mean/std columns) ·
`accuracy_table_readable.csv` (presentation form) · `delta_vs_baseline.csv` ·
`convergence_summary.csv` · `gain_vs_separability.csv` · `methodology.csv`

### Plots (`plots/`)

| File | Shows |
| --- | --- |
| `accuracy_vs_k.png` | **Core result** — accuracy vs. training-set size, all five methods, error bars |
| `delta_vs_baseline.png` | All 36 FM cells against the baseline, grouped by setting |
| `training_curves.png` | **Core result** — both objectives, mean over 3 seeds, min/max band |
| `gain_analysis.png` | What predicts the gain: separability (it does not) and $K$ (it does) |
| `gain_vs_headroom.png` | The relationship that orders all nine cells, and the break-even threshold |

---

## Next step

**Step 05 — Visualisations.** The remaining two results: feature-space comparisons
(original encoder features vs. after standard FM vs. after rolled-out FM — same test examples,
same class colours, projection fitted jointly across the compared sets), and flow trajectories
showing intermediate states between the original feature and its class prototype.
