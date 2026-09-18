# Step 5 — Feature-Space Visualisations & Flow Trajectories

**A Computer Vision Project — *Flow Matching as a Layer*** · Stage 2
University of Haifa

---

## Purpose

Stage 2's remaining two results, plus a reverse-flow exploration and a look at
intermediate flow times.

| # | Result | What it shows |
| --- | --- | --- |
| 3 | **Feature-space visualisations** | Original features vs. after standard FM vs. after rolled-out FM. Same test examples, same class colours, corresponding prototypes, projection fitted **jointly**. |
| 4 | **Flow trajectories** | Intermediate FM states with the original feature, the transported feature, and the class prototype. |
| — | Reverse flow | Included and labelled exploratory. |

No model is trained here — the representative networks (full data, seed 0) are loaded from
steps 2 and 3, and only forward rollouts on cached test features are computed.

---

## Quick start

```text
1. Open  code/05_visualizations.ipynb  in VS Code
2. Select the kernel:  Python (CVLAB Stage 1)
3. Run All
```

**Runtime: under a minute.** Requires steps 1–3's artefacts.

---

## Two methodological choices

**The projection is fitted jointly.** The compared views must "correspond to the same
low-dimensional representation" for a side-by-side reading to mean anything. A separate PCA
per panel would rotate and rescale
each view independently, so a cluster that *appears* to tighten might merely have been
re-scaled. One PCA is therefore fitted on the concatenation of every set being compared —
original features, standard-FM outputs, rolled-out outputs and all prototypes — and applied
to all of them. Distances are then comparable across panels.

**Classes are chosen by an even stride** through the sorted class list, never hand-picked, so
the subset cannot be quietly selected to flatter the method. Six classes, and the same test
images and colours in every panel of a comparison.

**A 2D projection is not a measurement.** PCA keeps only two directions out of 384 or 512 —
it captures 22.8 % of the variance on ResNet-18/DTD, 23.8 % on ResNet-18/Aircraft and 56.9 %
on DINOv2/Aircraft. Every visual claim made here is therefore re-measured in the full space
(section below), and the plots are read as illustrations of a trend rather than as evidence.

---

## The visual claim, measured in full dimensionality

- **spread** — mean distance from a class's test features to that class's own prototype;
- **separation** — mean distance between different prototypes (fixed by the prototypes, so it
  is a constant reference the flow cannot change);
- **ratio** — spread / separation. Lower = a tighter class relative to how far apart classes are.

| Encoder / dataset | original | standard FM | rolled-out FM |
| --- | --- | --- | --- |
| ResNet-18 / DTD | 1.038 | 0.616 | **0.534** |
| ResNet-18 / FGVC-Aircraft | 1.599 | 0.777 | **0.623** |
| DINOv2 / FGVC-Aircraft | 0.772 | 0.365 | **0.316** |

Both flows genuinely contract each class toward its prototype, and **rolled-out contracts more
in every combination** — consistent with step 3, where it reached the prototype more closely
yet classified worse.

This is the point the whole stage turns on: **contraction toward the correct prototype is not
the same thing as separation from the wrong ones.** The plots show contraction; the accuracy
table shows it does not always help.

---

## The most informative result: accuracy along the rollout

Classifying the intermediate state $\hat z_k$ at every Euler step (`accuracy_by_step.png`,
$T = 12$) separates the two objectives more sharply than any other measurement in Stage 2.

| | Standard FM | Rolled-out FM |
| --- | --- | --- |
| ResNet-18 / DTD | 58.8 → peaks **60.4 at step 8** → back to 58.8 | **falls to 50.0 at step 5**, recovers to 55.5 |
| ResNet-18 / Aircraft | rises steadily to **31.6 at step 11**, ends 31.3 | **falls to 16.3 at step 6**, recovers to 25.7 |
| DINOv2 / Aircraft | rises steadily 34.3 → **56.05**, its own maximum | peaks **53.6 at step 10**, ends 52.3 |

Note that neither objective necessarily *ends* at its best step. On ResNet-18/DTD standard FM
climbs to 60.4 by step 8 and then gives the entire gain back, finishing at 58.8 — its own
starting point. That is the same phenomenon as finding 3 in the top-level README ($T$ barely
matters) seen at finer resolution: past a certain point the extra Euler steps are not adding
anything, and here they actively cost. These are single-seed ($K = \text{full}$, seed 0)
curves, so they end exactly on the seed-0 entry of the run tables, not on the 3-seed means
reported in step 4.

**Rolled-out training sends the features on a detour through a region where they are markedly
*less* classifiable, and only pulls them back at the final step.** On ResNet-18/Aircraft the
exact midpoint of the rollout (step 6 of 12) is **9.2 pp worse than doing nothing at all**.

This follows directly from the objective. $\mathcal{L}_\mathrm{roll}$ supervises **only**
$\hat z_T$; every intermediate state is unconstrained, so nothing requires the path to be
sensible. Standard FM, by contrast, supervises the velocity at every $t \sim \mathcal{U}(0,1)$,
which keeps the whole path meaningful.

`flow_trajectories_dinov2.png` shows the same thing geometrically: standard-FM trajectories are
smooth and roughly direct, while rolled-out trajectories **zigzag** — arbitrary paths that
happen to terminate near the right place.

### Why this matters beyond Stage 2

A flow whose intermediate states are meaningless is a flow that cannot be truncated, inspected,
or reused at partial depth. Standard FM's monotone curve means $\hat z_k$ is a usable
representation at any $k$; rolled-out FM's is not.

---

## Reading the feature-space figures

`feature_space_dinov2_aircraft.png` — the combination where the FM layer helps most
(+22.3 pp). Original features overlap heavily in the centre; after either flow the classes
contract onto their prototypes.

`feature_space_resnet18_dtd.png` — the combination where the FM layer **hurts** (−1.6 pp at
$K = 5$; rolled-out is worse still). Included deliberately: the picture looks *superficially
similar* to the successful case — the clusters contract here too — which is exactly why a
visual impression cannot substitute for the accuracy table. Two panels that look alike
correspond to +22 pp and −1.6 pp.

---

## Reverse flow

Integrating the learned field backwards from each prototype,
$\hat z_k = \hat z_{k+1} - \tfrac{1}{T} v_\theta(\hat z_{k+1}, \tfrac{k}{T})$, asks what a
prototype "unfolds" into.

The reverse trajectories travel **0.418** on average, against a mean feature-to-prototype
distance of **0.754** — roughly 55 % of the way back — and move outward into the region the
class's features occupy rather than to any single point.

**This is an extrapolation and is labelled as such in the notebook.** The network was only ever
trained on states produced by the *forward* pass; nothing constrains its behaviour in reverse,
so this is exploratory and is not used to support any claim.

---

## Samples and prototypes at intermediate flow times

A further exploration beyond the two core results. `distance_by_step.png` measures,
at every Euler step, how
far $\hat z_k$ is from **its own** prototype and from the **nearest competing** one — in full
dimensionality, not in the PCA projection. It turns out to explain the two findings above
geometrically rather than just illustrating them.

Three things are visible at once:

**1. The detour is real, and large.** Rolled-out FM moves features *further from every
prototype* before bringing them back:

| Encoder / dataset | start | rolled-out peak | at step | end |
| --- | --- | --- | --- | --- |
| ResNet-18 / DTD | 0.707 | **1.077** | 4 | 0.367 |
| ResNet-18 / Aircraft | 0.447 | **1.114** | 5 | 0.168 |
| DINOv2 / Aircraft | 0.7225 | — (descends throughout) | — | 0.277 |

On ResNet-18/Aircraft the midpoint of the rollout is **149 % further** from the target than
the untransported feature. Standard FM descends monotonically in all three. This is the same
detour the accuracy-per-step curve shows, and the same one the trajectory figure draws —
measured here rather than inferred.

**2. Rolled-out FM is the better transport.** It ends closer to the target prototype than
standard FM in all three combinations (0.367 vs 0.428, 0.168 vs 0.218, 0.277 vs 0.339). It is
doing precisely what $\mathcal{L}_\mathrm{roll}$ asks of it, and doing it better.

**3. And that is exactly why it classifies worse.** The solid and dashed lines stay glued
together the whole way down: whatever pulls a feature toward its own prototype pulls it
toward the nearest rival by almost the same amount. What survives at the end is the *gap*
between the two, and standard FM ends with a wider one in every combination:

| Encoder / dataset | gap at step 0 | standard FM at $T$ | rolled-out at $T$ |
| --- | --- | --- | --- |
| ResNet-18 / DTD | +0.0224 | **+0.0325** | +0.0210 |
| ResNet-18 / Aircraft | −0.0210 | **−0.0123** | −0.0177 |
| DINOv2 / Aircraft | −0.0329 | **+0.0175** | +0.0076 |

The wider gap picks the more accurate method in all three (seed 0, $T = 12$: 58.78 vs 55.48,
31.26 vs 25.68, 56.05 vs 52.33). Two caveats worth stating rather than leaving implicit: three
combinations is not enough to call this a law, and the quantity is not fully independent of
the thing it predicts — with unit-norm prototypes, an example is classified correctly exactly
when its own-prototype distance is the smaller of the two, so a *mean* gap and an accuracy are
related, though not the same statistic. The negative starting gaps on both Aircraft rows are
worth noticing on their own: on average the nearest wrong prototype begins *closer* than the
right one, which is what a 16–25 % baseline looks like geometrically.

---

## Outputs

### Tables (`tables/`)

| File | Contents |
| --- | --- |
| `feature_geometry.csv` | Spread, separation and ratio per view — the full-dimensional backing for the visual claim |
| `accuracy_by_step.csv` | Accuracy of $\hat z_k$ at every Euler step, both objectives, all combinations (78 rows) |
| `distance_by_step.csv` | Distance from $\hat z_k$ to its own prototype *and* to the nearest competing one, per step |
| `methodology.csv` | Every choice made here, including why the projection is joint |

### Plots (`plots/`)

| File | Shows |
| --- | --- |
| `feature_space_dinov2_aircraft.png` | **Result 3** — the success case (+22.3 pp) |
| `feature_space_resnet18_dtd.png` | **Result 3** — the failure case (−1.6 pp), which looks similar |
| `feature_space_resnet18_aircraft.png` | **Result 3** — the third combination |
| `flow_trajectories_dinov2.png` | **Result 4** — smooth vs. zigzagging paths |
| `flow_trajectories_resnet18_dtd.png` | **Result 4** — the same contrast on DTD |
| `accuracy_by_step.png` | Accuracy of the intermediate state at every step — the mid-rollout collapse |
| `distance_by_step.png` | Samples vs. prototypes at intermediate flow times — own prototype against nearest rival |
| `contraction_measured.png` | The contraction claim, measured in full dimensionality |
| `reverse_flow.png` | Exploratory, clearly labelled |

---

## Stage 2 results — complete

| # | Result | Where |
| --- | --- | --- |
| 1 | Classification results, $\Delta\mathrm{Acc}$, accuracy-vs-$K$ with error bars | step 4 |
| 2 | Training curves, both objectives | step 4 |
| 3 | Feature-space visualisations | **step 5** |
| 4 | Flow trajectories | **step 5** |
| — | Reverse flow | **step 5** |
