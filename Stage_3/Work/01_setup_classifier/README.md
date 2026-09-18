# Step 01 — Setup and the frozen classifier

Stage 3 puts a flow-matching transformation in front of a linear classifier that is already
trained and then held fixed:

$$z \;\xrightarrow{\ \text{FM},\ T \text{ Euler steps}\ }\; \hat z \;\xrightarrow{\ \text{frozen } W,\,b\ }\; s$$

No flow is trained here. This step establishes the four things everything downstream stands
on, and **measures** each one rather than assuming it.

---

## 1 · The Stage 1 classifier had to be recovered

Stage 1 saved accuracies, training curves and test predictions — but **not the probe's
weights**. `train_linear_probe` let the trained layer go out of scope, because Stage 1 only
ever reported numbers and Stage 2 used prototypes instead. Stage 3 needs the classifier
itself.

**Stages 1 and 2 are finished and published, so neither was modified.** Stage 3 carries its
own `cvlab3.probe.train_frozen_classifier`, which reproduces Stage 1's loop and additionally
returns the weights. The hyperparameters are *not* duplicated — `ProbeConfig` is imported
from Stage 1, so the learning rate, weight decay, batch size, epoch budget and checkpoint
rule keep exactly one definition in the project.

Duplicated logic is normally a liability in this project. Two things stop it being one here:

- this step asserts the re-derived probes reproduce Stage 1's published numbers exactly;
- `Stage_3/tests/test_probe_equivalence.py` runs both implementations against each other and
  asserts they agree on accuracy, predictions, selected epoch, and **every** training curve.

Drift cannot be silent. This step re-trains all nine probes (3 combinations × 3 seeds,
K = 10), and they reproduce Stage 1 exactly:

| Check | Result |
| --- | --- |
| Test accuracy vs. Stage 1's published runs | **max \|Δ\| = 0.00e+00 pp** |
| Test predictions, per image | **identical** — 0 of 16,920 labels differ |

| Encoder / dataset | K = 10 accuracy | Matches Stage 1 |
| --- | --- | --- |
| ResNet-18 / DTD | 52.04 ± 1.16 | yes |
| DINOv2 / FGVC-Aircraft | 51.54 ± 1.70 | yes |
| ResNet-18 / FGVC-Aircraft | 27.87 ± 1.33 | yes |

That matters because every number Stage 3 reports is a *difference* against these. If the
frozen classifier were not the classifier Stage 1 published, every Δ downstream would be
measured against the wrong baseline.

---

## 2 · "Frozen" is mechanical, not a promise

`FrozenClassifier` constructs its parameters with `requires_grad=False` and overrides
`.train()` to ignore requests to leave eval mode — so a parent module calling `.train()` on
the whole system cannot silently unfreeze it. Verified in the notebook: 0 parameters require
grad, the module stays in eval mode after `.train(True)`, and a backward pass reaches the
*input* while leaving `W.grad` as `None`.

Gradient flowing *through* the layer is deliberate and necessary — Strategy 1 differentiates
the classification loss back into the flow behind it. That is a different thing from the
classifier's own parameters being updated.

---

## 3 · The flow starts at identity — exactly

We wanted an FM initialised *close to* identity. Zeroing the velocity network's
output layer gives something stronger: $v(z,t) = 0$ everywhere, so each Euler update adds
exactly zero and $\hat z = z$ **bit-for-bit**.

| Check | Result |
| --- | --- |
| `max` \|$\hat z - z$\| over all test features | **0.0e+00** |
| Accuracy at init vs. the Stage 1 probe | **Δ = 0.0e+00 pp** |
| Predictions at init | identical |

Stage 2's `VelocityNet` is constructed **unmodified** and the readout is zeroed afterwards,
so Stage 2 needed no change either. `identity_flow` asserts the result rather than assuming
it, since zeroing by index relies on Stage 2's layer ordering.

One consequence is worth stating, because it is easy to assume the opposite. A hidden layer's
gradient is proportional to the readout weight above it, so **while the readout is exactly
zero the hidden layers receive exactly zero gradient** — on the first optimiser step only the
readout moves. From the second step onwards the readout is non-zero and the network trains
normally:

| optimiser step | hidden-layer \|grad\| | loss |
| --- | --- | --- |
| 0 | **0.0** | 6.75 |
| 1 | 3.7e-03 | 6.53 |
| 2 | 7.7e-03 | 6.18 |

That is the standard behaviour of a zero-initialised readout, and it is the price of starting
*exactly* at identity rather than near it. Because the untrained system **is** the probe, any
later gain or loss is attributable to training and nothing else.

---

## 4 · Stage 3 does **not** L2-normalize — and here is what that is worth

This is the one design decision Stage 3 inherits differently from Stage 2, so it is measured
rather than argued.

**Stage 2 normalized** because its targets were class prototypes, which are unit-norm by
construction; it showed raw features cost 15.8–20.3 pp there. **Stage 3 must not**, because
its frozen classifier was *trained on raw features* at a scale of 24–50.

Feeding the frozen classifier L2-normalized features instead:

| Encoder / dataset | mean ‖z‖ | raw | L2-normalized | cost |
| --- | --- | --- | --- | --- |
| ResNet-18 / DTD | 23.7 | 52.04% | 47.55% | **−4.49 pp** |
| DINOv2 / FGVC-Aircraft | 49.7 | 51.54% | 46.19% | **−5.34 pp** |
| ResNet-18 / FGVC-Aircraft | 28.9 | 27.87% | 24.44% | **−3.43 pp** |

Not a collapse — normalizing preserves each feature's *direction*, where most of the class
information lives — but a free loss of 3.4–5.3 pp on every combination.

**The DINOv2 row is the sharpest version of the argument.** Its norms are already tightly
clustered (46.9–52.0), so normalizing is close to dividing every feature by the same
constant — and it still costs the most of the three. A near-uniform rescale is not harmless
to an affine classifier: $W(z/\lVert z\rVert) + b$ rescales $Wz$ but leaves $b$ untouched, so
the decision boundary moves relative to the data. Unlike Stage 2's cosine rule, this
classifier is **not** scale-invariant.

This is also why the identity initialisation is load-bearing rather than book-keeping: a
flow initialised the ordinary way would displace features from the first forward pass, and
the classifier in front of it has no slack for that.

---

## Outputs

| File | What it holds |
| --- | --- |
| `../../results/frozen_classifiers.pt` | the 9 frozen classifiers, their curves, the protocol, and the verification record |
| `tables/stage1_probe_reproduction.csv` | per-run comparison against Stage 1 (Δ and prediction match) |
| `tables/frozen_classifier_accuracy.csv` | mean ± std per combination |
| `tables/identity_initialisation_check.csv` | per-run identity-init verification |
| `tables/l2_normalization_ablation.csv` | per-run raw vs. normalized |
| `tables/l2_normalization_summary.csv` | the table above |
| `plots/setup_verification.png` | Stage 1 reproduction, and identity init |
| `plots/why_no_normalization.png` | feature-norm distributions, and the cost of normalizing |

## Protocol fixed here for all of Stage 3

- **T = 4** Euler steps, fixed throughout. Stage 2 measured T = 4 against T = 12 across its
  whole grid and found them within 1 pp with inconsistent sign, so we use the cheaper one —
  and it is meaningfully cheaper here,
  because Strategy 1 backpropagates through every step.
- **K = 10**, seeds **{0, 1, 2}**, the same k-shot subsets as Stages 1 and 2.
- **Raw features**, no normalization (section 4).
- Velocity network: Stage 2's design unchanged (2 × 512 hidden, SiLU, `t` concatenated),
  with the output layer zeroed.

**Next:** step 02 trains the end-to-end rolled-out classification objective.
