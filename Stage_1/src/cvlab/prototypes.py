"""
Image-derived class prototypes — Stage 1's second baseline.

First L2-normalize every feature, then for each
class average its (normalized) training examples and re-normalize::

    mu_c = normalize( mean_{i in S_c} normalize(z_i) )

Classification is nearest-prototype by cosine similarity::

    y_hat = argmax_c cos(z, mu_c)

No parameters are trained. This is a nearest-centroid rule in a normalized (cosine) space,
which is what makes it directly comparable to the linear probe: same cached features, same
k-shot subsets (via :func:`cvlab.data.make_kshot_subset`), same evaluation protocol — the
only thing that differs is whether a decision boundary is *learned* (linear probe) or
*averaged* (prototypes).

Why L2-normalize before averaging
----------------------------------
Without normalizing first, an image with a larger raw feature magnitude would pull the
class average toward itself for a reason unrelated to its class identity. Step 2 measured
exactly this risk: DTD/ResNet-18 features have a mean norm of ~23.6 with a std of ~5,
FGVC-Aircraft/DINOv2 features are ~49.7 with std ~0.8 — normalizing first means every
example contributes equally in *direction*, and cosine similarity then only cares about
that direction.

Once both sides are unit-length, cosine similarity is a plain dot product — no separate
similarity formula is needed, it falls out of the normalization for free.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

__all__ = ["PrototypeResult", "l2_normalize", "compute_prototypes", "run_prototype_classifier"]


def l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize along `dim`. Thin wrapper over `F.normalize` kept for the protocol's
    own notation (`normalize(z)`), so the code and the formula above read the
    same way."""
    return F.normalize(x, p=2, dim=dim, eps=eps)


def compute_prototypes(features: torch.Tensor, labels: torch.Tensor,
                       num_classes: int) -> torch.Tensor:
    """Build one prototype per class: mu_c = normalize(mean_{i in S_c} normalize(z_i)).

    Args:
        features: training features for the selected subset, shape (N, D).
        labels: corresponding class indices, shape (N,).
        num_classes: total number of classes (prototypes are still produced, as zero
            vectors, for classes absent from this particular subset — see the assertion
            below, which should never fire for the k-shot subsets this project uses since
            every class is guaranteed >= K examples).

    Returns:
        Prototype matrix, shape (num_classes, D), each row unit-norm.
    """
    normed = l2_normalize(features)
    prototypes = torch.zeros(num_classes, features.shape[1], dtype=features.dtype)

    present = torch.unique(labels)
    for c in present.tolist():
        prototypes[c] = normed[labels == c].mean(dim=0)

    missing = num_classes - len(present)
    assert missing == 0, (
        f"{missing} class(es) have zero examples in this subset - the k-shot sampler "
        "should guarantee every class is represented"
    )

    return l2_normalize(prototypes)


@dataclass
class PrototypeResult:
    """Everything one prototype run produces. Mirrors `cvlab.probe.ProbeResult` so step 5
    can treat both baselines uniformly."""

    test_acc: float
    n_train: int
    elapsed_s: float
    prototypes: torch.Tensor        # (num_classes, D), unit-norm
    test_preds: torch.Tensor        # (N_test,) int64, on CPU


def run_prototype_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    num_classes: int,
) -> PrototypeResult:
    """Build prototypes from the training subset and classify the test split once.

    No hyperparameters, no randomness beyond whatever subset was already selected — this
    function is deterministic given its inputs, which is why only one run is needed for the
    full-data setting.
    """
    start = time.perf_counter()

    prototypes = compute_prototypes(train_x, train_y, num_classes)
    test_normed = l2_normalize(test_x)
    similarities = test_normed @ prototypes.T          # cosine similarity, unit-norm both sides
    preds = similarities.argmax(dim=1)
    acc = (preds == test_y).float().mean().item()

    return PrototypeResult(
        test_acc=acc,
        n_train=train_x.shape[0],
        elapsed_s=time.perf_counter() - start,
        prototypes=prototypes,
        test_preds=preds,
    )
