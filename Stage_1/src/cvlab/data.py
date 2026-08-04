"""
Dataset loading and protocol constants.

Two jobs:

1. **Load the official splits under the exact protocol this project follows** — DTD
   partition 1, FGVC-Aircraft at the ``variant`` annotation level, ``download=False`` so
   the notebooks are offline and can never re-download over a partial extraction.

2. **Hide torchvision's private attributes behind a stable interface.** The per-image
   paths and labels of a torchvision dataset live in ``_image_files`` and ``_labels``,
   which are private and can be renamed without notice. Reaching into them from five
   notebooks means five things to fix if that happens; :func:`image_files` and
   :func:`labels` make it one.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import DTD, FGVCAircraft

from .paths import AIRCRAFT_ROOT, DTD_ROOT

__all__ = [
    "SPLITS",
    "DATASET_SPECS",
    "load_split",
    "load_datasets",
    "image_files",
    "labels",
    "class_counts",
    "missing_files",
    "make_kshot_subset",
]

#: The three official splits, in canonical order. Never merged.
SPLITS: tuple[str, str, str] = ("train", "val", "test")

#: Expected shape of each dataset, from the protocol and the official dataset papers.
#: Used by notebook 01 to turn "it loaded" into "it loaded correctly".
DATASET_SPECS: dict[str, dict] = {
    "DTD": {
        "n_classes": 47,
        "split_sizes": {"train": 1880, "val": 1880, "test": 1880},
        "n_images": 5640,
        "note": "official partition 1",
    },
    "FGVC-Aircraft": {
        "n_classes": 100,
        "split_sizes": {"train": 3334, "val": 3333, "test": 3333},
        "n_images": 10000,
        "note": "variant annotation level",
    },
}


def load_split(dataset_name: str, split: str, transform=None) -> Dataset:
    """Load a single official split with the mandated protocol arguments applied.

    Preferred over :func:`load_datasets` when different datasets need different
    preprocessing — which is the case from step 02 onward, since FGVC-Aircraft gets the
    copyright banner cropped and DTD does not.

    Args:
        dataset_name: ``"DTD"`` or ``"FGVC-Aircraft"``.
        split: one of :data:`SPLITS`.
        transform: torchvision transform applied to every image.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    if dataset_name == "DTD":
        return DTD(root=str(DTD_ROOT), split=split, partition=1,
                   download=False, transform=transform)
    if dataset_name == "FGVC-Aircraft":
        return FGVCAircraft(root=str(AIRCRAFT_ROOT), split=split,
                            annotation_level="variant", download=False,
                            transform=transform)
    raise KeyError(f"Unknown dataset {dataset_name!r}. Known: {sorted(DATASET_SPECS)}")


def load_datasets(transform=None) -> dict[str, dict[str, Dataset]]:
    """Load all six official splits: ``{dataset_name: {split: Dataset}}``.

    ``download=False`` is explicit rather than relying on the default: it guarantees the
    notebooks run offline and can never silently re-download on top of the local copy.

    Args:
        transform: optional torchvision transform applied to every image. Left as ``None``
            in notebook 01 (which inspects raw PIL images) and set to the encoder's own
            preprocessing in notebook 02.

    Returns:
        Nested dict keyed by dataset name then split name.
    """
    return {
        "DTD": {
            split: DTD(root=str(DTD_ROOT), split=split, partition=1,
                       download=False, transform=transform)
            for split in SPLITS
        },
        "FGVC-Aircraft": {
            split: FGVCAircraft(root=str(AIRCRAFT_ROOT), split=split,
                                annotation_level="variant", download=False,
                                transform=transform)
            for split in SPLITS
        },
    }


def image_files(dataset: Dataset) -> list[str]:
    """Absolute path of every image in ``dataset``, as strings.

    Wraps torchvision's private ``_image_files``, which is a ``list[Path]`` for DTD but a
    ``list[str]`` for FGVCAircraft. Normalising to ``str`` here means callers can compare
    across datasets (e.g. the split-disjointness check) without worrying about the type.
    """
    return [str(p) for p in dataset._image_files]  # noqa: SLF001


def labels(dataset: Dataset) -> np.ndarray:
    """Integer class index of every image in ``dataset``, as a numpy array."""
    return np.asarray(dataset._labels)  # noqa: SLF001


def class_counts(dataset: Dataset) -> Counter:
    """Number of images per class index."""
    return Counter(labels(dataset).tolist())


def make_kshot_subset(
    features,
    label_tensor,
    k: int,
    seed: int,
) -> tuple:
    """Sample a *balanced* K-images-per-class subset, reproducibly.

    Used by both the linear probe (step 03) and the prototype baseline (step 04). Sharing
    one implementation is what makes those two baselines directly comparable: for a given
    ``(k, seed)`` they train on exactly the same images, so any accuracy difference comes
    from the classifier design rather than from a luckier draw.

    Sampling is without replacement and forces exactly ``k`` per class, which is what
    "balanced subsets" means here.

    Args:
        features: feature tensor, shape ``(N, D)``.
        label_tensor: label tensor, shape ``(N,)``.
        k: images to take from each class.
        seed: subset seed. Used with {0, 1, 2} throughout this project.

    Returns:
        ``(subset_features, subset_labels)``.

    Raises:
        ValueError: if any class has fewer than ``k`` images.
    """
    rng = np.random.RandomState(seed)
    labels_np = label_tensor.numpy()

    selected: list[int] = []
    for class_idx in np.unique(labels_np):
        in_class = np.where(labels_np == class_idx)[0]
        if len(in_class) < k:
            raise ValueError(
                f"class {class_idx} has only {len(in_class)} images, need {k}"
            )
        selected.extend(rng.choice(in_class, size=k, replace=False).tolist())

    index = np.array(selected)
    return features[index], label_tensor[index]


def missing_files(dataset: Dataset) -> list[str]:
    """Images referenced by the split file but absent from disk.

    A non-empty result means a truncated or partial extraction — the failure mode that
    otherwise surfaces as a crash halfway through feature extraction.
    """
    return [p for p in image_files(dataset) if not Path(p).exists()]
