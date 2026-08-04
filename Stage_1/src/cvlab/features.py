"""
Feature extraction and the on-disk feature cache.

The assignment requires the encoders to be run **exactly once** and the resulting features
cached, with all classifier training done on the cache. That is what makes the 27-run
linear-probe sweep in step 03 cheap, and it is what guarantees the linear probe and the
prototype baseline see byte-identical inputs — so any accuracy difference between them
comes from the classifier design rather than from one of them getting better features.

Every cache file is **self-describing**: alongside the tensors it stores the encoder,
dataset, split, whether the aircraft banner was cropped, the exact transform, and the
library versions used. A cache whose provenance is unknown is a cache you cannot trust.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .paths import FEATURES_ROOT

__all__ = [
    "FeatureBundle",
    "feature_path",
    "extract_features",
    "save_features",
    "load_features",
    "cache_exists",
]


@dataclass
class FeatureBundle:
    """Cached features for one (encoder, dataset, split) combination."""

    features: torch.Tensor          # (N, D) float32, on CPU
    labels: torch.Tensor            # (N,)   int64,   on CPU
    classes: list[str]              # length C, index i is the name of class i
    meta: dict                      # provenance, see save_features

    def __len__(self) -> int:
        return int(self.features.shape[0])

    @property
    def dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def __repr__(self) -> str:
        return (f"FeatureBundle({self.meta.get('encoder')}/{self.meta.get('dataset')}/"
                f"{self.meta.get('split')}: {len(self)}x{self.dim}, {self.n_classes} classes)")


def feature_path(encoder: str, dataset: str, split: str,
                 root: Path | None = None, suffix: str = "") -> Path:
    """Canonical cache path, e.g. ``features/resnet18__fgvc_aircraft__train.pt``.

    Args:
        suffix: optional tag for ablation caches, e.g. ``"nobannercrop"``.
    """
    slug = dataset.lower().replace("-", "_")
    tag = f"__{suffix}" if suffix else ""
    root = FEATURES_ROOT if root is None else Path(root)
    return root / f"{encoder}__{slug}__{split}{tag}.pt"


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    dataset: Dataset,
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 0,
    progress_every: float = 0.25,
    label: str = "",
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Run a frozen encoder over a whole dataset once.

    The dataset must already have the encoder's transform attached (pass it to
    ``load_split``), so no wrapper Dataset is needed — which conveniently sidesteps the
    Windows pickling problem that a function-local wrapper class would create.

    Args:
        model: a frozen encoder from :func:`cvlab.encoders.build_encoder`.
        dataset: dataset with the encoder's preprocessing already applied.
        device: ``"cuda"`` or ``"cpu"``.
        batch_size: 64 keeps peak VRAM well inside 4 GB for both encoders (measured peak
            0.46 GB for ResNet-18, 0.38 GB for DINOv2).
        num_workers: 4 is roughly 2x faster than 0 on this machine. Windows spawns workers
            with ``spawn``, which pickles the dataset and its transform — so this only
            works because every transform is defined at module level in this package. A
            transform defined inside a function would raise a pickling error here.
        progress_every: print a progress line each time this fraction of batches completes.
        label: prefix for the progress lines.

    Returns:
        ``(features, labels, elapsed_seconds)`` with both tensors on the CPU.
    """
    model.eval().to(device)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )

    n_batches = len(loader)
    step = max(1, int(n_batches * progress_every))
    chunks_f: list[torch.Tensor] = []
    chunks_y: list[torch.Tensor] = []

    start = time.perf_counter()
    for i, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        chunks_f.append(model(images).float().cpu())
        chunks_y.append(targets)
        if i % step == 0 or i == n_batches:
            done = i / n_batches
            rate = (i * loader.batch_size) / (time.perf_counter() - start)
            print(f"    {label}{done:5.0%}  batch {i}/{n_batches}  ~{rate:.0f} img/s")

    elapsed = time.perf_counter() - start
    return torch.cat(chunks_f), torch.cat(chunks_y), elapsed


def save_features(
    bundle_features: torch.Tensor,
    bundle_labels: torch.Tensor,
    classes: list[str],
    encoder: str,
    dataset: str,
    split: str,
    transform,
    crop_banner: bool,
    device: str,
    elapsed_s: float,
    root: Path | None = None,
    suffix: str = "",
) -> Path:
    """Write one feature cache file, with full provenance metadata."""
    import torchvision

    path = feature_path(encoder, dataset, split, root=root, suffix=suffix)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "encoder": encoder,
        "dataset": dataset,
        "split": split,
        "n_samples": int(bundle_features.shape[0]),
        "feature_dim": int(bundle_features.shape[1]),
        "n_classes": len(classes),
        "crop_banner": bool(crop_banner),
        "transform": repr(transform),
        "device": device,
        "elapsed_s": round(float(elapsed_s), 2),
        "images_per_s": round(float(bundle_features.shape[0]) / max(elapsed_s, 1e-9), 1),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    torch.save(
        {"features": bundle_features, "labels": bundle_labels, "classes": classes, "meta": meta},
        path,
    )
    return path


def load_features(encoder: str, dataset: str, split: str,
                  root: Path | None = None, suffix: str = "") -> FeatureBundle:
    """Read one feature cache file.

    ``weights_only=False`` is required because the payload contains the class-name list
    and the metadata dict, not just tensors. The file is one we wrote ourselves.
    """
    path = feature_path(encoder, dataset, split, root=root, suffix=suffix)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached features at {path}. Run notebook 02 (feature extraction) first."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return FeatureBundle(
        features=payload["features"],
        labels=payload["labels"],
        classes=payload["classes"],
        meta=payload.get("meta", {}),
    )


def cache_exists(encoder: str, dataset: str, split: str,
                 root: Path | None = None, suffix: str = "") -> bool:
    """Whether a given cache file is already on disk."""
    return feature_path(encoder, dataset, split, root=root, suffix=suffix).exists()
