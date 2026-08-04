"""
Frozen pretrained encoders and their preprocessing.

The protocol is strict on two points, and both are enforced here rather than in the
notebooks so they cannot drift apart between steps:

1. **All encoder parameters stay frozen.** Every builder calls ``requires_grad_(False)``
   and ``eval()`` before returning. Nothing downstream ever trains a backbone.
2. **Use the preprocessing associated with each checkpoint.** For ResNet-18 that means
   ``weights.transforms()`` — the exact transform torchvision recorded with the weights —
   rather than a hand-written resize/crop/normalise that merely looks equivalent.

One dataset-specific addition
-----------------------------
Every FGVC-Aircraft image carries a 20-pixel copyright banner burned into the bottom
edge (measured and visualised in step 01). :class:`CropBottomBanner` removes it *before*
the checkpoint's own transform runs, so the banner never reaches the network. This is
cleaning the input image, not altering the checkpoint's preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

__all__ = [
    "AIRCRAFT_BANNER_PX",
    "CropBottomBanner",
    "EncoderSpec",
    "ENCODER_SPECS",
    "build_encoder",
    "preprocessing_for",
]

#: Height in pixels of the burned-in copyright banner on every FGVC-Aircraft image.
AIRCRAFT_BANNER_PX: int = 20


class CropBottomBanner:
    """Remove a fixed-height strip from the bottom of a PIL image.

    Defined at module level on purpose: Windows spawns DataLoader workers with the
    ``spawn`` start method, which pickles the transform. A class defined inside a function
    cannot be pickled, which is exactly why the original Colab notebook's ``num_workers=2``
    could not run outside Colab.

    Args:
        px: number of pixel rows to remove from the bottom edge.
    """

    def __init__(self, px: int = AIRCRAFT_BANNER_PX) -> None:
        self.px = px

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if self.px <= 0 or self.px >= height:
            return image
        return image.crop((0, 0, width, height - self.px))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(px={self.px})"


@dataclass(frozen=True)
class EncoderSpec:
    """Static description of one frozen encoder."""

    name: str
    display_name: str
    feature_dim: int
    checkpoint: str
    note: str
    datasets: tuple[str, ...] = field(default_factory=tuple)


#: The encoders this project uses, and which datasets each runs on.
ENCODER_SPECS: dict[str, EncoderSpec] = {
    "resnet18": EncoderSpec(
        name="resnet18",
        display_name="ResNet-18",
        feature_dim=512,
        checkpoint="torchvision IMAGENET1K_V1",
        note="512-d penultimate features (final fc replaced by Identity)",
        datasets=("DTD", "FGVC-Aircraft"),
    ),
    "dinov2": EncoderSpec(
        name="dinov2",
        display_name="DINOv2 ViT-S/14",
        feature_dim=384,
        checkpoint="facebookresearch/dinov2 : dinov2_vits14",
        note="384-d final CLS token",
        datasets=("FGVC-Aircraft",),
    ),
}


def _build_resnet18() -> tuple[nn.Module, Callable]:
    """ImageNet-1K ResNet-18 with the classifier head removed.

    Replacing ``fc`` with ``nn.Identity()`` makes the forward pass return the 512-d vector
    that used to feed the classifier — precisely the "512-dimensional representation before
    the final classification layer" this project needs.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    return model, weights.transforms()


def _build_dinov2() -> tuple[nn.Module, Callable]:
    """DINOv2 ViT-S/14 from the official Meta hub repo.

    Calling the model returns the final CLS-token embedding (384-d) directly.

    The preprocessing is DINOv2's published evaluation recipe: resize the shorter side to
    256 with bicubic interpolation, centre-crop 224, and normalise with the ImageNet
    statistics. DINOv2 is self-supervised and never saw ImageNet labels, but it was trained
    with these normalisation constants.
    """
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return model, transform


_BUILDERS: dict[str, Callable[[], tuple[nn.Module, Callable]]] = {
    "resnet18": _build_resnet18,
    "dinov2": _build_dinov2,
}


def build_encoder(name: str) -> tuple[nn.Module, Callable]:
    """Build a frozen encoder and its checkpoint preprocessing.

    The returned model is already in ``eval()`` mode with every parameter frozen.

    Args:
        name: ``"resnet18"`` or ``"dinov2"``.

    Returns:
        ``(model, base_transform)``. Use :func:`preprocessing_for` to get the
        dataset-adjusted transform rather than using ``base_transform`` directly.
    """
    if name not in _BUILDERS:
        raise KeyError(f"Unknown encoder {name!r}. Known: {sorted(_BUILDERS)}")

    model, base_transform = _BUILDERS[name]()
    model.requires_grad_(False)
    model.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 0, f"{name} has {trainable} trainable parameters - it must be frozen"

    return model, base_transform


def preprocessing_for(base_transform: Callable, dataset_name: str,
                      crop_banner: bool = True) -> Callable:
    """Adjust a checkpoint transform for one dataset.

    For FGVC-Aircraft this prepends :class:`CropBottomBanner`; for every other dataset the
    checkpoint transform is returned unchanged.

    Args:
        base_transform: the transform returned by :func:`build_encoder`.
        dataset_name: ``"DTD"`` or ``"FGVC-Aircraft"``.
        crop_banner: set ``False`` to reproduce the un-cropped behaviour, used by the
            banner ablation in step 02.
    """
    if dataset_name == "FGVC-Aircraft" and crop_banner:
        return transforms.Compose([CropBottomBanner(AIRCRAFT_BANNER_PX), base_transform])
    return base_transform
