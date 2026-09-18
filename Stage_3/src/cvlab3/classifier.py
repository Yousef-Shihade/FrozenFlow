"""
The frozen linear classifier, and the Stage 3 protocol constants.

Stage 3 inserts a flow-matching transformation *in front of* a linear classifier that has
already been trained (Stage 1) and is then held fixed::

    z --[ FM, T Euler steps ]--> z_hat --[ frozen W, b ]--> logits

Two things in this module carry the weight of that sentence:

* :class:`FrozenClassifier` makes "frozen" mechanical rather than a promise. Its parameters
  have ``requires_grad=False`` and it is never handed to an optimiser, so a gradient cannot
  reach the classifier even by accident. Stage 3's whole comparison against the Stage 1
  linear probe depends on ``W`` and ``b`` being *the same numbers* Stage 1 reported.
* :func:`identity_flow` builds a velocity network whose output layer is zeroed, so the
  rollout is the identity map and the untrained Stage 3 system reproduces the Stage 1
  probe **exactly**, not approximately. Step 01 asserts this.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cvlabfm.flow import VelocityNet, euler_rollout

__all__ = [
    "T_STEPS",
    "K_SHOT",
    "SEEDS",
    "COMBOS",
    "MAIN_COMBOS",
    "FrozenClassifier",
    "identity_flow",
    "transform",
    "transform_and_classify",
]


# --- Stage 3 protocol ------------------------------------------------------

#: Euler steps, fixed for all of Stage 3. Stage 2 measured T = 4 against T = 12 across its
#: whole grid and found them within 1 pp of each other with inconsistent sign, so the
#: cheaper one is chosen -
#: and it is meaningfully cheaper here, because Stage 3's end-to-end objective
#: backpropagates through every step.
T_STEPS: int = 4

#: Training-set size, fixed for all of Stage 3.
K_SHOT: int = 10

#: Unchanged from Stages 1 and 2, so the k-shot subsets are literally the same images.
SEEDS: tuple[int, ...] = (0, 1, 2)

#: The two main combinations - one representative encoder per dataset.
MAIN_COMBOS: tuple[tuple[str, str], ...] = (
    ("resnet18", "DTD"),
    ("dinov2", "FGVC-Aircraft"),
)

#: Every combination actually run. ResNet-18 on Aircraft goes beyond the two main
#: combinations; it is included because it is the one combination where Stage 1's features
#: are known to be
#: badly entangled (own-vs-other cosine margin 0.041), which makes it the most informative
#: test of whether a flow can rescue a representation a linear classifier struggles with.
COMBOS: tuple[tuple[str, str], ...] = MAIN_COMBOS + (("resnet18", "FGVC-Aircraft"),)


# --- The frozen classifier -------------------------------------------------

@dataclass(frozen=True)
class _ClassifierMeta:
    """Provenance of one frozen classifier, carried alongside its weights."""

    encoder: str
    dataset: str
    k: int
    seed: int
    stage1_test_acc: float


class FrozenClassifier(nn.Module):
    """Stage 1's trained linear probe, reloaded and locked.

    Wraps ``s = W z + b`` where ``W`` and ``b`` come from a Stage 1 probe run. Every
    parameter has ``requires_grad=False`` from construction, and the module is put in
    ``eval()`` mode, so it contributes no gradient and no training-mode behaviour.

    Args:
        weight: ``(num_classes, dim)``.
        bias: ``(num_classes,)``.
        meta: optional provenance record (which Stage 1 run these weights came from).
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor,
                 meta: _ClassifierMeta | None = None) -> None:
        super().__init__()
        if weight.dim() != 2:
            raise ValueError(f"weight must be 2-D (num_classes, dim), got {tuple(weight.shape)}")
        if bias.shape != (weight.shape[0],):
            raise ValueError(
                f"bias must have shape ({weight.shape[0]},) to match weight, "
                f"got {tuple(bias.shape)}"
            )

        self.linear = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)

        for p in self.linear.parameters():
            p.requires_grad_(False)
        self.eval()

        self.meta = meta

    @property
    def num_classes(self) -> int:
        return self.linear.out_features

    @property
    def dim(self) -> int:
        return self.linear.in_features

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor],
                        meta: _ClassifierMeta | None = None) -> "FrozenClassifier":
        """Build from a ``nn.Linear`` state dict, i.e. what a Stage 1 probe run returns."""
        return cls(state["weight"], state["bias"], meta=meta)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Logits for features ``z`` of shape ``(N, D)``.

        Not wrapped in ``no_grad``: Stage 3's end-to-end objective has to differentiate the
        classification loss *through* this layer to reach the flow behind it. The
        classifier's own parameters stay fixed because they carry ``requires_grad=False``,
        which is a separate thing from whether gradient flows through the op.
        """
        return self.linear(z)

    def train(self, mode: bool = True) -> "FrozenClassifier":
        """Ignore requests to enter training mode - this module is frozen by contract.

        Calling ``.train()`` on a parent module would otherwise silently flip this one back,
        which is exactly the kind of accident the class exists to prevent.
        """
        return super().train(False)

    def extra_repr(self) -> str:
        base = f"dim={self.dim}, num_classes={self.num_classes}, frozen=True"
        if self.meta is None:
            return base
        return (f"{base}, from={self.meta.encoder}/{self.meta.dataset} "
                f"K={self.meta.k} seed={self.meta.seed}")


# --- The flow in front of it -----------------------------------------------

def identity_flow(dim: int, seed: int, hidden_dim: int = 512,
                  n_hidden: int = 2, device: str = "cuda") -> VelocityNet:
    """A velocity network that starts as the identity transformation.

    The output layer is zeroed, so ``v(z, t) = 0`` for every input and the Euler rollout
    returns ``z`` unchanged - bit-for-bit, not approximately. The Stage 3 system therefore
    *is* the Stage 1 linear probe before training starts, which is what makes ``delta``
    against the probe baseline meaningful.

    Only the readout is zeroed; the hidden layers keep the standard random initialisation.
    Note what that implies for the first optimiser step: a hidden layer's gradient is
    proportional to the readout weight sitting above it, so at initialisation the hidden
    layers receive **exactly zero** gradient and only the readout moves. From the second step
    onwards the readout is non-zero and the whole network trains normally. (Measured: hidden
    grad 0.0 at step 0, then 3.7e-03, 7.7e-03 at steps 1 and 2, with the loss falling
    throughout.) This is the usual behaviour of a zero-initialised readout, not a defect - it
    is the price of starting exactly at identity rather than near it.

    Stage 2's :class:`~cvlabfm.flow.VelocityNet` is constructed **unmodified** and the
    readout is zeroed here afterwards. Stage 2 is finished and published, so it is not edited
    to add an initialisation option that only Stage 3 needs. The cost is that this function
    reaches into ``net.net[-1]``, which relies on Stage 2's layer ordering - so the result is
    asserted rather than assumed: if that structure ever changes, this raises immediately
    instead of silently returning a flow that moves features.

    Args:
        dim: feature dimensionality (512 for ResNet-18, 384 for DINOv2).
        seed: controls the hidden layers' initialisation.
        hidden_dim: width of each hidden layer. Stage 2's value, unchanged.
        n_hidden: number of hidden layers. Stage 2's value, unchanged.
        device: ``"cuda"`` or ``"cpu"``.

    Raises:
        RuntimeError: if zeroing the readout does not actually produce ``v = 0``.
    """
    torch.manual_seed(seed)
    net = VelocityNet(dim, hidden_dim=hidden_dim, n_hidden=n_hidden)

    readout = net.net[-1]
    if not isinstance(readout, nn.Linear) or readout.out_features != dim:
        raise RuntimeError(
            "expected the last module of VelocityNet.net to be the Linear readout with "
            f"out_features={dim}, found {readout!r}. Stage 2's architecture changed; "
            "identity_flow needs updating."
        )
    with torch.no_grad():
        readout.weight.zero_()
        readout.bias.zero_()

    net = net.to(device)

    # The property the whole stage depends on, checked rather than trusted.
    with torch.no_grad():
        probe_z = torch.randn(8, dim, device=device)
        v = net(probe_z, torch.rand(8, device=device))
        if bool(v.abs().max() != 0):
            raise RuntimeError(
                f"identity initialisation failed: max |v| = {v.abs().max().item():.3e}, "
                "expected exactly 0"
            )

    return net


def transform(net: VelocityNet, z: torch.Tensor, T: int = T_STEPS) -> torch.Tensor:
    """Apply the flow: ``T`` Euler steps from ``z``, returning ``z_hat``.

    Thin alias for :func:`cvlabfm.flow.euler_rollout` so Stage 3 reads in its own terms
    while provably running Stage 2's integrator - the same code, not a copy of it.
    """
    return euler_rollout(net, z, T)


def transform_and_classify(net: VelocityNet, clf: FrozenClassifier,
                           z: torch.Tensor, T: int = T_STEPS) -> torch.Tensor:
    """The complete Stage 3 system: ``z -> flow -> z_hat -> frozen classifier -> logits``.

    Differentiable end to end with respect to the flow's parameters, which is what
    Strategy 1 trains through.
    """
    return clf(transform(net, z, T))
