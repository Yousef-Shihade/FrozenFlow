"""Flow Matching velocity model and standard Euler transport."""

from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "VelocityNetwork",
    "flow_matching_loss",
    "rolled_out_loss",
    "euler_inference",
    "euler_inference_trajectory",
]


class VelocityNetwork(nn.Module):
    """Predict a feature-space velocity from a feature vector and time."""

    def __init__(self, feature_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.network = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, features: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return velocities for ``features`` at times ``t``."""
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must have shape (N, {self.feature_dim}), got {tuple(features.shape)}"
            )

        if t.ndim == 0:
            t = t.expand(features.shape[0])
        if t.ndim == 1:
            t = t.unsqueeze(1)
        if t.shape != (features.shape[0], 1):
            raise ValueError(f"t must have shape (N,) or (N, 1), got {tuple(t.shape)}")

        return self.network(torch.cat((features, t.to(features)), dim=1))


def flow_matching_loss(
    model: VelocityNetwork,
    source_features: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute the standard Flow Matching loss for prototype transport.

    For each source ``z_i``, the target endpoint is its labeled prototype. A time
    ``t ~ U(0, 1)`` is sampled independently per example, then
    ``z_t = (1 - t) z_i + t prototype`` and the target velocity is ``prototype - z_i``.
    """
    if source_features.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("source_features and prototypes must both be rank-2 tensors")
    if source_features.shape[1] != prototypes.shape[1]:
        raise ValueError("source_features and prototypes must have the same feature dimension")
    if labels.ndim != 1 or labels.shape[0] != source_features.shape[0]:
        raise ValueError("labels must have one entry per source feature")

    labels = labels.to(device=source_features.device, dtype=torch.long)
    target_prototypes = prototypes.to(source_features)[labels]
    t = torch.rand(source_features.shape[0], device=source_features.device)
    z_t = (1.0 - t[:, None]) * source_features + t[:, None] * target_prototypes
    target_velocity = target_prototypes - source_features
    predicted_velocity = model(z_t, t)
    return torch.mean((predicted_velocity - target_velocity) ** 2)


def rolled_out_loss(
    model: VelocityNetwork,
    source_features: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Rolled-out Flow Matching loss (Stage 2 spec, ``L_roll``).

    Run the full ``steps``-step Euler transport used at inference, starting from
    each ``source_features`` row, then penalise the squared distance of the final
    transported state to that row's labeled prototype::

        z_hat_0 = z_i
        z_hat_{k+1} = z_hat_k + (1/steps) * v_theta(z_hat_k, k/steps)
        L_roll = mean( (z_hat_steps - prototype[label])**2 )

    The autograd graph is kept through the whole chain of ``steps`` velocity
    predictions, so ``.backward()`` optimises the endpoint directly rather than
    the individual velocities. ``steps`` must match the value used at inference.

    The reduction is ``torch.mean`` over batch *and* feature dim, matching
    :func:`flow_matching_loss`, so the two objectives stay on the same scale for
    a fair standard-vs-rolled-out comparison (spec: keep the main training
    choices fixed).
    """
    if source_features.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("source_features and prototypes must both be rank-2 tensors")
    if source_features.shape[1] != prototypes.shape[1]:
        raise ValueError("source_features and prototypes must have the same feature dimension")
    if labels.ndim != 1 or labels.shape[0] != source_features.shape[0]:
        raise ValueError("labels must have one entry per source feature")
    if steps < 1:
        raise ValueError("steps must be at least 1")

    labels = labels.to(device=source_features.device, dtype=torch.long)
    target_prototypes = prototypes.to(source_features)[labels]

    transported = source_features
    dt = 1.0 / steps
    for step in range(steps):
        t = torch.full(
            (source_features.shape[0],), step * dt,
            device=source_features.device,
            dtype=source_features.dtype,
        )
        transported = transported + dt * model(transported, t)
    return torch.mean((transported - target_prototypes) ** 2)


@torch.no_grad()
def euler_inference(
    model: VelocityNetwork,
    starting_features: torch.Tensor,
    steps: int = 100,
) -> torch.Tensor:
    """Transport features from ``t=0`` to ``t=1`` with ``steps`` Euler updates."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if starting_features.ndim != 2:
        raise ValueError("starting_features must have shape (N, D)")

    was_training = model.training
    model.eval()
    transported = starting_features.clone()
    dt = 1.0 / steps
    for step in range(steps):
        t = torch.full(
            (transported.shape[0],), step * dt,
            device=transported.device,
            dtype=transported.dtype,
        )
        transported = transported + dt * model(transported, t)
    model.train(was_training)
    return transported


@torch.no_grad()
def euler_inference_trajectory(
    model: VelocityNetwork,
    starting_features: torch.Tensor,
    steps: int = 100,
) -> torch.Tensor:
    """Transport features from ``t=0`` to ``t=1``, keeping every intermediate state.

    Same Euler stepping as :func:`euler_inference`, but each state is recorded
    instead of overwritten. Returns a tensor of shape ``(steps + 1, N, D)``:

    * ``trajectory[0]``  is ``starting_features``, unchanged;
    * ``trajectory[-1]`` equals :func:`euler_inference`'s return value for the
      same ``model`` and ``steps``;
    * the rows between are the intermediate Euler states — the connected path an
      example traces through feature space.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if starting_features.ndim != 2:
        raise ValueError("starting_features must have shape (N, D)")

    was_training = model.training
    model.eval()
    transported = starting_features.clone()
    states = [transported.clone()]
    dt = 1.0 / steps
    for step in range(steps):
        t = torch.full(
            (transported.shape[0],), step * dt,
            device=transported.device,
            dtype=transported.dtype,
        )
        transported = transported + dt * model(transported, t)
        states.append(transported)
    model.train(was_training)
    return torch.stack(states, dim=0)