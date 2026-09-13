"""
Strategy 1: end-to-end rolled-out classification training.

For each training feature :math:`z`, run the complete :math:`T`-step Euler rollout to get
:math:`\\hat z`, push it through the **frozen** classifier, and minimise

.. math::  \\mathcal{L}_\\mathrm{cls} = \\mathrm{CE}(W \\hat z + b,\\; y)

backpropagating through the whole rollout and updating only the flow's parameters.

Optional regularisation
-----------------------
The brief invites penalising "unnecessarily large changes to the representation". Two forms
are implemented, both off by default:

* ``displacement_weight`` penalises :math:`\\lVert \\hat z - z \\rVert^2` - how far the
  feature ended up from where it started.
* ``velocity_weight`` penalises the mean :math:`\\lVert v_k \\rVert^2` over the rollout - how
  hard the flow pushed at each step.

They are not the same thing: a flow can take a long, wandering path and still land close to
where it began, which Stage 2 saw happen. Velocities are recovered from the trajectory
(:math:`v_k = T (\\hat z_{k+1} - \\hat z_k)`) rather than by re-running the network, so the
penalty is measured on exactly the states the loss is computed from.

Why validation-based checkpointing here, when Stage 2 deliberately avoided it
----------------------------------------------------------------------------
Stage 2 used a fixed epoch budget because its two objectives were on different scales and
measured different things (velocity error vs. endpoint distance), so no single validation
quantity could select fairly between them.

Stage 3 is the opposite case. Both strategies ultimately produce a classifier, and validation
**accuracy** is directly comparable between them - it is also the exact rule Stage 1's probe
used, which is the baseline being compared against. Selecting on it is therefore both fair
and consistent with the thing Stage 3 is measured against.

It is also necessary. The velocity network has roughly 660k-790k parameters and trains on
470-1000 features, about 30x the parameter count of the classifier it is trying to help. A
fixed budget would report an overfitted final epoch rather than the model the method can
actually deliver.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from cvlabfm.flow import euler_rollout

from cvlab3.classifier import T_STEPS, FrozenClassifier, identity_flow

__all__ = ["EndToEndConfig", "EndToEndResult", "rollout_with_velocities",
           "train_end_to_end"]


@dataclass(frozen=True)
class EndToEndConfig:
    """Settings for Strategy 1.

    The optimiser settings mirror Stage 1's ``ProbeConfig`` and Stage 2's ``FlowConfig``
    where they overlap, rather than introducing new numbers that would need their own
    justification. The architecture is Stage 2's, unchanged and untuned.
    """

    hidden_dim: int = 512
    n_hidden: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200
    optimizer: str = "AdamW"
    #: Stage 2 found that backpropagating through T composed Euler steps can amplify
    #: gradients enough to diverge. The same risk applies here, so the same clip is used.
    max_grad_norm: float = 1.0
    #: Weight on ||z_hat - z||^2 (mean over examples, summed over dimensions).
    displacement_weight: float = 0.0
    #: Weight on the mean ||v_k||^2 over the T rollout steps.
    velocity_weight: float = 0.0
    checkpoint_metric: str = "val_accuracy"

    def as_dict(self) -> dict:
        return {
            "hidden_dim": self.hidden_dim,
            "n_hidden": self.n_hidden,
            "activation": "SiLU",
            "optimizer": self.optimizer,
            "learning_rate": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "max_grad_norm": self.max_grad_norm,
            "displacement_weight": self.displacement_weight,
            "velocity_weight": self.velocity_weight,
            "checkpoint_selection": self.checkpoint_metric,
        }

    @property
    def is_regularized(self) -> bool:
        return self.displacement_weight > 0 or self.velocity_weight > 0


@dataclass
class EndToEndResult:
    """Everything one Strategy 1 run produces."""

    test_acc: float
    best_val_acc: float
    best_epoch: int
    baseline_acc: float                     # the frozen probe's own test accuracy
    final_train_loss: float
    elapsed_s: float
    n_train: int
    n_params: int
    test_preds: torch.Tensor                # (N_test,) int64, on CPU
    #: Mean ||z_hat - z|| on the test split, at the selected epoch. How far the flow
    #: actually moved the features - small values mean the flow stayed near identity.
    mean_displacement: float = 0.0
    #: Mean ||z|| on the test split, so displacement can be read as a fraction of scale.
    mean_feature_norm: float = 0.0
    state_dict: dict[str, torch.Tensor] = field(default_factory=dict)
    history: dict[str, list[float]] = field(default_factory=dict)

    @property
    def delta_pp(self) -> float:
        """Change against the frozen linear probe, in percentage points."""
        return (self.test_acc - self.baseline_acc) * 100

    @property
    def relative_displacement(self) -> float:
        """Displacement as a fraction of the feature norm - scale-free, so comparable
        across encoders whose features live at different magnitudes."""
        if self.mean_feature_norm == 0:
            return 0.0
        return self.mean_displacement / self.mean_feature_norm


def rollout_with_velocities(net, z: torch.Tensor, T: int):
    """Return ``(z_hat, mean_squared_velocity)`` using Stage 2's integrator unchanged.

    ``euler_rollout`` already returns every intermediate state, and the Euler update is
    ``z_{k+1} = z_k + v_k / T``, so ``v_k = T (z_{k+1} - z_k)``. Recovering the velocities
    this way costs nothing and guarantees the penalty is measured on exactly the states the
    classification loss saw - re-running the network to get them could silently diverge.
    """
    z_hat, traj = euler_rollout(net, z, T, return_trajectory=True)
    steps = (traj[:, 1:, :] - traj[:, :-1, :]) * T          # (N, T, D)
    return z_hat, steps.pow(2).sum(dim=2).mean()


@torch.no_grad()
def _evaluate(net, clf: FrozenClassifier, x: torch.Tensor, y: torch.Tensor,
              T: int, criterion: nn.Module) -> tuple[float, float, torch.Tensor]:
    """Return ``(loss, accuracy, predictions)`` for a whole split."""
    net.eval()
    logits = clf(euler_rollout(net, x, T))
    preds = logits.argmax(dim=1)
    return criterion(logits, y).item(), (preds == y).float().mean().item(), preds


def train_end_to_end(
    net_dim: int,
    clf: FrozenClassifier,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    seed: int,
    baseline_acc: float,
    T: int = T_STEPS,
    config: EndToEndConfig | None = None,
    device: str = "cuda",
) -> EndToEndResult:
    """Train one flow end-to-end through a frozen classifier.

    The flow starts at identity, so epoch 0 of this run *is* the Stage 1 linear probe. The
    test split is touched once, at the end, using the best-validation-accuracy weights.

    Args:
        net_dim: feature dimensionality.
        clf: the frozen classifier. Its parameters are never passed to the optimiser.
        train_x/train_y: k-shot training features and labels.
        val_x/val_y: the full official validation split, for checkpoint selection only.
        test_x/test_y: the full official test split, touched once.
        seed: controls the flow's hidden-layer initialisation and batch order.
        baseline_acc: the frozen probe's test accuracy, carried through so ``delta_pp``
            is computed against the right number rather than re-derived later.
        T: Euler steps. Fixed at 4 for all of Stage 3.
        config: hyperparameters.
        device: ``"cuda"`` or ``"cpu"``.
    """
    config = config or EndToEndConfig()

    net = identity_flow(net_dim, seed=seed, hidden_dim=config.hidden_dim,
                        n_hidden=config.n_hidden, device=device)
    # Only the flow's parameters are optimised. The classifier is frozen by construction,
    # but passing only net.parameters() makes that true twice over.
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    n_train = train_x.shape[0]
    generator = torch.Generator().manual_seed(seed)

    history: dict[str, list[float]] = {
        "train_loss": [], "train_cls_loss": [], "train_reg": [],
        "val_loss": [], "val_acc": [], "train_acc": [],
    }
    best_val_acc, best_epoch, best_state = -1.0, -1, None
    need_velocity = config.velocity_weight > 0

    start = time.perf_counter()
    for epoch in range(config.max_epochs):
        net.train()
        order = torch.randperm(n_train, generator=generator).to(device)
        run_total = torch.zeros((), device=device)
        run_cls = torch.zeros((), device=device)
        run_reg = torch.zeros((), device=device)

        for start_i in range(0, n_train, config.batch_size):
            idx = order[start_i:start_i + config.batch_size]
            z = train_x[idx]
            optimizer.zero_grad(set_to_none=True)

            if need_velocity:
                z_hat, v_sq = rollout_with_velocities(net, z, T)
            else:
                z_hat, v_sq = euler_rollout(net, z, T), None

            cls_loss = criterion(clf(z_hat), train_y[idx])

            reg = torch.zeros((), device=device)
            if config.displacement_weight > 0:
                reg = reg + config.displacement_weight * (z_hat - z).pow(2).sum(1).mean()
            if need_velocity:
                reg = reg + config.velocity_weight * v_sq

            (cls_loss + reg).backward()
            if config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm)
            optimizer.step()

            run_total += (cls_loss + reg).detach() * idx.numel()
            run_cls += cls_loss.detach() * idx.numel()
            run_reg += reg.detach() * idx.numel()

        _, train_acc, _ = _evaluate(net, clf, train_x, train_y, T, criterion)
        val_loss, val_acc, _ = _evaluate(net, clf, val_x, val_y, T, criterion)

        history["train_loss"].append((run_total / n_train).item())
        history["train_cls_loss"].append((run_cls / n_train).item())
        history["train_reg"].append((run_reg / n_train).item())
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Strict '>' keeps the earliest best epoch, matching Stage 1's rule.
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

    net.load_state_dict(best_state)
    _, test_acc, test_preds = _evaluate(net, clf, test_x, test_y, T, criterion)

    with torch.no_grad():
        net.eval()
        z_hat_test = euler_rollout(net, test_x, T)
        mean_disp = (z_hat_test - test_x).norm(dim=1).mean().item()
        mean_norm = test_x.norm(dim=1).mean().item()

    return EndToEndResult(
        test_acc=test_acc,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        baseline_acc=baseline_acc,
        final_train_loss=history["train_loss"][-1],
        elapsed_s=time.perf_counter() - start,
        n_train=n_train,
        n_params=sum(p.numel() for p in net.parameters()),
        test_preds=test_preds.cpu(),
        mean_displacement=mean_disp,
        mean_feature_norm=mean_norm,
        state_dict={k: v.detach().cpu().clone() for k, v in best_state.items()},
        history=history,
    )
