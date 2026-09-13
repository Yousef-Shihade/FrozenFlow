"""
Strategy 2: classifier-guided targets with standard FM training.

Instead of differentiating the classification loss through the whole rollout (Strategy 1),
the frozen classifier is used to *construct an explicit target* for each feature, and the flow
is then trained toward that target with an ordinary Flow Matching update.

One training step, following the brief exactly:

1. Roll :math:`z` through the **current** flow to get :math:`\\hat z`.
2. Push :math:`\\hat z` through the frozen classifier and compute the classification loss.
3. Use :math:`\\partial \\mathcal{L} / \\partial \\hat z` to build a nearby *improved*
   representation :math:`\\hat z'` - one or more gradient steps taken in feature space.
4. Treat :math:`z` as the source and :math:`\\hat z'` as the target.
5. Perform a standard FM update between them::

       t ~ U(0, 1),  z_t = (1 - t) z + t z',  u = z' - z
       L_FM = || v(z_t, t) - u ||^2

6. Recompute the targets as the flow changes during training.

How this differs from Stage 2's standard FM
-------------------------------------------
Mechanically the update is identical - and deliberately so, since the brief asks for "a
standard FM training update". The difference is where the target comes from. Stage 2's targets
were **fixed class prototypes**: one vector per class, known before training, shared by every
example of that class. Here the target is **per-example and moves during training**, because it
is derived from the current flow's own output through a classifier gradient.

That has a consequence worth stating: Stage 2's targets were a fixed point the flow could
converge to. These targets are recomputed from the flow that is chasing them, so the objective
is non-stationary by construction. ``refresh_every`` controls how non-stationary: recomputing
every epoch tracks the flow closely, recomputing rarely gives a stable target that goes stale.

Normalising the guidance step
-----------------------------
With ``normalize_step=True`` the gradient is scaled to unit norm before the step, so
``eta`` *is* the distance moved in feature space and can be read directly against
:math:`\\lVert z \\rVert \\approx 24\\text{-}50`. Without it, the step size inherits the scale of
the classification loss, which step 02 measured at 0.005-0.33 depending on the combination -
three combinations would then be running three very different effective step sizes under one
nominal setting. Both are offered; the sweep reports both.

Strictly, ``eta`` is an *upper* bound on the step: the division uses ``clamp_min(1e-12)``,
so an example whose gradient underflows below that takes a shorter step rather than being
blown up by dividing by a denormal. That is the behaviour we want - there is no meaningful
descent direction for such an example - and it never triggers on the data Stage 3 runs
(0 of 2,470 training features across the three combinations). The test suite asserts both
halves rather than assuming it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from cvlabfm.flow import euler_rollout

from cvlab3.classifier import T_STEPS, FrozenClassifier, identity_flow

__all__ = ["GuidedConfig", "GuidedResult", "build_targets", "fm_pair",
           "train_classifier_guided"]


@dataclass(frozen=True)
class GuidedConfig:
    """Settings for Strategy 2."""

    hidden_dim: int = 512
    n_hidden: int = 2
    lr: float = 1e-4                 # step 02 measured 1e-3 to be actively harmful here
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200
    optimizer: str = "AdamW"
    max_grad_norm: float = 1.0

    #: Feature-space step size for the guidance. With ``normalize_step`` this is the
    #: distance moved per step, directly comparable to ||z|| ~ 24-50.
    eta: float = 1.0
    #: Number of gradient steps taken to build each target.
    n_target_steps: int = 1
    #: Rebuild the targets every this many epochs. 1 tracks the flow closely; larger
    #: values hold a staler but more stable target.
    refresh_every: int = 1
    #: Scale the guidance gradient to unit norm before stepping.
    normalize_step: bool = True

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
            "eta": self.eta,
            "n_target_steps": self.n_target_steps,
            "refresh_every": self.refresh_every,
            "normalize_step": self.normalize_step,
            "checkpoint_selection": self.checkpoint_metric,
        }


@dataclass
class GuidedResult:
    """Everything one Strategy 2 run produces."""

    test_acc: float
    best_val_acc: float
    best_epoch: int
    baseline_acc: float
    final_train_loss: float
    elapsed_s: float
    n_train: int
    n_params: int
    test_preds: torch.Tensor
    mean_displacement: float = 0.0
    mean_feature_norm: float = 0.0
    #: Accuracy of the *targets themselves* at the last refresh. The flow can never do
    #: better than what it is being aimed at, so this is the method's own ceiling.
    target_train_acc: float = 0.0
    #: Cross-entropy of the frozen classifier on those targets. On combinations where the
    #: target accuracy is already 100%, this is the only quantity that shows what the
    #: guidance is doing - pushing confidence rather than correctness.
    target_train_ce: float = 0.0
    state_dict: dict[str, torch.Tensor] = field(default_factory=dict)
    history: dict[str, list[float]] = field(default_factory=dict)

    @property
    def delta_pp(self) -> float:
        return (self.test_acc - self.baseline_acc) * 100

    @property
    def relative_displacement(self) -> float:
        if self.mean_feature_norm == 0:
            return 0.0
        return self.mean_displacement / self.mean_feature_norm


def build_targets(net, clf: FrozenClassifier, z: torch.Tensor, y: torch.Tensor,
                  config: GuidedConfig, T: int = T_STEPS) -> tuple[torch.Tensor, float]:
    """Construct the guided targets :math:`\\hat z'` for a batch of features.

    Steps 1-3 of the procedure: roll ``z`` through the current flow, then descend the
    classification loss in feature space.

    The flow itself is not differentiated here - only the *feature* carries a gradient. The
    returned target is detached, so the FM update in
    :func:`train_classifier_guided` treats it as a fixed vector, which is what makes the
    update an ordinary Flow Matching step rather than a second end-to-end objective.

    Returns:
        ``(targets, target_accuracy, target_ce)`` - the improved features, how often the
        frozen classifier gets them right, and its cross-entropy on them.

        Both numbers are needed. Step 02 measured that the probe already classifies its own
        k-shot training set at 96-100%, so on two of the three combinations the target
        accuracy is pinned at 100% and cannot report anything. The cross-entropy still moves:
        what the guidance does there is push features toward *higher-confidence* regions, not
        toward correctness it has already achieved.
    """
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        z_hat = euler_rollout(net, z, T)

    current = z_hat.detach()
    for _ in range(config.n_target_steps):
        current = current.detach().requires_grad_(True)
        loss = criterion(clf(current), y)
        grad, = torch.autograd.grad(loss, current)

        if config.normalize_step:
            # Unit-norm per example, so `eta` is the distance actually travelled.
            grad = grad / grad.norm(dim=1, keepdim=True).clamp_min(1e-12)

        current = current - config.eta * grad

    targets = current.detach()
    with torch.no_grad():
        logits = clf(targets)
        target_acc = (logits.argmax(dim=1) == y).float().mean().item()
        target_ce = criterion(logits, y).item()
    return targets, target_acc, target_ce


def fm_pair(z: torch.Tensor, z_prime: torch.Tensor,
            t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The standard Flow Matching pair for source ``z`` and target ``z_prime`` at time ``t``.

    Returns ``(z_t, u)`` where::

        z_t = (1 - t) z + t z'          the interpolated state
        u   = z' - z                    the target velocity

    Lifted out of the training loop so the formula is reachable by a test. Getting either
    term backwards still trains and still converges - to the wrong transport - so this is
    checked directly rather than trusted.
    """
    t = t.unsqueeze(1) if t.dim() == 1 else t
    return (1 - t) * z + t * z_prime, z_prime - z


@torch.no_grad()
def _evaluate(net, clf: FrozenClassifier, x: torch.Tensor, y: torch.Tensor,
              T: int, criterion: nn.Module) -> tuple[float, float, torch.Tensor]:
    net.eval()
    logits = clf(euler_rollout(net, x, T))
    preds = logits.argmax(dim=1)
    return criterion(logits, y).item(), (preds == y).float().mean().item(), preds


def train_classifier_guided(
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
    config: GuidedConfig | None = None,
    device: str = "cuda",
) -> GuidedResult:
    """Train one flow with classifier-guided targets and standard FM updates.

    The flow starts at identity, so the first set of targets is built from the original
    features. Checkpoint selection is on validation accuracy, matching step 02 and Stage 1,
    so the two strategies are selected by the same rule and stay comparable.
    """
    config = config or GuidedConfig()

    net = identity_flow(net_dim, seed=seed, hidden_dim=config.hidden_dim,
                        n_hidden=config.n_hidden, device=device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    n_train = train_x.shape[0]
    generator = torch.Generator().manual_seed(seed)
    # Separate generator for the FM time samples, so batch order and t-draws do not
    # interfere with each other across configurations.
    t_generator = torch.Generator(device=device).manual_seed(seed + 10_000)

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "val_acc": [], "train_acc": [],
        "target_acc": [], "target_ce": [],
    }
    best_val_acc, best_epoch, best_state = -1.0, -1, None
    targets, target_acc, target_ce = None, 0.0, 0.0

    start = time.perf_counter()
    for epoch in range(config.max_epochs):
        # Step 6: recompute the targets as the flow changes.
        if epoch % config.refresh_every == 0:
            net.eval()
            targets, target_acc, target_ce = build_targets(
                net, clf, train_x, train_y, config, T)

        net.train()
        order = torch.randperm(n_train, generator=generator).to(device)
        running = torch.zeros((), device=device)

        for start_i in range(0, n_train, config.batch_size):
            idx = order[start_i:start_i + config.batch_size]
            z, z_prime = train_x[idx], targets[idx]

            # Step 5: a standard FM update between source z and target z'.
            t = torch.rand(z.shape[0], device=device, generator=t_generator)
            z_t, u = fm_pair(z, z_prime, t)

            optimizer.zero_grad(set_to_none=True)
            loss = mse(net(z_t, t), u)
            loss.backward()
            if config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm)
            optimizer.step()
            running += loss.detach() * idx.numel()

        _, train_acc, _ = _evaluate(net, clf, train_x, train_y, T, criterion)
        val_loss, val_acc, _ = _evaluate(net, clf, val_x, val_y, T, criterion)

        history["train_loss"].append((running / n_train).item())
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["target_acc"].append(target_acc)
        history["target_ce"].append(target_ce)

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

    return GuidedResult(
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
        target_train_acc=target_acc,
        target_train_ce=target_ce,
        state_dict={k: v.detach().cpu().clone() for k, v in best_state.items()},
        history=history,
    )
