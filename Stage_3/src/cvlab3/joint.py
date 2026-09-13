"""
The optional extension: unfreezing the classifier and training it jointly with the flow.

Everything up to this point held the Stage 1 classifier fixed, which is what makes
"$\\Delta$ against the linear probe" mean something: the only thing that changed was the flow.
Unfreezing breaks that guarantee, and it introduces a confound that has to be controlled for.

The confound
------------
If ``flow + unfrozen classifier`` beats the probe, there are two possible explanations:

1. the flow is helping, or
2. **the classifier simply got more training**, and would have improved on its own.

Stage 1 trained the probe for up to 200 epochs with validation-accuracy checkpointing, so it
should already be at its ceiling - but "should" is not a measurement. :func:`train_joint`
therefore supports ``use_flow=False``, which continues training the classifier alone from the
Stage 1 weights, with the same optimiser, schedule, budget and checkpoint rule as the joint
runs. That control is what separates the two explanations, and step 06 runs it.

What `unfreeze` does
--------------------
:class:`~cvlab3.classifier.FrozenClassifier` is frozen by contract - its parameters carry
``requires_grad=False`` and it refuses to leave eval mode. Rather than defeating that (which
would make the class useless as a guarantee everywhere else), :func:`unfreeze` builds a
**separate, ordinary** ``nn.Linear`` initialised from the same weights. The frozen object stays
frozen; the trainable copy is explicitly a different object.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from cvlabfm.flow import euler_rollout

from cvlab3.classifier import T_STEPS, FrozenClassifier, identity_flow

__all__ = ["JointConfig", "JointResult", "unfreeze", "train_joint"]


def unfreeze(clf: FrozenClassifier, device: str = "cuda") -> nn.Linear:
    """Return a *trainable* copy of a frozen classifier, initialised from its weights.

    The original :class:`FrozenClassifier` is left untouched, so the frozen-classifier
    guarantee the rest of Stage 3 depends on cannot be weakened by this module existing.
    """
    layer = nn.Linear(clf.dim, clf.num_classes).to(device)
    with torch.no_grad():
        layer.weight.copy_(clf.linear.weight)
        layer.bias.copy_(clf.linear.bias)
    for p in layer.parameters():
        p.requires_grad_(True)
    return layer


@dataclass(frozen=True)
class JointConfig:
    """Settings for joint flow + classifier training."""

    hidden_dim: int = 512
    n_hidden: int = 2
    #: Learning rate for the flow. Step 02 measured 1e-3 to be harmful in this setting.
    lr: float = 1e-4
    #: Learning rate for the classifier. Separated from the flow's because the classifier
    #: starts already fitted while the flow starts at identity - they are not at comparable
    #: points in training, so one shared rate would be an arbitrary choice, not a neutral one.
    classifier_lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200
    max_grad_norm: float = 1.0
    #: Hold the classifier fixed for this many epochs before letting it move. 0 unfreezes
    #: immediately.
    unfreeze_epoch: int = 0
    #: Set False to train the classifier alone, with no flow at all - the control that
    #: separates "the flow helped" from "the classifier just got more training".
    use_flow: bool = True
    checkpoint_metric: str = "val_accuracy"

    def as_dict(self) -> dict:
        return {
            "hidden_dim": self.hidden_dim, "n_hidden": self.n_hidden,
            "flow_lr": self.lr, "classifier_lr": self.classifier_lr,
            "weight_decay": self.weight_decay, "batch_size": self.batch_size,
            "max_epochs": self.max_epochs, "max_grad_norm": self.max_grad_norm,
            "unfreeze_epoch": self.unfreeze_epoch, "use_flow": self.use_flow,
            "checkpoint_selection": self.checkpoint_metric,
        }


@dataclass
class JointResult:
    """Everything one joint run produces."""

    test_acc: float
    best_val_acc: float
    best_epoch: int
    baseline_acc: float
    final_train_loss: float
    elapsed_s: float
    n_train: int
    test_preds: torch.Tensor
    #: How far the classifier's weights moved from Stage 1's, in relative Frobenius norm.
    classifier_drift: float = 0.0
    mean_displacement: float = 0.0
    mean_feature_norm: float = 0.0
    history: dict[str, list[float]] = field(default_factory=dict)

    @property
    def delta_pp(self) -> float:
        return (self.test_acc - self.baseline_acc) * 100

    @property
    def relative_displacement(self) -> float:
        if self.mean_feature_norm == 0:
            return 0.0
        return self.mean_displacement / self.mean_feature_norm


@torch.no_grad()
def _evaluate(net, layer, x, y, T, criterion, use_flow):
    layer.eval()
    if use_flow:
        net.eval()
        feats = euler_rollout(net, x, T)
    else:
        feats = x
    logits = layer(feats)
    preds = logits.argmax(dim=1)
    return criterion(logits, y).item(), (preds == y).float().mean().item(), preds


def train_joint(
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
    config: JointConfig | None = None,
    device: str = "cuda",
) -> JointResult:
    """Train the flow and the classifier together (or the classifier alone).

    The flow starts at identity and the classifier starts at Stage 1's weights, so epoch 0 of
    every configuration here - including the control - **is** the Stage 1 linear probe. As in
    steps 02 and 03, the test split is touched once, using the best-validation-accuracy state.
    """
    config = config or JointConfig()

    net = identity_flow(net_dim, seed=seed, hidden_dim=config.hidden_dim,
                        n_hidden=config.n_hidden, device=device)
    layer = unfreeze(clf, device=device)
    original_w = layer.weight.detach().clone()

    groups = [{"params": layer.parameters(), "lr": config.classifier_lr}]
    if config.use_flow:
        groups.insert(0, {"params": net.parameters(), "lr": config.lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    n_train = train_x.shape[0]
    generator = torch.Generator().manual_seed(seed)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": [],
                                       "train_acc": [], "classifier_drift": []}
    best_val_acc, best_epoch, best_state = -1.0, -1, None

    start = time.perf_counter()
    for epoch in range(config.max_epochs):
        classifier_active = epoch >= config.unfreeze_epoch
        for p in layer.parameters():
            p.requires_grad_(classifier_active)

        if config.use_flow:
            net.train()
        layer.train()
        order = torch.randperm(n_train, generator=generator).to(device)
        running = torch.zeros((), device=device)

        for start_i in range(0, n_train, config.batch_size):
            idx = order[start_i:start_i + config.batch_size]
            z = train_x[idx]
            optimizer.zero_grad(set_to_none=True)

            feats = euler_rollout(net, z, T) if config.use_flow else z
            loss = criterion(layer(feats), train_y[idx])
            loss.backward()
            if config.max_grad_norm > 0:
                params = list(layer.parameters())
                if config.use_flow:
                    params += list(net.parameters())
                nn.utils.clip_grad_norm_(params, config.max_grad_norm)
            optimizer.step()
            running += loss.detach() * idx.numel()

        _, train_acc, _ = _evaluate(net, layer, train_x, train_y, T, criterion, config.use_flow)
        val_loss, val_acc, _ = _evaluate(net, layer, val_x, val_y, T, criterion, config.use_flow)
        drift = ((layer.weight.detach() - original_w).norm() / original_w.norm()).item()

        history["train_loss"].append((running / n_train).item())
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["classifier_drift"].append(drift)

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = ({k: v.detach().clone() for k, v in net.state_dict().items()},
                          {k: v.detach().clone() for k, v in layer.state_dict().items()})

    net.load_state_dict(best_state[0])
    layer.load_state_dict(best_state[1])
    _, test_acc, test_preds = _evaluate(net, layer, test_x, test_y, T, criterion, config.use_flow)

    with torch.no_grad():
        drift = ((layer.weight.detach() - original_w).norm() / original_w.norm()).item()
        if config.use_flow:
            net.eval()
            z_hat = euler_rollout(net, test_x, T)
            mean_disp = (z_hat - test_x).norm(dim=1).mean().item()
        else:
            mean_disp = 0.0
        mean_norm = test_x.norm(dim=1).mean().item()

    return JointResult(
        test_acc=test_acc, best_val_acc=best_val_acc, best_epoch=best_epoch,
        baseline_acc=baseline_acc, final_train_loss=history["train_loss"][-1],
        elapsed_s=time.perf_counter() - start, n_train=n_train,
        test_preds=test_preds.cpu(), classifier_drift=drift,
        mean_displacement=mean_disp, mean_feature_norm=mean_norm, history=history,
    )
