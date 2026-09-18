"""
Recovering the Stage 1 linear probe, without modifying Stage 1.

Stage 3 needs the *weights* of the Stage 1 linear probe: it freezes that classifier and
trains a flow in front of it. Stage 1's :func:`cvlab.probe.train_linear_probe` does not
return them - it reports accuracy, curves and predictions, and lets the trained layer go out
of scope. Stage 1 is finished and published, so it is not edited to add them.

Instead the training loop is reproduced here, and :func:`train_frozen_classifier` returns the
weights alongside everything Stage 1 reported.

Why this is safe, given that duplicated logic is exactly what this project warns against
------------------------------------------------------------------------------------------
Two things keep the copy honest:

* **The hyperparameters are not duplicated.** ``ProbeConfig`` is imported from Stage 1, so
  the learning rate, weight decay, batch size, epoch budget and checkpoint rule have exactly
  one definition in the project. Only the loop that consumes them lives here.
* **Divergence cannot be silent.** Stage 3 step 01 re-derives all nine probes with this
  function and asserts the result matches Stage 1's published accuracies to 0.00e+00 pp with
  bit-identical test predictions, and ``Stage_3/tests/test_probe_equivalence.py`` runs this
  implementation against Stage 1's on the same inputs and asserts the two agree exactly. If
  this copy ever drifts, both fail immediately and loudly.

The evaluation helper is imported from Stage 1 rather than rewritten, because bit-exact
agreement depends on reducing the same way: the same ``argmax`` and the same float32 mean, on
the same device.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

# Single-sourced from Stage 1 on purpose - see the module docstring. `_evaluate` is private
# to `cvlab.probe`; it is reused rather than reimplemented because the numerics have to match
# bit-for-bit, and the equivalence test guards the coupling.
from cvlab.probe import ProbeConfig, _evaluate

__all__ = ["FrozenClassifierResult", "train_frozen_classifier"]


@dataclass
class FrozenClassifierResult:
    """One recovered probe: everything Stage 1 reported, plus the weights Stage 3 needs."""

    test_acc: float
    best_val_acc: float
    best_epoch: int
    final_train_loss: float
    elapsed_s: float
    n_train: int
    test_preds: torch.Tensor                      # (N_test,) int64, on CPU
    #: Weights of the selected (best-validation) epoch, on CPU:
    #: ``{"weight": (C, D), "bias": (C,)}``. This is the field Stage 1's result lacks and
    #: the reason this function exists.
    state_dict: dict[str, torch.Tensor] = field(default_factory=dict)
    history: dict[str, list[float]] = field(default_factory=dict)


def train_frozen_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    num_classes: int,
    seed: int,
    config: ProbeConfig | None = None,
    device: str = "cuda",
) -> FrozenClassifierResult:
    """Train one linear probe exactly as Stage 1 does, and keep its weights.

    The classifier needs to be trained exactly as in Stage 1 and then frozen. This
    reproduces Stage 1's procedure step for step: the same seeding order (the
    layer is constructed *after* ``manual_seed`` so its initialisation is reproducible), the
    same separate generator for batch order, the same optimiser, the same strict ``>``
    checkpoint rule that keeps the earliest best-validation epoch, and the same single
    evaluation on the test split at the end.

    Args:
        train_x/train_y: training features and labels, already k-shot subsampled.
        val_x/val_y: the **full** official validation split, for checkpoint selection only.
        test_x/test_y: the **full** official test split, touched once at the end.
        num_classes: 47 for DTD, 100 for FGVC-Aircraft.
        seed: controls weight initialisation and batch order.
        config: hyperparameters; defaults to Stage 1's ``ProbeConfig``.
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        A :class:`FrozenClassifierResult`, whose ``state_dict`` is the classifier Stage 3
        freezes.
    """
    config = config or ProbeConfig()

    # Determinism: seed before the layer is constructed so initialisation is reproducible.
    torch.manual_seed(seed)
    model = nn.Linear(train_x.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    n_train = train_x.shape[0]
    generator = torch.Generator().manual_seed(seed)   # batch order, independent of init

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch, best_state = -1.0, -1, None

    start = time.perf_counter()
    for epoch in range(config.max_epochs):
        model.train()
        order = torch.randperm(n_train, generator=generator).to(device)
        running = torch.zeros((), device=device)

        for start_i in range(0, n_train, config.batch_size):
            idx = order[start_i:start_i + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_x[idx]), train_y[idx])
            loss.backward()
            optimizer.step()
            running += loss.detach() * idx.numel()

        train_loss = (running / n_train).item()
        val_loss, val_acc, _ = _evaluate(model, val_x, val_y, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Strict '>' keeps the *earliest* epoch achieving the best accuracy, matching Stage 1.
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_loss, test_acc, test_preds = _evaluate(model, test_x, test_y, criterion)
    elapsed = time.perf_counter() - start

    return FrozenClassifierResult(
        test_acc=test_acc,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        final_train_loss=history["train_loss"][-1],
        elapsed_s=elapsed,
        n_train=n_train,
        test_preds=test_preds.cpu(),
        state_dict={k: v.detach().cpu().clone() for k, v in best_state.items()},
        history=history,
    )
