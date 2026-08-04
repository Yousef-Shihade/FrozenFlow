"""
The linear probe — Stage 1's required baseline.

Implements exactly what the assignment specifies::

    s = W z + b

with ``z`` a frozen cached feature, only ``W`` and ``b`` trained, softmax cross-entropy
loss, and the suggested configuration (AdamW, lr 1e-3, weight decay 1e-4, batch size 64,
up to 200 epochs, checkpoint on **highest validation accuracy**).

Two details that are easy to get subtly wrong
---------------------------------------------
**The two kinds of seed.** For 5-shot and 10-shot the seed selects *which images* form the
training subset; for the full setting there is no subsampling left to vary, so the seed
instead controls the classifier's *initialisation* (and the batch order). Both are driven
by the same ``seed`` argument here, and the caller decides which role it plays by whether
it passes a subsampled training set or the whole one.

**Checkpoint selection uses validation accuracy, not loss.** The assignment says accuracy,
and the two disagree in practice: cross-entropy keeps rising as the model grows confident
on the examples it already gets right, long after accuracy has plateaued. Selecting on loss
would pick a systematically earlier epoch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

__all__ = ["ProbeConfig", "ProbeResult", "train_linear_probe"]


@dataclass(frozen=True)
class ProbeConfig:
    """The assignment's suggested configuration, used unmodified.

    The spec allows adjusting these if the defaults behave poorly and reporting the change.
    They did not, so they are used exactly as given — which is itself worth stating
    explicitly rather than leaving unsaid.
    """

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200
    optimizer: str = "AdamW"
    checkpoint_metric: str = "val_accuracy"

    def as_dict(self) -> dict:
        return {
            "optimizer": self.optimizer,
            "learning_rate": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "checkpoint_selection": self.checkpoint_metric,
        }


@dataclass
class ProbeResult:
    """Everything one probe run produces."""

    test_acc: float
    best_val_acc: float
    best_epoch: int
    final_train_loss: float
    elapsed_s: float
    n_train: int
    test_preds: torch.Tensor                      # (N_test,) int64, on CPU
    history: dict[str, list[float]] = field(default_factory=dict)


@torch.no_grad()
def _evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
              criterion: nn.Module) -> tuple[float, float, torch.Tensor]:
    """Return ``(loss, accuracy, predictions)`` for a whole split in one forward pass."""
    model.eval()
    logits = model(x)
    loss = criterion(logits, y).item()
    preds = logits.argmax(dim=1)
    acc = (preds == y).float().mean().item()
    return loss, acc, preds


def train_linear_probe(
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
) -> ProbeResult:
    """Train one linear probe on cached features and evaluate it once on the test split.

    The test split is touched exactly once, at the end, using the weights from the
    best-validation-accuracy epoch — never the final epoch, which is the most overfit one.

    Args:
        train_x/train_y: training features and labels (already k-shot subsampled if needed).
        val_x/val_y: the **full** official validation split, used only for checkpoint
            selection, as the assignment permits.
        test_x/test_y: the **full** official test split, used only for the final number.
        num_classes: 47 for DTD, 100 for FGVC-Aircraft.
        seed: controls weight initialisation and batch order.
        config: hyperparameters; defaults to the assignment's suggested configuration.
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        A :class:`ProbeResult` with the test accuracy, the selected epoch, per-epoch
        curves, and the test predictions (kept so step 5 can build confusion matrices
        without retraining anything).
    """
    import time

    config = config or ProbeConfig()

    # Determinism: seed before the layer is constructed so initialisation is reproducible.
    torch.manual_seed(seed)
    model = nn.Linear(train_x.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Features are small enough to live on the GPU for the whole run, which removes any
    # host-to-device copy from the inner loop.
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    n_train = train_x.shape[0]
    generator = torch.Generator().manual_seed(seed)   # batch order, independent of init

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "val_acc": []
    }
    best_val_acc, best_epoch, best_state = -1.0, -1, None

    start = time.perf_counter()
    for epoch in range(config.max_epochs):
        model.train()
        # Equivalent to DataLoader(shuffle=True) but without the per-batch overhead,
        # which matters when the whole run is 200 epochs of tiny batches.
        order = torch.randperm(n_train, generator=generator).to(device)
        running = torch.zeros((), device=device)

        for start_i in range(0, n_train, config.batch_size):
            idx = order[start_i:start_i + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_x[idx]), train_y[idx])
            loss.backward()
            optimizer.step()
            # Accumulate on-device; one sync per epoch instead of one per batch.
            running += loss.detach() * idx.numel()

        train_loss = (running / n_train).item()
        val_loss, val_acc, _ = _evaluate(model, val_x, val_y, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Strict '>' keeps the *earliest* epoch achieving the best accuracy, which is both
        # deterministic under ties and the less overfit of the tied choices.
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_loss, test_acc, test_preds = _evaluate(model, test_x, test_y, criterion)
    elapsed = time.perf_counter() - start

    return ProbeResult(
        test_acc=test_acc,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        final_train_loss=history["train_loss"][-1],
        elapsed_s=elapsed,
        n_train=n_train,
        test_preds=test_preds.cpu(),
        history=history,
    )
