"""
Guards the one piece of logic Stage 3 duplicates from Stage 1.

Stage 3 needs the linear probe's *weights*, which ``cvlab.probe.train_linear_probe`` does not
return. Stage 1 is finished and published, so rather than editing it, Stage 3 reproduces its
training loop in ``cvlab3.probe.train_frozen_classifier``.

Duplicated logic is precisely what this project warns against, because a fix applied to one
copy silently leaves the other wrong. These tests remove the "silently": they run both
implementations on the same inputs and assert the results are bit-identical. If Stage 1's
probe is ever changed, or Stage 3's copy drifts, this fails.

The velocity-network tests do the same job for the other thing Stage 3 could have changed in
a finished stage: rather than adding an initialisation flag to Stage 2's ``VelocityNet``,
``cvlab3.identity_flow`` constructs it unmodified and zeroes the readout afterwards.
"""

from __future__ import annotations

import torch

from cvlab.probe import ProbeConfig, train_linear_probe
from cvlab3 import FrozenClassifier, transform
from cvlab3.classifier import identity_flow
from cvlab3.probe import train_frozen_classifier

DEVICE = "cpu"          # deterministic and dependency-free; the loop under test is identical


def _frozen_toy_classifier(dim: int, num_classes: int, seed: int = 0) -> FrozenClassifier:
    """A small frozen classifier, so the gradient tests exercise the real Stage 3 path."""
    g = torch.Generator().manual_seed(seed)
    return FrozenClassifier(torch.randn(num_classes, dim, generator=g),
                            torch.randn(num_classes, generator=g))


def _toy_problem(dim: int = 16, num_classes: int = 4, seed: int = 0):
    """A small, separable classification problem, generated deterministically."""
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(num_classes, dim, generator=g) * 3.0

    def make(n_per_class: int):
        y = torch.arange(num_classes).repeat_interleave(n_per_class)
        x = centres[y] + torch.randn(len(y), dim, generator=g)
        return x, y

    return make(8), make(6), make(6), num_classes


def _run_both(epochs: int = 6, seed: int = 0):
    (tr_x, tr_y), (va_x, va_y), (te_x, te_y), n_cls = _toy_problem()
    config = ProbeConfig(max_epochs=epochs)
    args = dict(train_x=tr_x, train_y=tr_y, val_x=va_x, val_y=va_y,
                test_x=te_x, test_y=te_y, num_classes=n_cls, seed=seed,
                config=config, device=DEVICE)
    return train_linear_probe(**args), train_frozen_classifier(**args)


def test_test_accuracy_is_identical():
    stage1, stage3 = _run_both()
    assert stage3.test_acc == stage1.test_acc


def test_test_predictions_are_identical():
    stage1, stage3 = _run_both()
    assert torch.equal(stage3.test_preds, stage1.test_preds)


def test_selected_epoch_and_val_accuracy_are_identical():
    """The checkpoint rule is the subtlest part of the loop, so it is checked directly."""
    stage1, stage3 = _run_both()
    assert stage3.best_epoch == stage1.best_epoch
    assert stage3.best_val_acc == stage1.best_val_acc


def test_training_curves_are_identical():
    """Every epoch, not just the endpoint - this catches drift mid-run."""
    stage1, stage3 = _run_both()
    assert stage3.history.keys() == stage1.history.keys()
    for key in stage1.history:
        assert stage3.history[key] == stage1.history[key], f"{key} diverged"


def test_stage3_additionally_returns_usable_weights():
    """The entire reason the Stage 3 copy exists."""
    _, stage3 = _run_both()
    assert set(stage3.state_dict) == {"weight", "bias"}
    assert stage3.state_dict["weight"].shape == (4, 16)
    assert stage3.state_dict["bias"].shape == (4,)
    assert all(t.device.type == "cpu" for t in stage3.state_dict.values())


def test_returned_weights_reproduce_the_reported_predictions():
    """The weights must be the ones that produced the reported result, not a later epoch."""
    (_, _), (_, _), (te_x, te_y), n_cls = _toy_problem()
    _, stage3 = _run_both()

    logits = te_x @ stage3.state_dict["weight"].T + stage3.state_dict["bias"]
    assert torch.equal(logits.argmax(dim=1), stage3.test_preds)


def test_identity_flow_leaves_features_bit_identical():
    """Stage 3's core setup requirement: the untrained system must be the probe itself."""
    net = identity_flow(32, seed=0, device=DEVICE)
    z = torch.randn(20, 32)
    assert torch.equal(transform(net, z, T=4), z)


def test_identity_flow_predicts_exactly_zero_velocity():
    net = identity_flow(32, seed=1, device=DEVICE)
    v = net(torch.randn(20, 32), torch.rand(20))
    assert v.abs().max().item() == 0.0


def test_identity_flow_zeroes_only_the_readout():
    """Zeroing the readout must not zero the whole network, or nothing could be learned."""
    net = identity_flow(32, seed=2, device=DEVICE)
    assert net.net[0].weight.abs().max().item() > 0
    assert net.net[-1].weight.abs().max().item() == 0.0


def test_identity_init_gives_hidden_layers_no_gradient_on_the_first_step():
    """The real dynamics of a zero-initialised readout, asserted rather than assumed.

    A hidden layer's gradient is proportional to the readout weight above it, so while that
    readout is exactly zero the hidden layers get exactly zero gradient. Only the readout
    moves on the first step. This is worth pinning down because it is easy to assume the
    opposite - that random hidden layers mean an informative gradient everywhere.
    """
    net = identity_flow(32, seed=3, device=DEVICE)
    clf = _frozen_toy_classifier(dim=32, num_classes=5)
    z = torch.randn(16, 32)
    y = torch.randint(0, 5, (16,))

    torch.nn.functional.cross_entropy(clf(transform(net, z, T=4)), y).backward()

    assert net.net[0].weight.grad.abs().max().item() == 0.0, "hidden layer got gradient"
    assert net.net[-1].weight.grad.abs().max().item() > 0, "readout got no gradient"


def test_hidden_layers_start_training_after_the_first_step():
    """...and the zero gradient lasts exactly one step, so the network does train."""
    net = identity_flow(32, seed=4, device=DEVICE)
    clf = _frozen_toy_classifier(dim=32, num_classes=5)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    z = torch.randn(64, 32)
    y = torch.randint(0, 5, (64,))

    hidden_grads, losses = [], []
    for _ in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(clf(transform(net, z, T=4)), y)
        loss.backward()
        hidden_grads.append(net.net[0].weight.grad.abs().max().item())
        losses.append(loss.item())
        opt.step()

    assert hidden_grads[0] == 0.0
    assert hidden_grads[1] > 0 and hidden_grads[2] > 0
    assert losses[-1] < losses[0], "loss did not decrease"
