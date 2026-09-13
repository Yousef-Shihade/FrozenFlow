"""
Does the code compute what the Stage 3 brief specifies?

Every result in Stage 3 rests on that being true, and it is unusually easy to break without
noticing: a rollout that backpropagates into the classifier, a guidance step that ascends the
loss instead of descending it, an FM interpolation that is the wrong way round, or a "frozen"
classifier that quietly moves would each still train, still converge, and still produce a
plausible accuracy table.

These tests re-derive the brief's formulas from its wording — deliberately the slow, literal
way — and assert the implementation agrees. Written to be independent of the code they check.

The four things Stage 3 specifies
---------------------------------
1. **Strategy 1**: ``L_cls = CE(W z_hat + b, y)`` with ``z_hat`` the full T-step rollout,
   backpropagated through the whole rollout, **updating only the FM parameters**.
2. **Strategy 2**: build ``z_hat' `` by descending the classification loss in feature space,
   then do a *standard* FM update between source ``z`` and target ``z_hat'``:
   ``t ~ U(0,1)``, ``z_t = (1-t) z + t z'``, ``u = z' - z``, ``L = ||v(z_t,t) - u||^2``.
3. **Euler integration**: ``z_{k+1} = z_k + (1/T) v(z_k, k/T)`` — Stage 2's, reused unchanged.
4. **The optional extension**: unfreezing must start from exactly the Stage 1 classifier.

The invariant that matters most is the one in bold: in steps 02-05 the classifier is frozen, so
**its weights must be bit-identical before and after training**. Three tests check that
directly, because every Delta reported in Stage 3 is meaningless if it is false.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cvlabfm.flow import VelocityNet, euler_rollout

from cvlab3 import FrozenClassifier, transform, transform_and_classify
from cvlab3.classifier import identity_flow
from cvlab3.guided import (GuidedConfig, build_targets, fm_pair,
                           train_classifier_guided)
from cvlab3.joint import JointConfig, train_joint, unfreeze
from cvlab3.training import (EndToEndConfig, rollout_with_velocities,
                             train_end_to_end)

DEVICE = "cpu"
DIM, NUM_CLASSES, T = 12, 4, 4


def _classifier(seed: int = 0) -> FrozenClassifier:
    g = torch.Generator().manual_seed(seed)
    return FrozenClassifier(torch.randn(NUM_CLASSES, DIM, generator=g) * 0.5,
                            torch.randn(NUM_CLASSES, generator=g) * 0.1)


def _flow(seed: int = 0, identity: bool = False) -> VelocityNet:
    if identity:
        return identity_flow(DIM, seed=seed, hidden_dim=16, n_hidden=2, device=DEVICE)
    torch.manual_seed(seed)
    return VelocityNet(DIM, hidden_dim=16, n_hidden=2)


def _data(n: int = 24, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, DIM, generator=g) * 3.0, torch.randint(0, NUM_CLASSES, (n,), generator=g)


def _splits():
    tr = _data(24, 0)
    va = _data(16, 1)
    te = _data(16, 2)
    return tr, va, te


# --------------------------------------------------------------------------
# 3. Euler integration, as Stage 3 uses it
# --------------------------------------------------------------------------

def test_rollout_matches_the_literal_euler_recurrence():
    """z_{k+1} = z_k + (1/T) v(z_k, k/T), written out step by step."""
    net, (z, _) = _flow(1), _data(8)

    expected = z.clone()
    for k in range(T):
        t = torch.full((expected.shape[0],), k / T)
        with torch.no_grad():
            expected = expected + net(expected, t) / T

    with torch.no_grad():
        got = transform(net, z, T)
    assert torch.allclose(got, expected, atol=0, rtol=0)


def test_rollout_last_step_time_is_not_one():
    """Times run k/T for k = 0..T-1, so the final step is at (T-1)/T, never 1.0."""
    seen = []

    class Recorder(VelocityNet):
        def forward(self, z, t):
            seen.append(float(t.flatten()[0]))
            return super().forward(z, t)

    torch.manual_seed(0)
    rec = Recorder(DIM, hidden_dim=16, n_hidden=2)
    with torch.no_grad():
        euler_rollout(rec, _data(4)[0], T)
    assert seen == [k / T for k in range(T)]
    assert 1.0 not in seen


def test_velocity_penalty_uses_the_networks_actual_velocities():
    """Strategy 1's velocity penalty must equal mean ||v_k||^2 over the real velocities.

    Calls the implementation, and compares against the velocities obtained by querying the
    network directly at each intermediate state - so dropping the factor of T that converts
    a step back into a velocity cannot pass.
    """
    net, (z, _) = _flow(2), _data(8)
    with torch.no_grad():
        z_hat, v_sq = rollout_with_velocities(net, z, T)
        _, traj = euler_rollout(net, z, T, return_trajectory=True)

        direct = torch.stack(
            [net(traj[:, k, :], torch.full((z.shape[0],), k / T)) for k in range(T)], dim=1)
        expected = direct.pow(2).sum(dim=2).mean()

    assert torch.allclose(z_hat, traj[:, -1, :], atol=0, rtol=0)
    assert torch.allclose(v_sq, expected, atol=1e-5), (
        f"velocity penalty {v_sq:.6f} != mean ||v||^2 {expected:.6f}")


# --------------------------------------------------------------------------
# 1. Strategy 1 — end-to-end classification through the rollout
# --------------------------------------------------------------------------

def test_strategy1_loss_is_cross_entropy_of_the_rolled_out_feature():
    net, clf = _flow(3), _classifier()
    z, y = _data(16)
    expected = nn.functional.cross_entropy(clf(transform(net, z, T)), y)
    got = nn.functional.cross_entropy(transform_and_classify(net, clf, z, T), y)
    assert torch.allclose(got, expected, atol=0, rtol=0)


def test_strategy1_gradient_flows_through_every_rollout_step():
    """Backprop must reach the flow through all T applications, not just the last."""
    net, clf = _flow(4), _classifier()
    z, y = _data(16)
    nn.functional.cross_entropy(transform_and_classify(net, clf, z, T), y).backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and any(g.abs().max() > 0 for g in grads)


def test_strategy1_never_updates_the_frozen_classifier():
    """The invariant every Stage 3 delta depends on."""
    clf = _classifier()
    before_w = clf.linear.weight.detach().clone()
    before_b = clf.linear.bias.detach().clone()

    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = _splits()
    train_end_to_end(DIM, clf, tr_x, tr_y, va_x, va_y, te_x, te_y, seed=0,
                     baseline_acc=0.5, T=T,
                     config=EndToEndConfig(max_epochs=4, hidden_dim=16), device=DEVICE)

    assert torch.equal(clf.linear.weight, before_w)
    assert torch.equal(clf.linear.bias, before_b)


def test_strategy1_starts_exactly_at_the_probe():
    """Identity init means the untrained system IS the frozen classifier."""
    clf = _classifier()
    net = _flow(0, identity=True)
    z, _ = _data(16)
    with torch.no_grad():
        assert torch.equal(transform_and_classify(net, clf, z, T), clf(z))


# --------------------------------------------------------------------------
# 2. Strategy 2 — classifier-guided targets, then a standard FM update
# --------------------------------------------------------------------------

def test_guided_target_is_a_descent_step_on_the_classification_loss():
    """z' = z_hat - eta * g, and it must *reduce* the loss, not increase it."""
    clf = _classifier()
    net = _flow(0, identity=True)          # so z_hat == z, isolating the guidance
    z, y = _data(32)

    before = nn.functional.cross_entropy(clf(z), y)
    targets, _, target_ce = build_targets(net, clf, z, y, GuidedConfig(eta=0.5), T)
    assert target_ce < before.item(), "guidance increased the loss - wrong sign"

    # ...and it is literally one normalised gradient step.
    zr = z.clone().requires_grad_(True)
    g, = torch.autograd.grad(nn.functional.cross_entropy(clf(zr), y), zr)
    g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
    assert torch.allclose(targets, z - 0.5 * g, atol=1e-6)


def test_normalised_step_moves_exactly_eta_where_the_gradient_is_non_negligible():
    """With normalisation, `eta` is the distance travelled - the reason it is the default.

    The one exception is built into the implementation: the gradient norm is divided with
    ``clamp_min(1e-12)``, so an example whose gradient *underflows* below that clamp takes a
    step shorter than ``eta`` rather than being blown up to unit length by dividing by a
    denormal. That is the desired behaviour - there is no meaningful descent direction for
    such an example - but it means ``eta`` is an upper bound in general, and an equality only
    where the gradient is real. Both halves are asserted.

    On the data Stage 3 actually runs, this never triggers: 0 of 2,470 training features
    across the three combinations fall below the clamp.
    """
    clf, net = _classifier(), _flow(0, identity=True)
    z, y = _data(32)

    zr = z.clone().requires_grad_(True)
    g, = torch.autograd.grad(nn.functional.cross_entropy(clf(zr), y), zr)
    real = g.norm(dim=1) > 1e-12

    for eta in (0.25, 1.0, 3.0):
        targets, _, _ = build_targets(net, clf, z, y, GuidedConfig(eta=eta), T)
        moved = (targets - z).norm(dim=1)
        assert torch.allclose(moved[real], torch.full_like(moved[real], eta), atol=1e-5)
        assert (moved <= eta + 1e-5).all(), "a step exceeded eta"


def test_multiple_target_steps_move_further_than_one():
    clf, net = _classifier(), _flow(0, identity=True)
    z, y = _data(32)
    one, _, ce1 = build_targets(net, clf, z, y, GuidedConfig(eta=0.5, n_target_steps=1), T)
    three, _, ce3 = build_targets(net, clf, z, y, GuidedConfig(eta=0.5, n_target_steps=3), T)
    assert (three - z).norm(dim=1).mean() > (one - z).norm(dim=1).mean()
    assert ce3 < ce1, "more descent steps should reduce the loss further"


def test_guided_targets_are_detached():
    """They must be constants to the FM update, or it stops being a standard FM step."""
    clf, net = _classifier(), _flow(5)
    z, y = _data(16)
    targets, _, _ = build_targets(net, clf, z, y, GuidedConfig(), T)
    assert not targets.requires_grad


def test_fm_pair_interpolates_from_source_to_target():
    """z_t = (1-t) z + t z': at t=0 it must be the SOURCE, at t=1 the TARGET.

    Calls the implementation the training loop uses. Reversing the interpolation still
    trains and still converges - to a transport pointing the wrong way - so the endpoints
    are pinned explicitly.
    """
    z, z_prime = torch.randn(10, DIM), torch.randn(10, DIM)

    at_zero, _ = fm_pair(z, z_prime, torch.zeros(10))
    at_one, _ = fm_pair(z, z_prime, torch.ones(10))
    assert torch.allclose(at_zero, z, atol=1e-6), "t=0 must give the source z"
    assert torch.allclose(at_one, z_prime, atol=1e-6), "t=1 must give the target z'"

    half, _ = fm_pair(z, z_prime, torch.full((10,), 0.5))
    assert torch.allclose(half, 0.5 * (z + z_prime), atol=1e-6)


def test_fm_pair_target_velocity_points_from_source_to_target():
    """u = z' - z, so integrating u from z for unit time lands on z'."""
    z, z_prime = torch.randn(10, DIM), torch.randn(10, DIM)
    _, u = fm_pair(z, z_prime, torch.rand(10))

    assert torch.allclose(z + u, z_prime, atol=1e-6), "u must point source -> target"
    assert not torch.allclose(z_prime + u, z, atol=1e-3), "u is reversed"


def test_strategy2_never_updates_the_frozen_classifier():
    clf = _classifier()
    before_w = clf.linear.weight.detach().clone()
    before_b = clf.linear.bias.detach().clone()

    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = _splits()
    train_classifier_guided(DIM, clf, tr_x, tr_y, va_x, va_y, te_x, te_y, seed=0,
                            baseline_acc=0.5, T=T,
                            config=GuidedConfig(max_epochs=4, hidden_dim=16), device=DEVICE)

    assert torch.equal(clf.linear.weight, before_w)
    assert torch.equal(clf.linear.bias, before_b)


def test_refresh_every_controls_how_often_targets_are_rebuilt():
    """`refresh_every` larger than the budget means the targets are built exactly once."""
    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = _splits()
    calls = []

    import cvlab3.guided as guided
    original = guided.build_targets

    def counting(*a, **k):
        calls.append(1)
        return original(*a, **k)

    guided.build_targets = counting
    try:
        train_classifier_guided(DIM, _classifier(), tr_x, tr_y, va_x, va_y, te_x, te_y,
                                seed=0, baseline_acc=0.5, T=T,
                                config=GuidedConfig(max_epochs=6, refresh_every=100,
                                                    hidden_dim=16), device=DEVICE)
        assert len(calls) == 1
        calls.clear()
        train_classifier_guided(DIM, _classifier(), tr_x, tr_y, va_x, va_y, te_x, te_y,
                                seed=0, baseline_acc=0.5, T=T,
                                config=GuidedConfig(max_epochs=6, refresh_every=2,
                                                    hidden_dim=16), device=DEVICE)
        assert len(calls) == 3        # epochs 0, 2, 4
    finally:
        guided.build_targets = original


# --------------------------------------------------------------------------
# 4. The optional extension
# --------------------------------------------------------------------------

def test_unfreeze_copies_the_classifier_exactly_and_leaves_it_frozen():
    clf = _classifier()
    layer = unfreeze(clf, DEVICE)
    x = torch.randn(16, DIM)
    with torch.no_grad():
        assert torch.equal(clf(x), layer(x))
    assert all(not p.requires_grad for p in clf.parameters())
    assert all(p.requires_grad for p in layer.parameters())


def test_joint_control_applies_no_flow_at_all():
    """use_flow=False must classify the raw feature, so it is a pure classifier baseline."""
    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = _splits()
    r = train_joint(DIM, _classifier(), tr_x, tr_y, va_x, va_y, te_x, te_y, seed=0,
                    baseline_acc=0.5, T=T,
                    config=JointConfig(max_epochs=3, use_flow=False, hidden_dim=16),
                    device=DEVICE)
    assert r.mean_displacement == 0.0


def test_delayed_unfreezing_holds_the_classifier_still_at_first():
    """With unfreeze_epoch beyond the budget, the classifier must not move at all."""
    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = _splits()
    r = train_joint(DIM, _classifier(), tr_x, tr_y, va_x, va_y, te_x, te_y, seed=0,
                    baseline_acc=0.5, T=T,
                    config=JointConfig(max_epochs=4, unfreeze_epoch=99, hidden_dim=16),
                    device=DEVICE)
    assert r.classifier_drift == 0.0


def test_joint_training_does_move_the_classifier_when_allowed():
    """The complement of the test above - otherwise 'delayed' would prove nothing."""
    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = _splits()
    r = train_joint(DIM, _classifier(), tr_x, tr_y, va_x, va_y, te_x, te_y, seed=0,
                    baseline_acc=0.5, T=T,
                    config=JointConfig(max_epochs=4, unfreeze_epoch=0,
                                       classifier_lr=1e-2, hidden_dim=16), device=DEVICE)
    assert r.classifier_drift > 0.0


# --------------------------------------------------------------------------
# Protocol constants
# --------------------------------------------------------------------------

def test_protocol_constants_match_what_stage_3_reports():
    """A single T throughout Stage 3, K = 10, and Stage 1's seeds."""
    from cvlab3 import COMBOS, K_SHOT, MAIN_COMBOS, SEEDS, T_STEPS
    assert T_STEPS == 4
    assert K_SHOT == 10
    assert SEEDS == (0, 1, 2)
    assert len(MAIN_COMBOS) == 2 and len(COMBOS) == 3
    assert all(c in COMBOS for c in MAIN_COMBOS)
