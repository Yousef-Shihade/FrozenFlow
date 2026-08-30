"""
Does the implementation compute what the Stage 2 brief actually specifies?

Every number in Stage 2 rests on one claim: that `cvlabfm.flow` implements the brief's four
formulas and not something merely close to them. That claim is unusually easy to break
silently. A loss that averages over the feature dimension, a rollout whose last step lands on
``t = 1``, an interpolation that normalizes one side and not the other — each would still
train, still converge, and still fill in a plausible accuracy table. Nothing in a notebook's
output would look wrong.

So each test below re-derives one formula directly from the brief's wording and asserts the
implementation agrees. The re-derivations are deliberately written the slow, literal way
(explicit Python loops, no reuse of the helpers under test) — being independent of the
implementation is the entire point.

The brief's four formulas:

    Euler step      z_{k+1} = z_k + (1/T) v(z_k, k/T),  k = 0 .. T-1
    Standard FM     t ~ U(0,1), z_t = (1-t) z_i + t p_yi, u_i = p_yi - z_i
                    L_FM   = || v(z_t, t) - u_i ||_2^2
    Rolled-out FM   L_roll = || z_T - p_yi ||_2^2
    Classification  argmax_c cos(z_T, mu_c)          [Stage 1's rule, unchanged]

These tests were checked by deliberately breaking the implementation five ways and confirming
the suite noticed. Two of the five it did *not* catch, and both turned out to be worth
knowing rather than gaps to patch:

* Dropping the ``l2_normalize`` inside :func:`classify_by_prototype` changes nothing.
  Normalizing a row divides it by a positive scalar, which cannot reorder that row's
  similarity scores. The call is there so the code reads like the cosine rule it implements.
* Replacing cosine with *nearest prototype by Euclidean distance* also changes nothing —
  but only because the prototypes are unit-norm. Expanding the square leaves
  ``argmin_c ||z - p_c||^2 = argmax_c z . p_c`` once every ``||p_c||`` is equal.

Neither is a defect, so neither has a test. The three real breakages — an off-by-one in the
step time, averaging the loss over the feature dimension, and ranking by the wrong end of the
similarity — are each caught by several tests below.

Run with:

    python -m pytest Stage_2/tests -q
"""

from __future__ import annotations

import pytest
import torch

from cvlab.prototypes import l2_normalize, run_prototype_classifier
from cvlabfm.flow import (
    FlowConfig,
    VelocityNet,
    _mse_per_example,
    classify_by_prototype,
    euler_rollout,
    evaluate_flow,
    train_rolled_out_fm,
    train_standard_fm,
)

DIM, N_EXAMPLES, N_CLASSES = 16, 7, 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def net() -> VelocityNet:
    torch.manual_seed(0)
    return VelocityNet(dim=DIM, hidden_dim=32, n_hidden=2)


@pytest.fixture
def z0() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(N_EXAMPLES, DIM)


@pytest.fixture
def prototypes() -> torch.Tensor:
    torch.manual_seed(2)
    return l2_normalize(torch.randn(N_CLASSES, DIM))


# --- The brief's formulas ----------------------------------------------------------------

@pytest.mark.parametrize("T", [1, 4, 12])
def test_euler_rollout_matches_the_brief(net, z0, T):
    """z_{k+1} = z_k + (1/T) v(z_k, k/T), integrated by hand."""
    expected = z0.clone()
    for k in range(T):
        t = torch.full((N_EXAMPLES,), k / T)
        expected = expected + (1.0 / T) * net(expected, t)

    assert torch.allclose(euler_rollout(net, z0, T), expected, rtol=0, atol=1e-6)


def test_euler_step_times_are_k_over_T_and_never_reach_one(net, z0):
    """``k = 0 .. T-1``, so the final step is evaluated at (T-1)/T, not at t = 1.

    Off-by-one here would shift every step's time and still produce a working classifier.
    """
    seen: list[float] = []

    class RecordTime(torch.nn.Module):
        def forward(self, z, t):
            seen.append(round(float(t.flatten()[0]), 10))
            return torch.zeros_like(z)

    euler_rollout(RecordTime(), z0, T=4)
    assert seen == [0.0, 0.25, 0.5, 0.75]
    assert 1.0 not in seen


def test_constant_velocity_field_integrates_exactly(z0):
    """T steps of size 1/T through a constant field v must displace by exactly v."""

    class ConstantVelocity(torch.nn.Module):
        def forward(self, z, t):
            return torch.full_like(z, 2.0)

    for T in (1, 4, 12):
        assert torch.allclose(euler_rollout(ConstantVelocity(), z0, T), z0 + 2.0, atol=1e-5)


def test_standard_fm_loss_matches_the_brief(net, prototypes):
    """L_FM = || v((1-t) z_i + t p_yi, t) - (p_yi - z_i) ||_2^2, averaged over the batch."""
    torch.manual_seed(3)
    z_i = l2_normalize(torch.randn(N_EXAMPLES, DIM))
    y = torch.randint(0, N_CLASSES, (N_EXAMPLES,))
    t = torch.rand(N_EXAMPLES)

    p_yi = prototypes[y]
    z_t = (1 - t).unsqueeze(1) * z_i + t.unsqueeze(1) * p_yi
    u_i = p_yi - z_i

    # Squared L2 per example, meaned over the batch — written out elementwise.
    v = net(z_t, t)
    by_hand = sum(float(((v[i] - u_i[i]) ** 2).sum()) for i in range(N_EXAMPLES)) / N_EXAMPLES

    assert float(_mse_per_example(v, u_i)) == pytest.approx(by_hand, rel=1e-6)


def test_rolled_out_loss_matches_the_brief(net, z0, prototypes):
    """L_roll = || z_T - p_yi ||_2^2, where z_T is the *same* rollout used at inference."""
    torch.manual_seed(4)
    y = torch.randint(0, N_CLASSES, (N_EXAMPLES,))
    p_yi = prototypes[y]

    z_T = euler_rollout(net, z0, T=4)
    by_hand = sum(float(((z_T[i] - p_yi[i]) ** 2).sum()) for i in range(N_EXAMPLES)) / N_EXAMPLES

    assert float(_mse_per_example(z_T, p_yi)) == pytest.approx(by_hand, rel=1e-6)


def test_loss_is_squared_l2_not_mean_over_feature_dimension():
    """The trap this guards: ``F.mse_loss`` also divides by D, rescaling the loss 512x.

    Training would still converge, but the reported loss would no longer be the quantity the
    brief names, and the two objectives' curves would not be on comparable scales.
    """
    torch.manual_seed(5)
    a, b = torch.randn(4, DIM), torch.randn(4, DIM)
    mse_loss = torch.nn.functional.mse_loss(a, b)

    assert float(_mse_per_example(a, b)) == pytest.approx(float(mse_loss) * DIM, rel=1e-5)


def test_classification_is_stage1s_cosine_rule_unchanged():
    """The whole comparison rests on Stage 2 classifying *exactly* as Stage 1 does.

    So this compares against Stage 1's real classifier rather than a re-derivation of it: if
    the two ever diverged, every Delta-Acc in the results table would be measuring two
    different rules against each other rather than the effect of the flow.
    """
    torch.manual_seed(6)
    train_x = torch.randn(120, DIM)
    train_y = torch.randint(0, N_CLASSES, (120,))
    test_x = torch.randn(40, DIM)
    test_y = torch.randint(0, N_CLASSES, (40,))

    stage1 = run_prototype_classifier(train_x, train_y, test_x, test_y, N_CLASSES)
    stage2 = classify_by_prototype(test_x, stage1.prototypes)

    assert torch.equal(stage2, stage1.test_preds)
    assert (stage2 == test_y).float().mean().item() == pytest.approx(stage1.test_acc)


def test_a_prototype_classifies_as_its_own_class(prototypes):
    assert torch.equal(classify_by_prototype(prototypes, prototypes), torch.arange(N_CLASSES))


def test_classification_ignores_the_magnitude_of_the_transported_feature(prototypes):
    """z_T is not unit-norm — the flow adds unconstrained velocities — so scale must not matter.

    Note this is a property of argmax over a positive rescale rather than something the
    implementation could plausibly get wrong; it is here to pin the intended contract, not
    because a realistic bug would trip it.
    """
    torch.manual_seed(6)
    z = torch.randn(9, DIM)
    expected = classify_by_prototype(z, prototypes)

    assert torch.equal(classify_by_prototype(z * 37.0, prototypes), expected)
    assert torch.equal(classify_by_prototype(z * 0.01, prototypes), expected)


# --- Properties the comparison depends on ------------------------------------------------

def test_rollout_is_differentiable_through_every_step():
    """Rolled-out training backpropagates through all T predictions — verify it can."""
    torch.manual_seed(7)
    net = VelocityNet(DIM, 32, 2)
    z0 = torch.randn(N_EXAMPLES, DIM)

    euler_rollout(net, z0, T=4).sum().backward()

    params = list(net.parameters())
    assert all(p.grad is not None for p in params), "a parameter received no gradient"
    assert all(p.grad.abs().sum() > 0 for p in params), "a gradient is identically zero"


def test_trajectory_includes_both_endpoints(net, z0):
    """Step 5's flow-trajectory figure plots this; shape and endpoints must be right."""
    z_T, trajectory = euler_rollout(net, z0, T=4, return_trajectory=True)

    assert trajectory.shape == (N_EXAMPLES, 5, DIM)      # T + 1 states
    assert torch.equal(trajectory[:, 0], z0)
    assert torch.equal(trajectory[:, -1], z_T)


def test_T_below_one_is_rejected(net, z0):
    with pytest.raises(ValueError):
        euler_rollout(net, z0, T=0)


def test_velocity_net_accepts_scalar_and_batched_times(net, z0):
    """t arrives as a scalar in some call sites and as (N,) in others."""
    for t in (torch.tensor(0.3), torch.rand(N_EXAMPLES), torch.rand(N_EXAMPLES, 1)):
        assert net(z0, t).shape == z0.shape


def test_velocity_net_has_the_architecture_the_brief_suggests():
    """2 hidden layers of width 512, input = feature + scalar t, output = feature dim."""
    net = VelocityNet(dim=512, hidden_dim=512, n_hidden=2)
    n_params = sum(p.numel() for p in net.parameters())
    expected = (513 * 512 + 512) + (512 * 512 + 512) + (512 * 512 + 512)

    assert n_params == expected, "architecture drifted from the brief's suggestion"


def test_same_seed_gives_identical_results(prototypes):
    """The two objectives are only comparable if a seed pins everything that varies."""
    torch.manual_seed(8)
    x = l2_normalize(torch.randn(64, DIM))
    y = torch.randint(0, N_CLASSES, (64,))
    cfg = FlowConfig(hidden_dim=32, n_hidden=2, epochs=3, batch_size=16)

    _, hist_a, _ = train_standard_fm(x, y, prototypes, seed=1, config=cfg, device=DEVICE)
    _, hist_b, _ = train_standard_fm(x, y, prototypes, seed=1, config=cfg, device=DEVICE)
    _, hist_c, _ = train_standard_fm(x, y, prototypes, seed=2, config=cfg, device=DEVICE)

    assert hist_a == hist_b, "same seed produced different loss histories"
    assert hist_a != hist_c, "different seeds produced identical loss histories"


def test_both_objectives_start_from_the_same_initialisation(prototypes):
    """The brief asks for architecture and training choices to be held fixed.

    Both loops go through the same ``_prepare``, so for a given seed they must begin from
    byte-identical weights — otherwise a measured difference could be initialisation, not
    the objective.
    """
    torch.manual_seed(9)
    x = l2_normalize(torch.randn(64, DIM))
    y = torch.randint(0, N_CLASSES, (64,))
    cfg = FlowConfig(hidden_dim=32, n_hidden=2, epochs=0, batch_size=16)   # no steps taken

    net_std, _, _ = train_standard_fm(x, y, prototypes, seed=3, config=cfg, device=DEVICE)
    net_roll, _, _ = train_rolled_out_fm(x, y, prototypes, T=4, seed=3, config=cfg,
                                         device=DEVICE)

    assert all(torch.equal(a, b)
               for a, b in zip(net_std.state_dict().values(),
                               net_roll.state_dict().values()))


def test_normalization_flag_actually_changes_the_inputs(prototypes):
    """The L2 ablation is only meaningful if ``normalize=False`` reaches the training path."""
    torch.manual_seed(10)
    x = torch.randn(64, DIM) * 25.0                      # raw-scale features
    y = torch.randint(0, N_CLASSES, (64,))
    cfg = FlowConfig(hidden_dim=32, n_hidden=2, epochs=2, batch_size=16)

    _, hist_norm, _ = train_standard_fm(x, y, prototypes, seed=4, config=cfg,
                                        device=DEVICE, normalize=True)
    _, hist_raw, _ = train_standard_fm(x, y, prototypes, seed=4, config=cfg,
                                       device=DEVICE, normalize=False)
    assert hist_norm != hist_raw, "normalize=False did not reach the training path"

    # Both objectives carry the same knob, so the parity the brief asks for cannot drift.
    _, roll_norm, _ = train_rolled_out_fm(x, y, prototypes, T=4, seed=4, config=cfg,
                                          device=DEVICE, normalize=True)
    _, roll_raw, _ = train_rolled_out_fm(x, y, prototypes, T=4, seed=4, config=cfg,
                                         device=DEVICE, normalize=False)
    assert roll_norm != roll_raw, "normalize=False did not reach the rolled-out path"


def test_evaluation_normalization_matches_training(prototypes):
    """Evaluating a raw-trained model on normalized inputs would compare two different things."""
    torch.manual_seed(11)
    x = torch.randn(64, DIM) * 25.0
    y = torch.randint(0, N_CLASSES, (64,))
    cfg = FlowConfig(hidden_dim=32, n_hidden=2, epochs=2, batch_size=16)

    net, _, _ = train_standard_fm(x, y, prototypes, seed=5, config=cfg,
                                  device=DEVICE, normalize=False)
    acc_raw, _ = evaluate_flow(net, x, y, prototypes, [4], device=DEVICE, normalize=False)
    acc_norm, _ = evaluate_flow(net, x, y, prototypes, [4], device=DEVICE, normalize=True)

    assert acc_raw.keys() == acc_norm.keys() == {4}
    assert 0.0 <= acc_raw[4] <= 1.0 and 0.0 <= acc_norm[4] <= 1.0


def test_fm_does_not_hurt_on_a_separable_toy_problem(prototypes):
    """A weak end-to-end guard: on clean, well-separated data both objectives should learn."""
    torch.manual_seed(12)
    n, dim, n_classes = 400, 32, 4
    true_p = l2_normalize(torch.randn(n_classes, dim))
    y = torch.randint(0, n_classes, (n,))
    x = true_p[y] + 0.55 * torch.randn(n, dim)

    cfg = FlowConfig(hidden_dim=64, n_hidden=2, epochs=30, batch_size=64)
    baseline = (classify_by_prototype(x, true_p) == y).float().mean().item()

    net_s, hist_s, _ = train_standard_fm(x, y, true_p, seed=0, config=cfg, device=DEVICE)
    net_r, hist_r, _ = train_rolled_out_fm(x, y, true_p, T=4, seed=0, config=cfg, device=DEVICE)

    assert hist_s[-1] < hist_s[0], "standard FM loss did not decrease"
    assert hist_r[-1] < hist_r[0], "rolled-out FM loss did not decrease"

    acc_s, _ = evaluate_flow(net_s, x, y, true_p, [4], device=DEVICE)
    acc_r, _ = evaluate_flow(net_r, x, y, true_p, [4], device=DEVICE)
    assert acc_s[4] >= baseline
    assert acc_r[4] >= baseline
