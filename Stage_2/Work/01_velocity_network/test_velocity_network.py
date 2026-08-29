"""Synthetic smoke test for the Stage 2 velocity network."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Stage_1" / "src"))

from cvlab.flow_matching import (
    VelocityNetwork,
    euler_inference,
    euler_inference_trajectory,
    flow_matching_loss,
)


def main() -> None:
    torch.manual_seed(7)
    feature_dim = 16
    prototypes = torch.zeros(3, feature_dim)
    prototypes[0, 0] = 8.0
    prototypes[1, 1] = 8.0
    prototypes[2, 2] = 8.0

    labels = torch.arange(3).repeat_interleave(32)
    source_features = prototypes[labels] + 1.5 * torch.randn(labels.shape[0], feature_dim)
    model = VelocityNetwork(feature_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    for _ in range(400):
        optimizer.zero_grad()
        loss = flow_matching_loss(model, source_features, prototypes, labels)
        loss.backward()
        optimizer.step()

    transported = euler_inference(model, source_features, steps=100)
    correct_prototypes = prototypes[labels]
    before = torch.linalg.vector_norm(source_features - correct_prototypes, dim=1)
    after = torch.linalg.vector_norm(transported - correct_prototypes, dim=1)
    predicted_labels = torch.cdist(transported, prototypes).argmin(dim=1)

    print(f"before mean distance: {before.mean().item():.4f}")
    print(f"after mean distance:  {after.mean().item():.4f}")
    print(f"correct nearest-prototype rate: {(predicted_labels == labels).float().mean().item():.1%}")
    assert after.mean() < before.mean()
    assert (predicted_labels == labels).all()

    # --- Trajectory variant: every Euler state, not just the last -------------
    for steps in (4, 100):
        trajectory = euler_inference_trajectory(model, source_features, steps=steps)
        assert trajectory.shape == (steps + 1, *source_features.shape), trajectory.shape
        assert trajectory.shape[0] == steps + 1
        assert torch.equal(trajectory[0], source_features)
        assert torch.equal(
            trajectory[-1], euler_inference(model, source_features, steps=steps)
        )
        step_move = torch.linalg.vector_norm(
            trajectory[1:] - trajectory[:-1], dim=2
        ).mean().item()
        print(f"trajectory steps={steps:3d}: {trajectory.shape[0]} states "
              f"(== steps+1), start==input, end==euler_inference, "
              f"mean per-step move {step_move:.4f}")


if __name__ == "__main__":
    main()