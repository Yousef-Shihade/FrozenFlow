"""
Synthetic smoke test for the FM classification-evaluation code.

Checks, on toy data with known ground truth, the property the real evaluation
depends on: :func:`classify_by_cosine` is *exactly* Stage 1's prototype classify
rule, and the FM path is scored by that same rule.

Run this before ``classification_eval.py`` touches the 27 real checkpoints:

    C:\\cvlab_env\\Scripts\\python.exe Stage_2/Work/03_classification_eval/test_classification_eval.py
"""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Stage_1" / "src"))

from cvlab.flow_matching import VelocityNetwork, euler_inference, flow_matching_loss
from cvlab.prototypes import compute_prototypes, l2_normalize, run_prototype_classifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classification_eval import classify_by_cosine, evaluate  # noqa: E402


def _toy_dataset(n_per_class=64, dim=16, n_classes=3, noise=0.55, seed=7):
    """Overlapping classes with a deliberate magnitude confound: each example
    points along its class direction but is scaled by a random factor in [1, 6],
    so raw feature norms range ~1-6 while every prototype is unit-norm. The class
    directions are non-orthogonal (pairwise cosine ~0.4-0.6) and the noise is
    large enough that the raw-feature cosine baseline makes some mistakes -- so
    acc_baseline is a non-trivial number, not a free 100%."""
    g = torch.Generator().manual_seed(seed)
    directions = torch.zeros(n_classes, dim)
    directions[0, 0] = 1.0
    directions[1, 0], directions[1, 1] = 0.6, 0.8
    directions[2, 0], directions[2, 2] = 0.6, 0.8
    directions = directions / directions.norm(dim=1, keepdim=True)

    labels = torch.arange(n_classes).repeat_interleave(n_per_class)
    base = directions[labels]
    scale = 1.0 + 5.0 * torch.rand(labels.shape[0], 1, generator=g)
    features = scale * base + noise * torch.randn(labels.shape[0], dim, generator=g)
    return features, labels, n_classes


def main() -> None:
    torch.manual_seed(0)
    features, labels, n_classes = _toy_dataset()

    # Prototypes via Stage 1's function (unit-norm rows).
    prototypes = compute_prototypes(features, labels, n_classes)
    print(f"prototype row-norms: {prototypes.norm(dim=1).tolist()}")
    print(f"raw feature norms: min {features.norm(dim=1).min():.2f}  "
          f"max {features.norm(dim=1).max():.2f}")

    # --- 1. classify_by_cosine IS l2_normalize(x) @ protos.T ; argmax ----------
    manual = (l2_normalize(features) @ prototypes.T).argmax(dim=1)
    got = classify_by_cosine(features, prototypes)
    assert torch.equal(got, manual), "classify_by_cosine diverged from the normalize->dot->argmax rule"
    print("[ok] classify_by_cosine == l2_normalize -> dot -> argmax")

    # --- 2. identical predictions to Stage 1's run_prototype_classifier --------
    stage1 = run_prototype_classifier(features, labels, features, labels, n_classes)
    assert torch.equal(got, stage1.test_preds), "classify_by_cosine disagrees with Stage 1's classifier"
    print(f"[ok] matches run_prototype_classifier predictions exactly "
          f"(baseline acc {stage1.test_acc*100:.2f}%)")

    # --- 3. scale-invariance: the property Stage 1's convention relies on ------
    scaled = features * (0.1 + 10 * torch.rand(features.shape[0], 1))
    assert torch.equal(classify_by_cosine(scaled, prototypes), got), \
        "predictions changed under per-row rescaling - normalization is not doing its job"
    print("[ok] predictions unchanged when test rows are arbitrarily rescaled")

    # --- 4. the guard rejects non-unit prototypes ----------------------------
    try:
        classify_by_cosine(features, prototypes * 3.0)
    except ValueError:
        print("[ok] classify_by_cosine rejects non-unit-norm prototypes")
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for non-unit-norm prototypes")

    # --- 5. FM transport is scored by the same rule and does not hurt --------
    model = VelocityNetwork(features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for _ in range(400):
        optimizer.zero_grad()
        loss = flow_matching_loss(model, features, prototypes, labels)
        loss.backward()
        optimizer.step()

    result = evaluate(model, features, labels, prototypes, steps=12)
    print(f"\nevaluate(steps=12): {result}")

    before = torch.linalg.vector_norm(
        euler_inference(model, features, steps=12) - prototypes[labels], dim=1)
    print(f"mean dist to correct prototype after transport: {before.mean():.4f}")

    assert result["mean_norm_after"] < result["mean_norm_before"], \
        "transport should pull features toward the unit-norm prototypes"
    assert result["acc_fm"] >= 0.90, f"FM accuracy unexpectedly low: {result['acc_fm']:.3f}"
    assert result["delta_acc"] >= -0.05, f"FM materially hurt accuracy vs baseline: delta={result['delta_acc']:.4f}"
    print(f"[ok] acc_fm {result['acc_fm']*100:.2f}%  acc_baseline {result['acc_baseline']*100:.2f}%  "
          f"Delta_Acc {result['delta_acc']*100:+.2f} pp")

    print("\nall checks passed")


if __name__ == "__main__":
    main()
