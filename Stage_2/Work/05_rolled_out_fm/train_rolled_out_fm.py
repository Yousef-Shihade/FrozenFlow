"""
Rolled-out Flow Matching — the 54-run sweep, as a reviewable script.

Same pattern as ``Stage_2/Work/02_standard_fm/train_standard_fm.py``, for the
rolled-out objective from the Stage 2 spec.

What one run is
---------------
For a given ``(encoder, dataset, K, seed, T)``:

1-3. Identical to the standard-FM script: load Stage 1's cached ``train``
     features, take Stage 1's exact k-shot subset (``make_kshot_subset``, same
     seed) or the whole split for ``K == "full"``, build unit-norm prototypes
     with ``compute_prototypes``.
4.   Train a ``VelocityNetwork`` with ``cvlab.flow_matching.rolled_out_loss`` at
     this run's ``T``: run the full ``T``-step Euler transport from each source
     feature and minimise ``mean((z_hat_T - prototype)**2)``, backpropagating
     through the whole chain of ``T`` velocity predictions. Full-batch, 200
     epochs, AdamW lr 1e-3, CPU.

Unlike standard FM (one model, evaluated at both T), rolled-out bakes T into the
weights: T=4 and T=12 are genuinely different trained models, so the grid is
27 x 2 = 54 runs. Per the spec, the same T is used for training and inference.

"Keep the network architecture and the main training choices fixed" -- so the
architecture, optimiser, lr, epoch count, full-batch regime and seed convention
are exactly the standard-FM script's. The only change is the loss.

Seed convention (identical to the standard-FM sweep)
---------------------------------------------------
``torch.manual_seed(seed)`` immediately before the model is constructed in every
run. ``rolled_out_loss`` has no internal randomness, so a run is fully
deterministic given its seed.

Output
------
``Stage_2/results/rolled_out_fm_models/`` -- 54 checkpoints +
``sweep_summary.json``. Kept separate from the standard-FM checkpoints because a
rolled-out model is only valid at its training T (classification_eval must be
told which T to use; it cannot just evaluate these at both).

Usage
-----
    C:\\cvlab_env\\Scripts\\python.exe Stage_2/Work/05_rolled_out_fm/train_rolled_out_fm.py

``--dry-run`` prints the grid and exits. Re-running overwrites in place.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_STAGE1_SRC = _PROJECT_ROOT / "Stage_1" / "src"
if str(_STAGE1_SRC) not in sys.path:
    sys.path.insert(0, str(_STAGE1_SRC))

from cvlab.data import make_kshot_subset  # noqa: E402
from cvlab.features import load_features  # noqa: E402
from cvlab.flow_matching import VelocityNetwork, rolled_out_loss  # noqa: E402
from cvlab.prototypes import compute_prototypes  # noqa: E402

# --- Sweep definition (COMBOS / K_VALUES / SEEDS / EPOCHS / LR match 02_standard_fm) ---
COMBOS: tuple[tuple[str, str], ...] = (
    ("resnet18", "DTD"),
    ("resnet18", "FGVC-Aircraft"),
    ("dinov2", "FGVC-Aircraft"),
)
K_VALUES: tuple[object, ...] = (5, 10, "full")
SEEDS: tuple[int, ...] = (0, 1, 2)
T_VALUES: tuple[int, ...] = (4, 12)

EPOCHS = 200
LEARNING_RATE = 1e-3
DEVICE = "cpu"

_OUT_DIR = _PROJECT_ROOT / "Stage_2" / "results" / "rolled_out_fm_models"
_SUMMARY_PATH = _OUT_DIR / "sweep_summary.json"
_SEED_CONVENTION = "NumPy subset seed for K=5/10; torch.manual_seed for full (rolled_out_loss is deterministic)"


def _dataset_slug(dataset: str) -> str:
    return dataset.lower().replace("-", "_")


def _k_token(k: object) -> str:
    return "full" if k == "full" else str(int(k))


def checkpoint_name(encoder: str, dataset: str, k: object, seed: int, steps: int) -> str:
    return f"{encoder}__{_dataset_slug(dataset)}__K{_k_token(k)}__T{steps}__seed{seed}.pt"


def train_one_run(encoder: str, dataset: str, k: object, seed: int, steps: int) -> dict:
    """Train a single rolled-out VelocityNetwork and return ``{state_dict, **metadata}``."""
    train_bundle = load_features(encoder, dataset, "train")
    features = train_bundle.features.to(DEVICE)
    labels = train_bundle.labels.to(DEVICE)
    num_classes = train_bundle.n_classes
    feature_dim = train_bundle.dim

    if k == "full":
        source_features, source_labels = features, labels
        seed_role = "weight init"
    else:
        source_features, source_labels = make_kshot_subset(features, labels, int(k), seed)
        seed_role = "subset selection"

    prototypes = compute_prototypes(source_features, source_labels, num_classes).to(DEVICE)
    n_train = int(source_features.shape[0])

    torch.manual_seed(seed)
    model = VelocityNetwork(feature_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    start = time.perf_counter()
    final_loss = float("nan")
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        loss = rolled_out_loss(model, source_features, prototypes, source_labels, steps)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    elapsed_s = time.perf_counter() - start

    if not (final_loss == final_loss) or final_loss == float("inf"):  # NaN / inf guard
        raise RuntimeError(
            f"non-finite final loss ({final_loss}) for "
            f"{encoder}/{dataset}/K{_k_token(k)}/T{steps}/seed{seed}"
        )

    return {
        "model_state_dict": {kk: v.cpu() for kk, v in model.state_dict().items()},
        "encoder": encoder,
        "dataset": dataset,
        "K": k if k == "full" else int(k),
        "seed": seed,
        "seed_role": seed_role,
        "T": steps,
        "training": "rolled_out",
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "n_train": n_train,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "final_train_loss": final_loss,
        "elapsed_s": elapsed_s,
        "device": DEVICE,
    }


def run_grid() -> list[tuple[str, str, object, int, int]]:
    return [
        (encoder, dataset, k, seed, steps)
        for (encoder, dataset) in COMBOS
        for k in K_VALUES
        for steps in T_VALUES
        for seed in SEEDS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the run grid and exit without training or writing anything")
    args = parser.parse_args()

    grid = run_grid()
    print(f"Rolled-out FM sweep: {len(grid)} runs "
          f"({len(COMBOS)} encoder/dataset x {len(K_VALUES)} K x {len(T_VALUES)} T x {len(SEEDS)} seeds), "
          f"{EPOCHS} epochs, AdamW lr={LEARNING_RATE}, device={DEVICE}")
    print(f"Output dir: {_OUT_DIR}")

    if args.dry_run:
        for encoder, dataset, k, seed, steps in grid:
            print(f"  {checkpoint_name(encoder, dataset, k, seed, steps)}")
        return

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    sweep_start = time.perf_counter()
    for i, (encoder, dataset, k, seed, steps) in enumerate(grid, start=1):
        name = checkpoint_name(encoder, dataset, k, seed, steps)
        print(f"[{i:2d}/{len(grid)}] {name} ...", end=" ", flush=True)
        record = train_one_run(encoder, dataset, k, seed, steps)

        torch.save(record, _OUT_DIR / name)
        runs.append({
            "encoder": encoder,
            "dataset": dataset,
            "K": _k_token(k),
            "T": steps,
            "seed": seed,
            "n_train": record["n_train"],
            "final_train_loss": record["final_train_loss"],
            "elapsed_s": record["elapsed_s"],
            "checkpoint": str(Path("Stage_2") / "results" / "rolled_out_fm_models" / name),
        })
        print(f"loss={record['final_train_loss']:.6f}  {record['elapsed_s']:.1f}s")

    total_wall_clock_s = time.perf_counter() - sweep_start
    summary = {
        "device": DEVICE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "T_values": list(T_VALUES),
        "seed_convention": _SEED_CONVENTION,
        "runs": runs,
        "total_wall_clock_s": total_wall_clock_s,
    }
    _SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nDone. {len(runs)} checkpoints + {_SUMMARY_PATH.name} written in "
          f"{total_wall_clock_s / 60:.1f} min.")
    groups: dict[tuple, list[float]] = {}
    for r in runs:
        groups.setdefault((r["encoder"], r["dataset"], r["K"], r["T"]), []).append(r["final_train_loss"])
    print("\nfinal_train_loss by group (min / mean / max over seeds):")
    for (enc, ds, k, t), losses in groups.items():
        lo, hi = min(losses), max(losses)
        mean = sum(losses) / len(losses)
        spread = (hi - lo) / mean if mean else 0.0
        print(f"  {enc:9s} {ds:14s} K{k:<4s} T{t:<2d}  {lo:.4f} / {mean:.4f} / {hi:.4f}   spread={spread:.1%}")


if __name__ == "__main__":
    main()
