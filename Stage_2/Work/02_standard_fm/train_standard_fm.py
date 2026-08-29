"""
Standard Flow Matching — the 27-run sweep, as a reviewable script.

This regenerates every artefact under ``Stage_2/results/flow_matching_models/``
(27 checkpoints + ``sweep_summary.json``) from a single script, so the exact
training recipe is on disk rather than living only in a scratch notebook.

What one run is
---------------
For a given ``(encoder, dataset, K, seed)``:

1. Load Stage 1's cached ``train`` features (``cvlab.features.load_features``).
2. Select the training set:
     * ``K in {5, 10}`` -> Stage 1's balanced k-shot subset, drawn with the
       *same* function and seed Stage 1 uses (``cvlab.data.make_kshot_subset``).
       The subset is NOT recomputed here — it is the same call.
     * ``K == "full"``  -> the whole ``train`` split, no subset.
3. Build the target prototypes from that same training set
   (``cvlab.prototypes.compute_prototypes`` — L2-normalised, unit-norm rows).
4. Train a ``VelocityNetwork`` with the standard Flow Matching loss
   (``cvlab.flow_matching.flow_matching_loss``): full-batch, 200 epochs, AdamW,
   lr 1e-3, on CPU.

Seed convention (identical to the archived sweep)
------------------------------------------------
``torch.manual_seed(seed)`` is called immediately before the model is
constructed in *every* run, so weight initialisation is reproducible for the
full-data runs too (this was the bug caught during the first pass — see
``Stage_2/NOTES_FOR_MIRA.md``). The seed's *protocol* role is still what the
checkpoint records in ``seed_role``:
    * ``K in {5, 10}`` -> "subset selection"  (the NumPy k-shot draw)
    * ``K == "full"``  -> "weight init"       (no subset to draw)

Feature / velocity conventions (see Stage_2/CLAUDE.md)
-----------------------------------------------------
Prototypes are unit-norm. Source features ``z_i`` are used RAW in the
interpolation ``z_t = (1 - t) z_i + t * prototype``. This is why DINOv2 losses
run several times higher than ResNet-18's (larger raw feature norm) — an
explainable scale artefact, not a bug, and it means loss values are not
comparable across encoders.

Usage
-----
    C:\\cvlab_env\\Scripts\\python.exe Stage_2/Work/02_standard_fm/train_standard_fm.py

Add ``--dry-run`` to print the run grid and exit without training or writing.
Re-running overwrites the 27 checkpoints and ``sweep_summary.json`` in place.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# --- Locate Stage 1's package (editable install is also fine; this makes the
#     script runnable straight from a plain checkout without relying on it). ---
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_STAGE1_SRC = _PROJECT_ROOT / "Stage_1" / "src"
if str(_STAGE1_SRC) not in sys.path:
    sys.path.insert(0, str(_STAGE1_SRC))

from cvlab.data import make_kshot_subset  # noqa: E402
from cvlab.features import load_features  # noqa: E402
from cvlab.flow_matching import VelocityNetwork, flow_matching_loss  # noqa: E402
from cvlab.prototypes import compute_prototypes  # noqa: E402

# --- Sweep definition ----------------------------------------------------------

#: (encoder cache-prefix, dataset name as understood by cvlab.data / load_features)
COMBOS: tuple[tuple[str, str], ...] = (
    ("resnet18", "DTD"),
    ("resnet18", "FGVC-Aircraft"),
    ("dinov2", "FGVC-Aircraft"),
)
K_VALUES: tuple[object, ...] = (5, 10, "full")
SEEDS: tuple[int, ...] = (0, 1, 2)

EPOCHS = 200
LEARNING_RATE = 1e-3
DEVICE = "cpu"  # no NVIDIA GPU on this machine; the MLP on cached features is CPU-cheap

_OUT_DIR = _PROJECT_ROOT / "Stage_2" / "results" / "flow_matching_models"
_SUMMARY_PATH = _OUT_DIR / "sweep_summary.json"
_SEED_CONVENTION = "NumPy subset seed for K=5/10; torch.manual_seed for full"


def _dataset_slug(dataset: str) -> str:
    """``"FGVC-Aircraft"`` -> ``"fgvc_aircraft"`` — same slug the feature cache uses."""
    return dataset.lower().replace("-", "_")


def _k_token(k: object) -> str:
    """Filename / summary token: ``5`` -> ``"5"``, ``"full"`` -> ``"full"``."""
    return "full" if k == "full" else str(int(k))


def checkpoint_name(encoder: str, dataset: str, k: object, seed: int) -> str:
    return f"{encoder}__{_dataset_slug(dataset)}__K{_k_token(k)}__seed{seed}.pt"


def train_one_run(encoder: str, dataset: str, k: object, seed: int) -> dict:
    """Train a single VelocityNetwork and return ``{state_dict, **metadata}``."""
    train_bundle = load_features(encoder, dataset, "train")
    features = train_bundle.features.to(DEVICE)
    labels = train_bundle.labels.to(DEVICE)
    num_classes = train_bundle.n_classes
    feature_dim = train_bundle.dim

    if k == "full":
        source_features, source_labels = features, labels
        seed_role = "weight init"
    else:
        # Stage 1's exact k-shot subset — same function, same seed. Not recomputed.
        source_features, source_labels = make_kshot_subset(features, labels, int(k), seed)
        seed_role = "subset selection"

    prototypes = compute_prototypes(source_features, source_labels, num_classes).to(DEVICE)
    n_train = int(source_features.shape[0])

    # Reproducible weight init for every run (full-data included).
    torch.manual_seed(seed)
    model = VelocityNetwork(feature_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    start = time.perf_counter()
    final_loss = float("nan")
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        loss = flow_matching_loss(model, source_features, prototypes, source_labels)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    elapsed_s = time.perf_counter() - start

    if not (final_loss == final_loss) or final_loss == float("inf"):  # NaN / inf guard
        raise RuntimeError(
            f"non-finite final loss ({final_loss}) for {encoder}/{dataset}/K{_k_token(k)}/seed{seed}"
        )

    return {
        "model_state_dict": {kk: v.cpu() for kk, v in model.state_dict().items()},
        "encoder": encoder,
        "dataset": dataset,
        "K": k if k == "full" else int(k),
        "seed": seed,
        "seed_role": seed_role,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "n_train": n_train,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "final_train_loss": final_loss,
        "elapsed_s": elapsed_s,
        "device": DEVICE,
    }


def run_grid() -> list[tuple[str, str, object, int]]:
    return [
        (encoder, dataset, k, seed)
        for (encoder, dataset) in COMBOS
        for k in K_VALUES
        for seed in SEEDS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the run grid and exit without training or writing anything",
    )
    args = parser.parse_args()

    grid = run_grid()
    print(f"Standard FM sweep: {len(grid)} runs "
          f"({len(COMBOS)} encoder/dataset x {len(K_VALUES)} K x {len(SEEDS)} seeds), "
          f"{EPOCHS} epochs, AdamW lr={LEARNING_RATE}, device={DEVICE}")
    print(f"Output dir: {_OUT_DIR}")

    if args.dry_run:
        for encoder, dataset, k, seed in grid:
            print(f"  {checkpoint_name(encoder, dataset, k, seed)}")
        return

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(torch.get_num_threads())  # keep whatever the env picked

    runs: list[dict] = []
    sweep_start = time.perf_counter()
    for i, (encoder, dataset, k, seed) in enumerate(grid, start=1):
        name = checkpoint_name(encoder, dataset, k, seed)
        print(f"[{i:2d}/{len(grid)}] {name} ...", end=" ", flush=True)
        record = train_one_run(encoder, dataset, k, seed)

        ckpt_path = _OUT_DIR / name
        torch.save(record, ckpt_path)

        runs.append({
            "encoder": encoder,
            "dataset": dataset,
            "K": _k_token(k),
            "seed": seed,
            "n_train": record["n_train"],
            "final_train_loss": record["final_train_loss"],
            "elapsed_s": record["elapsed_s"],
            "checkpoint": str(Path("Stage_2") / "results" / "flow_matching_models" / name),
        })
        print(f"loss={record['final_train_loss']:.6f}  {record['elapsed_s']:.1f}s")

    total_wall_clock_s = time.perf_counter() - sweep_start
    summary = {
        "device": DEVICE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "seed_convention": _SEED_CONVENTION,
        "runs": runs,
        "total_wall_clock_s": total_wall_clock_s,
    }
    _SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nDone. {len(runs)} checkpoints + {_SUMMARY_PATH.name} written in "
          f"{total_wall_clock_s / 60:.1f} min.")
    # Quick stability read-out: per (encoder, dataset, K) seed spread.
    groups: dict[tuple, list[float]] = {}
    for r in runs:
        groups.setdefault((r["encoder"], r["dataset"], r["K"]), []).append(r["final_train_loss"])
    print("\nfinal_train_loss by group (min / mean / max over seeds):")
    for (enc, ds, k), losses in groups.items():
        lo, hi = min(losses), max(losses)
        mean = sum(losses) / len(losses)
        spread = (hi - lo) / mean if mean else 0.0
        print(f"  {enc:9s} {ds:14s} K{k:<4s}  {lo:.4f} / {mean:.4f} / {hi:.4f}   spread={spread:.1%}")


if __name__ == "__main__":
    main()
