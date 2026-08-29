"""
Classification evaluation for the Flow Matching layer.

The question this answers: once a test feature has been transported by the FM
velocity field, does classifying it against Stage 1's prototypes beat Stage 1's
own prototype baseline? Covers both training regimes in one combined table:

  * **standard FM**  (``Stage_2/results/flow_matching_models/``, 27 checkpoints)
    -- one trained model, evaluated at *every* T in {4, 12};
  * **rolled-out FM** (``Stage_2/results/rolled_out_fm_models/``, 54 checkpoints)
    -- T is baked into the weights (backprop through the T-step Euler chain), so
    each checkpoint is evaluated at *only* its own training T (``ckpt["T"]``),
    never the other value.

The one rule that matters here
------------------------------
``euler_inference`` returns the transported feature ``z_hat_T`` **raw** -- it
does not normalise, and it does not compare to anything. Stage 1's prototype
classifier (``cvlab.prototypes.run_prototype_classifier``) classifies like this:

    test_normed  = l2_normalize(test_x)          # unit-norm the query
    similarities = test_normed @ prototypes.T    # cosine sim; prototypes already unit-norm
    preds        = similarities.argmax(dim=1)

:func:`classify_by_cosine` below is those three lines, verbatim, and is used
unchanged for standard FM, rolled-out FM, and the baseline, so every Delta_Acc in
the table is apples-to-apples. Anything that skips the ``l2_normalize`` -- e.g. a
Euclidean ``torch.cdist`` nearest-prototype, fine on unit-norm toy data but wrong
on raw features whose norm is 14-50 while every prototype is norm 1 -- is NOT
Stage 1's convention. The toy test (``test_classification_eval.py``) shows the gap.

Usage
-----
    C:\\cvlab_env\\Scripts\\python.exe Stage_2/Work/03_classification_eval/classification_eval.py

Writes ``Stage_2/results/classification_eval/eval_runs.csv`` + ``eval_summary.json``
(per-run rows + seed-aggregated Delta_Acc), with a ``training`` column
distinguishing ``standard`` from ``rolled_out``. Refuses to write unless the
rolled-out sweep is complete (54 checkpoints); pass ``--allow-partial`` to
override, or ``--no-write`` to smoke-test without touching the output files.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
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
from cvlab.flow_matching import VelocityNetwork, euler_inference  # noqa: E402
from cvlab.prototypes import compute_prototypes, l2_normalize  # noqa: E402

T_VALUES: tuple[int, ...] = (4, 12)
DEVICE = "cpu"

_STD_MODELS_DIR = _PROJECT_ROOT / "Stage_2" / "results" / "flow_matching_models"
_ROLLED_MODELS_DIR = _PROJECT_ROOT / "Stage_2" / "results" / "rolled_out_fm_models"
_OUT_DIR = _PROJECT_ROOT / "Stage_2" / "results" / "classification_eval"

_EXPECTED_STANDARD = 27
_EXPECTED_ROLLED_OUT = 54
_TRAINING_ORDER = {"standard": 0, "rolled_out": 1}


def classify_by_cosine(features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Nearest-prototype by cosine similarity -- Stage 1's exact classify step.

    ``prototypes`` is expected unit-norm (as returned by
    :func:`cvlab.prototypes.compute_prototypes`); ``features`` is normalised here,
    matching ``run_prototype_classifier``.
    """
    proto_norms = prototypes.norm(dim=1)
    if not torch.allclose(proto_norms, torch.ones_like(proto_norms), atol=1e-3):
        raise ValueError(
            "prototypes must be unit-norm (pass the output of compute_prototypes); "
            f"got norms in [{proto_norms.min():.4f}, {proto_norms.max():.4f}]"
        )
    features_normed = l2_normalize(features)
    similarities = features_normed @ prototypes.T
    return similarities.argmax(dim=1)


def evaluate(
    model: VelocityNetwork,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    prototypes: torch.Tensor,
    steps: int,
) -> dict:
    """Transport ``test_x`` through ``model`` in ``steps`` Euler steps, then score.

    Returns FM accuracy, the Stage 1 prototype baseline accuracy on the same raw
    test features, and ``delta_acc = acc_fm - acc_baseline`` -- all with the
    identical :func:`classify_by_cosine` rule.
    """
    model.eval()
    transported = euler_inference(model, test_x, steps=steps)

    fm_preds = classify_by_cosine(transported, prototypes)
    base_preds = classify_by_cosine(test_x, prototypes)

    acc_fm = (fm_preds == test_y).float().mean().item()
    acc_baseline = (base_preds == test_y).float().mean().item()

    return {
        "steps": steps,
        "n_test": int(test_x.shape[0]),
        "acc_fm": acc_fm,
        "acc_baseline": acc_baseline,
        "delta_acc": acc_fm - acc_baseline,
        "mean_norm_before": test_x.norm(dim=1).mean().item(),
        "mean_norm_after": transported.norm(dim=1).mean().item(),
    }


def rebuild_prototypes(encoder: str, dataset: str, k: object, seed: int) -> tuple[torch.Tensor, int]:
    """Rebuild the training-time prototypes by *reusing* Stage 1's functions.

    Same k-shot subset (same seed, same ``make_kshot_subset``) and same
    ``compute_prototypes`` the checkpoint was trained against -- nothing is
    reimplemented. Identical for standard and rolled-out checkpoints (both train
    against exactly these prototypes). Returns ``(prototypes, num_classes)``.
    """
    train_bundle = load_features(encoder, dataset, "train")
    features = train_bundle.features.to(DEVICE)
    labels = train_bundle.labels.to(DEVICE)
    num_classes = train_bundle.n_classes

    if k == "full":
        sub_f, sub_y = features, labels
    else:
        sub_f, sub_y = make_kshot_subset(features, labels, int(k), int(seed))

    prototypes = compute_prototypes(sub_f, sub_y, num_classes).to(DEVICE)
    return prototypes, num_classes


def _load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location=DEVICE, weights_only=False)


def steps_for(ckpt: dict) -> list[int]:
    """Which T values to evaluate this checkpoint at.

    Standard FM: one model, valid at every T in :data:`T_VALUES`.
    Rolled-out FM: T is trained into the weights, so the checkpoint is valid ONLY
    at ``ckpt["T"]`` -- evaluating it at the other T would be meaningless.
    """
    if ckpt.get("training", "standard") == "rolled_out":
        return [int(ckpt["T"])]
    return list(T_VALUES)


def evaluate_checkpoint(path: Path) -> list[dict]:
    """Evaluate one checkpoint at each T that applies to it (see :func:`steps_for`)."""
    ckpt = _load_checkpoint(path)
    training = ckpt.get("training", "standard")
    encoder, dataset = ckpt["encoder"], ckpt["dataset"]
    k, seed = ckpt["K"], ckpt["seed"]
    feature_dim = ckpt["feature_dim"]

    prototypes, num_classes = rebuild_prototypes(encoder, dataset, k, seed)

    test_bundle = load_features(encoder, dataset, "test")
    test_x = test_bundle.features.to(DEVICE)
    test_y = test_bundle.labels.to(DEVICE)

    model = VelocityNetwork(feature_dim).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    rows: list[dict] = []
    for steps in steps_for(ckpt):
        result = evaluate(model, test_x, test_y, prototypes, steps)
        rows.append({
            "checkpoint": path.name,
            "training": training,
            "encoder": encoder,
            "dataset": dataset,
            "K": "full" if k == "full" else str(int(k)),
            "seed": int(seed),
            "T": steps,
            "num_classes": num_classes,
            **result,
        })
    return rows


def _k_sort(k: str) -> int:
    return {"5": 0, "10": 1, "full": 2}.get(k, 99)


def _aggregate(rows: list[dict]) -> list[dict]:
    """Mean / sample-std over seeds for each (training, encoder, dataset, K, T)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["training"], r["encoder"], r["dataset"], r["K"], r["T"])
        groups.setdefault(key, []).append(r)

    out: list[dict] = []
    for (training, encoder, dataset, k, t), grp in groups.items():
        fm = [r["acc_fm"] for r in grp]
        base = [r["acc_baseline"] for r in grp]
        delta = [r["delta_acc"] for r in grp]
        n = len(grp)
        out.append({
            "training": training,
            "encoder": encoder,
            "dataset": dataset,
            "K": k,
            "T": t,
            "n_seeds": n,
            "acc_fm_mean": statistics.mean(fm),
            "acc_fm_std": statistics.stdev(fm) if n > 1 else 0.0,
            "acc_baseline_mean": statistics.mean(base),
            "acc_baseline_std": statistics.stdev(base) if n > 1 else 0.0,
            "delta_acc_mean": statistics.mean(delta),
            "delta_acc_std": statistics.stdev(delta) if n > 1 else 0.0,
        })
    out.sort(key=lambda d: (_TRAINING_ORDER.get(d["training"], 9), d["encoder"],
                            d["dataset"], _k_sort(d["K"]), d["T"]))
    return out


def _print_table(aggregated: list[dict]) -> None:
    print(f"\n{'train':10s} {'encoder':9s} {'dataset':14s} {'K':>4s} {'T':>3s}   "
          f"{'acc_fm':>16s} {'acc_base':>16s} {'Delta_Acc (pp)':>16s}")
    for a in aggregated:
        print(f"{a['training']:10s} {a['encoder']:9s} {a['dataset']:14s} {a['K']:>4s} {a['T']:>3d}   "
              f"{a['acc_fm_mean']*100:7.2f} +/- {a['acc_fm_std']*100:4.2f}   "
              f"{a['acc_baseline_mean']*100:7.2f} +/- {a['acc_baseline_std']*100:4.2f}   "
              f"{a['delta_acc_mean']*100:+7.2f} +/- {a['delta_acc_std']*100:4.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true",
                        help="run the full eval and print the table but do NOT write "
                             "eval_runs.csv / eval_summary.json (smoke test)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="write the summary even if the rolled-out sweep is incomplete")
    args = parser.parse_args()

    std_ckpts = sorted(_STD_MODELS_DIR.glob("*.pt"))
    rolled_ckpts = sorted(_ROLLED_MODELS_DIR.glob("*.pt"))
    checkpoints = std_ckpts + rolled_ckpts
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints in {_STD_MODELS_DIR} or {_ROLLED_MODELS_DIR}. "
            "Run train_standard_fm.py / train_rolled_out_fm.py first."
        )

    print(f"Evaluating {len(std_ckpts)} standard (at T in {list(T_VALUES)}) + "
          f"{len(rolled_ckpts)} rolled-out (each at its own training T) checkpoints, device={DEVICE}")

    rows: list[dict] = []
    skipped: list[str] = []
    start = time.perf_counter()
    for i, path in enumerate(checkpoints, start=1):
        tag = "rolled" if path.parent.name == _ROLLED_MODELS_DIR.name else "std"
        print(f"[{i:2d}/{len(checkpoints)}] ({tag:6s}) {path.name} ...", end=" ", flush=True)
        try:
            ckpt_rows = evaluate_checkpoint(path)
        except Exception as exc:  # e.g. a checkpoint still being written by a live sweep
            print(f"SKIPPED ({type(exc).__name__}: {exc})")
            skipped.append(path.name)
            continue
        rows.extend(ckpt_rows)
        print(", ".join(f"T{r['T']} d={r['delta_acc']*100:+.2f}" for r in ckpt_rows))
    elapsed_s = time.perf_counter() - start

    aggregated = _aggregate(rows)
    _print_table(aggregated)
    if skipped:
        print(f"\n{len(skipped)} checkpoint(s) skipped: {skipped}")

    complete = (len(std_ckpts) == _EXPECTED_STANDARD
                and len(rolled_ckpts) == _EXPECTED_ROLLED_OUT
                and not skipped)

    if args.no_write:
        print(f"\n--no-write set: {len(rows)} run rows computed, nothing written "
              f"({'complete' if complete else 'INCOMPLETE sweep'}).")
        return
    if not complete and not args.allow_partial:
        raise SystemExit(
            f"\nRefusing to write a combined summary: have {len(std_ckpts)}/{_EXPECTED_STANDARD} "
            f"standard + {len(rolled_ckpts)}/{_EXPECTED_ROLLED_OUT} rolled-out checkpoints"
            f"{f', {len(skipped)} skipped' if skipped else ''}. "
            "Wait for the sweep, or pass --allow-partial."
        )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _OUT_DIR / "eval_runs.csv"
    field_order = [
        "checkpoint", "training", "encoder", "dataset", "K", "seed", "T", "num_classes",
        "n_test", "acc_fm", "acc_baseline", "delta_acc",
        "mean_norm_before", "mean_norm_after",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_order)
        writer.writeheader()
        for r in rows:
            writer.writerow({key: r[key] for key in field_order})

    summary = {
        "device": DEVICE,
        "T_values": list(T_VALUES),
        "n_standard_checkpoints": len(std_ckpts),
        "n_rolled_out_checkpoints": len(rolled_ckpts),
        "T_policy": "standard FM: every T in T_values; rolled-out FM: only ckpt['T']",
        "classify_rule": "l2_normalize(features) @ prototypes.T ; argmax  (Stage 1 run_prototype_classifier)",
        "elapsed_s": elapsed_s,
        "runs": rows,
        "aggregated": aggregated,
    }
    (_OUT_DIR / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {csv_path.name} ({len(rows)} rows) and eval_summary.json in {elapsed_s:.1f}s")


if __name__ == "__main__":
    main()
