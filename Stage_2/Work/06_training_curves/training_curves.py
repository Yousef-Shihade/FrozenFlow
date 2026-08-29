"""
Training curves (deliverable 2): per-epoch train-loss for standard FM and
rolled-out FM, on one representative full-data run per (encoder, dataset).

Stage 2's standard-FM sweep (`train_standard_fm.py`) only stored the *final*
loss, so there were no curves to show. This recomputes the full per-epoch
history for:

  * standard FM
  * rolled-out FM, T = 4
  * rolled-out FM, T = 12

for each of the 3 combos, at K = full, seed 0 -- reusing the exact training
logic (`cvlab.flow_matching.flow_matching_loss` / `rolled_out_loss`, same
`VelocityNetwork`, full-batch, 200 epochs, AdamW lr 1e-3, CPU, same seed
convention). Nothing is used from the checkpoints; this is a fresh, self-contained
recompute whose only purpose is the curves.

Output (`Stage_2/results/training_curves/`):
  * `training_curves.png`  -- loss vs epoch, one panel per combo, log-y
  * `training_curves.json` -- every per-epoch history + a stability read-out

Stability is checked automatically: finite throughout, net decrease, no
divergence (post-warmup max not far above the start), and flattening (the last
fifth of training contributes little of the total drop).

Usage:
    C:\\cvlab_env\\Scripts\\python.exe Stage_2/Work/06_training_curves/training_curves.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_STAGE1_SRC = _PROJECT_ROOT / "Stage_1" / "src"
if str(_STAGE1_SRC) not in sys.path:
    sys.path.insert(0, str(_STAGE1_SRC))

from cvlab.features import load_features  # noqa: E402
from cvlab.flow_matching import VelocityNetwork, flow_matching_loss, rolled_out_loss  # noqa: E402
from cvlab.prototypes import compute_prototypes  # noqa: E402
from cvlab.plotting import apply_style, ArtifactWriter  # noqa: E402

COMBOS: tuple[tuple[str, str], ...] = (
    ("resnet18", "DTD"),
    ("resnet18", "FGVC-Aircraft"),
    ("dinov2", "FGVC-Aircraft"),
)
K = "full"
SEED = 0
EPOCHS = 200
LEARNING_RATE = 1e-3
ROLLED_T: tuple[int, ...] = (4, 12)
DEVICE = "cpu"

_OUT_DIR = _PROJECT_ROOT / "Stage_2" / "results" / "training_curves"

# Series key -> (matplotlib colour, label). Colours picked to read in the project style.
SERIES = {
    "standard": ("#4C72B0", "standard FM"),
    "rolled_out_T4": ("#DD8452", "rolled-out, T=4"),
    "rolled_out_T12": ("#937860", "rolled-out, T=12"),
}


def train_history(kind: str, encoder: str, dataset: str, steps: int | None) -> list[float]:
    """Full-data run, recording ``loss.item()`` every epoch. ``kind`` is
    ``"standard"`` or ``"rolled_out"`` (then ``steps`` is required)."""
    bundle = load_features(encoder, dataset, "train")
    features = bundle.features.to(DEVICE)
    labels = bundle.labels.to(DEVICE)
    prototypes = compute_prototypes(features, labels, bundle.n_classes).to(DEVICE)

    torch.manual_seed(SEED)  # same convention as train_standard_fm.py
    model = VelocityNetwork(bundle.dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    history: list[float] = []
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        if kind == "standard":
            loss = flow_matching_loss(model, features, prototypes, labels)
        else:
            loss = rolled_out_loss(model, features, prototypes, labels, steps)
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
    return history


def stability(history: list[float]) -> dict:
    """Automated read-out of the four things deliverable 2 must confirm."""
    finite = all(v == v and v not in (float("inf"), float("-inf")) for v in history)
    start, end, lo = history[0], history[-1], min(history)
    post_warmup_max = max(history[10:]) if len(history) > 10 else max(history)
    tail = history[-EPOCHS // 5:]                       # last fifth
    total_drop = start - end
    tail_drop = tail[0] - tail[-1]
    return {
        "finite": finite,
        "start": start,
        "end": end,
        "min": lo,
        "net_decrease": start - end,
        "decreased": end < start,
        "no_divergence": post_warmup_max <= start * 1.20,
        "post_warmup_max_over_start": post_warmup_max / start if start else float("nan"),
        "tail_fraction_of_drop": (tail_drop / total_drop) if total_drop > 0 else 0.0,
        "flattening": total_drop <= 0 or (tail_drop / total_drop) < 0.15,
    }


def main() -> None:
    apply_style()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    art = ArtifactWriter(_OUT_DIR, _OUT_DIR)

    results: dict[str, dict] = {}
    sweep_start = time.perf_counter()
    for encoder, dataset in COMBOS:
        combo = f"{encoder}__{dataset}"
        results[combo] = {"encoder": encoder, "dataset": dataset, "K": K, "seed": SEED,
                          "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "curves": {}, "stability": {}}
        for key, kind, steps in (
            ("standard", "standard", None),
            *[(f"rolled_out_T{t}", "rolled_out", t) for t in ROLLED_T],
        ):
            t0 = time.perf_counter()
            hist = train_history(kind, encoder, dataset, steps)
            st = stability(hist)
            results[combo]["curves"][key] = hist
            results[combo]["stability"][key] = st
            flags = "".join([
                "F" if st["finite"] else "!",
                "D" if st["decreased"] else "!",
                "N" if st["no_divergence"] else "!",
                "L" if st["flattening"] else "!",
            ])
            print(f"{combo:28s} {key:16s} {hist[0]:8.4f} -> {hist[-1]:8.4f}  "
                  f"[{flags}]  {time.perf_counter() - t0:5.1f}s")

    total_s = time.perf_counter() - sweep_start

    # --- Plot: one panel per combo, log-y ---------------------------------
    fig, axes = plt.subplots(1, len(COMBOS), figsize=(15.5, 4.8), squeeze=False)
    epochs_x = range(1, EPOCHS + 1)
    for ax, (encoder, dataset) in zip(axes[0], COMBOS):
        combo = f"{encoder}__{dataset}"
        for key, (color, label) in SERIES.items():
            ax.plot(epochs_x, results[combo]["curves"][key], color=color, lw=1.6, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("train loss (log scale)")
        ax.set_title(f"{encoder} / {dataset}  (K=full, seed {SEED})")
        ax.legend(fontsize=8)
    fig.suptitle("Training-loss curves: standard FM vs. rolled-out FM  "
                 f"({EPOCHS} epochs, AdamW lr={LEARNING_RATE}, full-batch, CPU)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    art.figure(fig, "training_curves")

    payload = {
        "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "K": K, "seed": SEED,
        "device": DEVICE, "rolled_out_T": list(ROLLED_T),
        "total_wall_clock_s": total_s,
        "combos": results,
    }
    (_OUT_DIR / "training_curves.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # --- Verdict --------------------------------------------------------
    all_ok = True
    print(f"\n{'combo / series':46s} {'finite':>6} {'decr':>5} {'no-div':>7} {'flat':>5}  net drop")
    for combo, r in results.items():
        for key, st in r["stability"].items():
            ok = st["finite"] and st["decreased"] and st["no_divergence"] and st["flattening"]
            all_ok &= ok
            print(f"{combo + ' / ' + key:46s} {str(st['finite']):>6} {str(st['decreased']):>5} "
                  f"{str(st['no_divergence']):>7} {str(st['flattening']):>5}  "
                  f"{st['net_decrease']:.4f}")
    print(f"\n{'ALL STABLE' if all_ok else 'CHECK FAILED — see flags above'} "
          f"({total_s / 60:.1f} min)")


if __name__ == "__main__":
    main()
