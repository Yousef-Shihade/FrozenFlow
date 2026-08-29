"""
Accuracy-vs-K plot (deliverable 1): top-1 test accuracy against training-set size,
one subplot per (encoder, dataset), five series on each:

    baseline   ·  Stage 1 prototype classifier (no FM)
    standard FM   at T = 4  and  T = 12
    rolled-out FM at T = 4  and  T = 12

Reads ``Stage_2/results/classification_eval/eval_summary.json`` (produced by
``classification_eval.py``) -- nothing is re-evaluated here. Error bars are the
sample std over the 3 seeds (0 at K=full, which is a single deterministic run for
the baseline; FM still has 3 seeds there).

Colour = method (baseline / standard / rolled-out); line style = T (dashed T=4,
solid T=12). Follows Stage 1's ``05_analysis`` accuracy-vs-K styling.

Usage:
    C:\\cvlab_env\\Scripts\\python.exe Stage_2/Work/03_classification_eval/accuracy_vs_k.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_STAGE1_SRC = _PROJECT_ROOT / "Stage_1" / "src"
if str(_STAGE1_SRC) not in sys.path:
    sys.path.insert(0, str(_STAGE1_SRC))

from cvlab.plotting import apply_style, ArtifactWriter  # noqa: E402

_EVAL_DIR = _PROJECT_ROOT / "Stage_2" / "results" / "classification_eval"

COMBOS: tuple[tuple[str, str], ...] = (
    ("resnet18", "DTD"),
    ("resnet18", "FGVC-Aircraft"),
    ("dinov2", "FGVC-Aircraft"),
)
K_ORDER = ["5", "10", "full"]
K_LABELS = ["5-shot", "10-shot", "full"]

#: (label, training, T, colour, linestyle, marker). training/T None -> the baseline.
SERIES = [
    ("baseline",          None,         None, "#333333", "-",  "o"),
    ("standard FM, T=4",   "standard",   4,   "#4C72B0", "--", "s"),
    ("standard FM, T=12",  "standard",   12,  "#4C72B0", "-",  "s"),
    ("rolled-out FM, T=4", "rolled_out", 4,   "#DD8452", "--", "^"),
    ("rolled-out FM, T=12","rolled_out", 12,  "#DD8452", "-",  "^"),
]


def main() -> None:
    summary_path = _EVAL_DIR / "eval_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found -- run classification_eval.py first.")
    aggregated = json.loads(summary_path.read_text(encoding="utf-8"))["aggregated"]

    # (training, encoder, dataset, K, T) -> aggregated row
    by_key = {(r["training"], r["encoder"], r["dataset"], r["K"], r["T"]): r for r in aggregated}

    apply_style()
    art = ArtifactWriter(_EVAL_DIR, _EVAL_DIR)

    fig, axes = plt.subplots(1, len(COMBOS), figsize=(15.5, 4.8), squeeze=False)
    x = list(range(len(K_ORDER)))

    for ax, (encoder, dataset) in zip(axes[0], COMBOS):
        for label, training, t, colour, ls, marker in SERIES:
            means, errs = [], []
            for k in K_ORDER:
                if training is None:  # baseline: identical across training/T, read from any FM row
                    row = by_key.get(("standard", encoder, dataset, k, 4))
                    means.append(row["acc_baseline_mean"] * 100)
                    errs.append(row["acc_baseline_std"] * 100)
                else:
                    row = by_key.get((training, encoder, dataset, k, t))
                    means.append(row["acc_fm_mean"] * 100)
                    errs.append(row["acc_fm_std"] * 100)
            ax.errorbar(x, means, yerr=errs, color=colour, ls=ls, marker=marker,
                        markersize=6, lw=1.8, capsize=4, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(K_LABELS)
        ax.set_xlabel("training-set size (K per class)")
        ax.set_ylabel("top-1 test accuracy (%)")
        ax.set_title(f"{encoder} / {dataset}")
        ax.legend(fontsize=7.5)

    fig.suptitle("Top-1 test accuracy vs. training-set size: prototype baseline, "
                 "standard FM, rolled-out FM  (error bars = seed std)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    art.figure(fig, "accuracy_vs_k")

    # console echo of the plotted numbers
    for encoder, dataset in COMBOS:
        print(f"\n{encoder} / {dataset}   (top-1 %, mean +/- seed std)")
        print(f"  {'series':20s} " + "  ".join(f"{lbl:>14s}" for lbl in K_LABELS))
        for label, training, t, *_ in SERIES:
            cells = []
            for k in K_ORDER:
                if training is None:
                    r = by_key[("standard", encoder, dataset, k, 4)]
                    cells.append(f"{r['acc_baseline_mean']*100:5.2f} +/- {r['acc_baseline_std']*100:4.2f}")
                else:
                    r = by_key[(training, encoder, dataset, k, t)]
                    cells.append(f"{r['acc_fm_mean']*100:5.2f} +/- {r['acc_fm_std']*100:4.2f}")
            print(f"  {label:20s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
