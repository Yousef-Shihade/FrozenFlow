"""
Shared result aggregation.

The assignment asks for **mean and standard deviation over 3 runs**. That sounds
unambiguous until two different parts of a report compute it two different ways: pandas'
``Series.std()`` defaults to the *sample* standard deviation (``ddof=1``) while numpy's
``ndarray.std()`` defaults to the *population* one (``ddof=0``). On three runs the two
differ by a factor of ``sqrt(3/2) ≈ 1.22`` — enough that an accuracy table and the error
bars on the plot beside it disagree, which is exactly what happened in the original Colab
notebooks (DTD 5-shot: 1.74 in the table, 1.42 in the plot).

Everything in this project therefore goes through :func:`summarize_runs`, which uses the
**sample** standard deviation (``ddof=1``) everywhere. That is the right choice here: the
three seeds are a sample drawn from the population of possible seeds, not the entire
population.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["DDOF", "summarize_runs", "format_mean_std"]

#: Sample standard deviation. Used for every mean +/- std in this project.
DDOF: int = 1


def format_mean_std(mean: float, std: float | None, count: int, decimals: int = 2) -> str:
    """Render one cell of the accuracy table.

    A single run has no spread to report, so it is labelled rather than shown with a
    meaningless ``0.00`` or ``NaN``. The assignment explicitly expects this for the
    full-data prototype result and for zero-shot CLIP.
    """
    if count <= 1 or std is None or (isinstance(std, float) and np.isnan(std)):
        return f"{mean:.{decimals}f} (1 run)"
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def summarize_runs(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str = "test_acc",
    scale: float = 100.0,
    decimals: int = 2,
) -> pd.DataFrame:
    """Aggregate per-run results into mean / std / n, with a formatted display column.

    Args:
        df: one row per run.
        group_cols: columns identifying an experiment, e.g.
            ``["encoder", "dataset", "K", "method"]``.
        value_col: the per-run metric, typically ``"test_acc"`` in [0, 1].
        scale: multiplier applied to mean and std, 100 to report percentages.
        decimals: rounding for the output columns.

    Returns:
        One row per group with ``mean``, ``std``, ``n_runs`` and ``accuracy`` (the
        formatted ``"62.84 +/- 0.20"`` string).
    """
    grouped = df.groupby(group_cols, dropna=False)[value_col]
    out = grouped.agg(
        mean="mean",
        std=lambda s: s.std(ddof=DDOF),
        n_runs="size",
    ).reset_index()

    out["mean"] = (out["mean"] * scale).round(decimals)
    out["std"] = (out["std"] * scale).round(decimals)
    out["accuracy"] = [
        format_mean_std(m, s, n, decimals)
        for m, s, n in zip(out["mean"], out["std"], out["n_runs"])
    ]
    return out
