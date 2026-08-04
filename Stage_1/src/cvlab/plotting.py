"""
Shared plotting style and artefact-saving helpers.

Every figure handed in should look like it came from the same report, so the style is
applied once here rather than being re-tuned per notebook. The save helpers exist so no
notebook has to know where its outputs live — they route to that step's ``plots/`` and
``tables/`` directories and echo what they wrote, which doubles as an execution log.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

__all__ = [
    "DATASET_COLORS",
    "SPLIT_COLORS",
    "METHOD_COLORS",
    "ENCODER_COLORS",
    "apply_style",
    "ArtifactWriter",
]

#: Consistent colour per dataset, used across every step.
DATASET_COLORS: dict[str, str] = {"DTD": "#4C72B0", "FGVC-Aircraft": "#DD8452"}

#: Consistent colour per split.
SPLIT_COLORS: dict[str, str] = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}

#: Consistent colour per classification baseline (steps 03-05).
METHOD_COLORS: dict[str, str] = {"linear_probe": "#4C72B0", "prototype": "#DD8452"}

#: Consistent colour per frozen encoder (steps 02-05).
ENCODER_COLORS: dict[str, str] = {"resnet18": "#4C72B0", "dinov2": "#937860"}


def apply_style() -> None:
    """Apply the project-wide matplotlib style. Call once per notebook."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


class ArtifactWriter:
    """Writes figures and tables into one pipeline step's output directories.

    Example:
        >>> from cvlab.paths import step_dirs
        >>> from cvlab.plotting import ArtifactWriter
        >>> art = ArtifactWriter(*step_dirs("01_data_setup"))
        >>> art.figure(fig, "split_sizes")
        >>> art.table(df, "split_summary")
        >>> art.manifest()
    """

    def __init__(self, plots_dir: Path, tables_dir: Path) -> None:
        self.plots_dir = Path(plots_dir)
        self.tables_dir = Path(tables_dir)

    def figure(self, fig, name: str) -> Path:
        """Save a matplotlib figure as ``plots/<name>.png``."""
        path = self.plots_dir / f"{name}.png"
        fig.savefig(path)
        print(f"  saved plot  -> plots/{path.name}")
        return path

    def table(self, df: pd.DataFrame, name: str, index: bool = False) -> Path:
        """Save a dataframe as ``tables/<name>.csv``."""
        path = self.tables_dir / f"{name}.csv"
        df.to_csv(path, index=index)
        print(f"  saved table -> tables/{path.name}  ({len(df)} rows)")
        return path

    def manifest(self, name: str = "_artifact_manifest") -> pd.DataFrame:
        """Index every artefact written so far, and save it as a table.

        Underscore-prefixed tables (the manifest itself) are excluded so re-running a
        notebook does not accumulate stale self-references.
        """
        rows = [
            {"kind": "plot", "file": f"plots/{p.name}",
             "size_KB": round(p.stat().st_size / 1024, 1)}
            for p in sorted(self.plots_dir.glob("*.png"))
        ] + [
            {"kind": "table", "file": f"tables/{p.name}",
             "size_KB": round(p.stat().st_size / 1024, 1)}
            for p in sorted(self.tables_dir.glob("*.csv"))
            if not p.name.startswith("_")
        ]
        df = pd.DataFrame(rows)
        df.to_csv(self.tables_dir / f"{name}.csv", index=False)
        print(f"{len(df)} artefacts in {self.plots_dir.parent}")
        return df
