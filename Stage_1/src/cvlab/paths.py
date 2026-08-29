"""
Project path resolution.

Replaces the Google-Drive mounting the original Colab notebooks relied on. Every path in
the project derives from one anchor — the ``Stage_1/`` directory — which is located
automatically, so nothing is hardcoded and moving the project breaks nothing.

Why the two dataset roots look asymmetric
-----------------------------------------
``torchvision`` appends its own sub-paths to whatever ``root`` it is given, and the two
datasets nest differently::

    torchvision.datasets.DTD(root=R)           reads  R/dtd/dtd/{images,labels}
    torchvision.datasets.FGVCAircraft(root=R)  reads  R/fgvc-aircraft-2013b/data

Our data on disk is laid out as::

    Stage_1/Data/dtd/dtd/{images,labels,imdb}
    Stage_1/Data/FGVC-Aircraft/fgvc-aircraft-2013b/data/

so DTD wants ``root=Data`` while Aircraft wants ``root=Data/FGVC-Aircraft``. Both are
resolved here once, rather than being rediscovered (painfully) in each notebook.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "STAGE1_ROOT",
    "PROJECT_ROOT",
    "DATA_ROOT",
    "DTD_ROOT",
    "AIRCRAFT_ROOT",
    "WORK_ROOT",
    "SRC_ROOT",
    "FEATURES_ROOT",
    "RESULTS_ROOT",
    "step_dirs",
    "describe",
]


def _looks_like_stage1(p: Path) -> bool:
    """Recognize a Stage_1 root without requiring raw datasets to be present."""
    return (p / "Work").is_dir() and (p / "src" / "cvlab").is_dir()


def _find_stage1_root() -> Path:
    """Locate ``Stage_1/``.

    Primary strategy: walk up from this file. Because the package lives at
    ``Stage_1/src/cvlab/paths.py``, ``parents[2]`` is ``Stage_1`` for a normal editable
    install, and this works no matter what the notebook's working directory is.

    Fallback: walk up from the current working directory, for the case where the package
    has been copied somewhere else (e.g. a site-packages install rather than editable).
    """
    for candidate in Path(__file__).resolve().parents:
        if _looks_like_stage1(candidate):
            return candidate

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if _looks_like_stage1(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not locate the Stage_1 project root (a directory containing 'Work/' and "
        "'src/cvlab/'). "
        f"Looked above {Path(__file__).resolve()} and above {cwd}.\n"
        "If you moved the project, reinstall the package with:\n"
        "    python -m pip install -e Stage_1"
    )


# --- Anchors ---------------------------------------------------------------

STAGE1_ROOT: Path = _find_stage1_root()
PROJECT_ROOT: Path = STAGE1_ROOT.parent

DATA_ROOT: Path = STAGE1_ROOT / "Data"
WORK_ROOT: Path = STAGE1_ROOT / "Work"
SRC_ROOT: Path = STAGE1_ROOT / "src"

# Dataset roots, pre-adjusted for torchvision's nesting (see module docstring).
DTD_ROOT: Path = DATA_ROOT
AIRCRAFT_ROOT: Path = DATA_ROOT / "FGVC-Aircraft"

# Artefacts produced once and consumed by several later steps.
FEATURES_ROOT: Path = STAGE1_ROOT / "features"
RESULTS_ROOT: Path = STAGE1_ROOT / "results"


def step_dirs(step_name: str, create: bool = True) -> tuple[Path, Path]:
    """Return ``(plots_dir, tables_dir)`` for one pipeline step.

    Each step owns its own ``plots/`` and ``tables/`` so figures and CSVs stay beside the
    notebook that produced them.

    Args:
        step_name: directory name under ``Work/``, e.g. ``"01_data_setup"``.
        create: create the directories if they do not exist.

    Example:
        >>> plots, tables = step_dirs("01_data_setup")
    """
    base = WORK_ROOT / step_name
    if not base.is_dir():
        known = sorted(p.name for p in WORK_ROOT.iterdir() if p.is_dir())
        raise FileNotFoundError(f"No step directory {step_name!r} under {WORK_ROOT}. Known steps: {known}")

    plots, tables = base / "plots", base / "tables"
    if create:
        plots.mkdir(parents=True, exist_ok=True)
        tables.mkdir(parents=True, exist_ok=True)
    return plots, tables


def describe() -> str:
    """Human-readable summary of every resolved path, with existence flags."""
    rows = [
        ("Stage_1 root", STAGE1_ROOT),
        ("Data root", DATA_ROOT),
        ("DTD root (-> torchvision)", DTD_ROOT),
        ("Aircraft root (-> torchvision)", AIRCRAFT_ROOT),
        ("Work root", WORK_ROOT),
        ("Feature cache", FEATURES_ROOT),
        ("Results", RESULTS_ROOT),
    ]
    width = max(len(label) for label, _ in rows)
    return "\n".join(
        f"{label:<{width}} : {path}  {'[ok]' if path.exists() else '[not created yet]'}"
        for label, path in rows
    )


if __name__ == "__main__":
    print(describe())
