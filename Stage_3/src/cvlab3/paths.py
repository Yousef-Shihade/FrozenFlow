"""
Stage 3 path resolution.

Mirrors :mod:`cvlab.paths` and :mod:`cvlabfm.paths`, anchored on ``Stage_3/`` so that
``step_dirs`` writes into ``Stage_3/Work/<step>/``. Importing all three in one notebook is
therefore unambiguous - ``cvlab.paths`` points at Stage 1's inputs, ``cvlabfm.paths`` at
Stage 2's, ``cvlab3.paths`` at Stage 3's outputs.

Stage 1's artefacts are re-exported rather than resolved independently, so there is exactly
one definition of where the cached features live and Stage 3 cannot drift from it.
"""

from __future__ import annotations

from pathlib import Path

from cvlab.paths import FEATURES_ROOT as STAGE1_FEATURES
from cvlab.paths import RESULTS_ROOT as STAGE1_RESULTS
from cvlab.paths import STAGE1_ROOT

__all__ = [
    "STAGE3_ROOT",
    "PROJECT_ROOT",
    "WORK_ROOT",
    "SRC_ROOT",
    "RESULTS_ROOT",
    "DOCS_ROOT",
    "STAGE1_ROOT",
    "STAGE1_FEATURES",
    "STAGE1_RESULTS",
    "step_dirs",
    "describe",
]


def _looks_like_stage3(p: Path) -> bool:
    """A Stage_3 root is the directory holding both ``Work/`` and ``src/cvlab3/``."""
    return (p / "Work").is_dir() and (p / "src" / "cvlab3").is_dir()


def _find_stage3_root() -> Path:
    """Locate ``Stage_3/``.

    Same two-strategy approach as the other stages: walk up from this file first (correct
    for an editable install, where this module lives at ``Stage_3/src/cvlab3/paths.py``),
    then fall back to walking up from the working directory.
    """
    for candidate in Path(__file__).resolve().parents:
        if _looks_like_stage3(candidate):
            return candidate

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if _looks_like_stage3(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not locate the Stage_3 project root (a directory containing both 'Work/' "
        f"and 'src/cvlab3/'). Looked above {Path(__file__).resolve()} and above {cwd}.\n"
        "If you moved the project, reinstall the package with:\n"
        "    python -m pip install -e Stage_3"
    )


# --- Anchors ---------------------------------------------------------------

STAGE3_ROOT: Path = _find_stage3_root()
PROJECT_ROOT: Path = STAGE3_ROOT.parent

WORK_ROOT: Path = STAGE3_ROOT / "Work"
SRC_ROOT: Path = STAGE3_ROOT / "src"
DOCS_ROOT: Path = STAGE3_ROOT / "docs"

# Stage 3's own shared artefacts (the frozen classifiers, trained velocity nets, run tables).
RESULTS_ROOT: Path = STAGE3_ROOT / "results"


def step_dirs(step_name: str, create: bool = True) -> tuple[Path, Path]:
    """Return ``(plots_dir, tables_dir)`` for one Stage 3 pipeline step.

    Args:
        step_name: directory name under ``Stage_3/Work/``, e.g. ``"01_setup_classifier"``.
        create: create the directories if they do not exist.

    Example:
        >>> plots, tables = step_dirs("01_setup_classifier")
    """
    base = WORK_ROOT / step_name
    if not base.is_dir():
        known = sorted(p.name for p in WORK_ROOT.iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"No step directory {step_name!r} under {WORK_ROOT}. Known steps: {known}"
        )

    plots, tables = base / "plots", base / "tables"
    if create:
        plots.mkdir(parents=True, exist_ok=True)
        tables.mkdir(parents=True, exist_ok=True)
    return plots, tables


def describe() -> str:
    """Human-readable summary of every resolved path, with existence flags."""
    rows = [
        ("Stage_3 root", STAGE3_ROOT),
        ("Work root", WORK_ROOT),
        ("Results (Stage 3)", RESULTS_ROOT),
        ("Docs", DOCS_ROOT),
        ("Stage_1 root", STAGE1_ROOT),
        ("Stage_1 feature cache", STAGE1_FEATURES),
        ("Stage_1 results", STAGE1_RESULTS),
    ]
    width = max(len(label) for label, _ in rows)
    return "\n".join(
        f"{label:<{width}} : {path}  {'[ok]' if path.exists() else '[not created yet]'}"
        for label, path in rows
    )


if __name__ == "__main__":
    print(describe())
