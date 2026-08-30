"""
Stage 2 path resolution.

Deliberately mirrors :mod:`cvlab.paths`, with one difference that matters: the anchor is the
``Stage_2/`` directory, so ``step_dirs`` writes into ``Stage_2/Work/<step>/`` rather than
Stage 1's. Importing both modules in the same notebook is therefore unambiguous —
``cvlab.paths`` points at Stage 1's inputs, ``cvlabfm.paths`` at Stage 2's outputs.

Stage 1's artefacts are re-exported here (``STAGE1_FEATURES``, ``STAGE1_RESULTS``) rather
than resolved independently, so there is exactly one definition of where the cached features
live and Stage 2 cannot drift from it.
"""

from __future__ import annotations

from pathlib import Path

from cvlab.paths import FEATURES_ROOT as STAGE1_FEATURES
from cvlab.paths import RESULTS_ROOT as STAGE1_RESULTS
from cvlab.paths import STAGE1_ROOT

__all__ = [
    "STAGE2_ROOT",
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


def _looks_like_stage2(p: Path) -> bool:
    """A Stage_2 root is the directory holding both ``Work/`` and ``src/cvlabfm/``."""
    return (p / "Work").is_dir() and (p / "src" / "cvlabfm").is_dir()


def _find_stage2_root() -> Path:
    """Locate ``Stage_2/``.

    Same two-strategy approach as :func:`cvlab.paths._find_stage1_root`: walk up from this
    file first (correct for an editable install, where this module lives at
    ``Stage_2/src/cvlabfm/paths.py``), then fall back to walking up from the working
    directory.
    """
    for candidate in Path(__file__).resolve().parents:
        if _looks_like_stage2(candidate):
            return candidate

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if _looks_like_stage2(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not locate the Stage_2 project root (a directory containing both 'Work/' "
        f"and 'src/cvlabfm/'). Looked above {Path(__file__).resolve()} and above {cwd}.\n"
        "If you moved the project, reinstall the package with:\n"
        "    python -m pip install -e Stage_2"
    )


# --- Anchors ---------------------------------------------------------------

STAGE2_ROOT: Path = _find_stage2_root()
PROJECT_ROOT: Path = STAGE2_ROOT.parent

WORK_ROOT: Path = STAGE2_ROOT / "Work"
SRC_ROOT: Path = STAGE2_ROOT / "src"
DOCS_ROOT: Path = STAGE2_ROOT / "docs"

# Stage 2's own shared artefacts (prototypes, trained velocity nets, run tables).
RESULTS_ROOT: Path = STAGE2_ROOT / "results"


def step_dirs(step_name: str, create: bool = True) -> tuple[Path, Path]:
    """Return ``(plots_dir, tables_dir)`` for one Stage 2 pipeline step.

    Args:
        step_name: directory name under ``Stage_2/Work/``, e.g. ``"01_setup_prototypes"``.
        create: create the directories if they do not exist.

    Example:
        >>> plots, tables = step_dirs("01_setup_prototypes")
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
    """Human-readable summary of every resolved path, with existence flags.

    Includes the Stage 1 paths Stage 2 depends on, so a missing feature cache is visible
    immediately rather than as a confusing error several cells later.
    """
    rows = [
        ("Stage_2 root", STAGE2_ROOT),
        ("Work root", WORK_ROOT),
        ("Results (Stage 2)", RESULTS_ROOT),
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
