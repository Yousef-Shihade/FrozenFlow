"""
cvlab — shared library for Stage 1 of the Flow Matching as a Layer project.

The project is organised as a **package + notebooks** pair:

* This package holds the *plumbing* — path resolution, dataset loading, the frozen
  encoders, the feature cache, the classifiers, and the plotting style. It is written
  once, imported everywhere, and is the only place a given piece of logic exists.
* The notebooks under ``Stage_1/Work/`` hold the *experiment* — what is being measured,
  why, and what the results mean. They read as a narrative and are what we present.

The split matters for reproducibility: results need to be reproducible and the protocol
needs to be explainable. Duplicating helper code across five notebooks — pasting
``load_features`` and the k-shot sampler into three of them each — makes both harder, since
a fix applied to one copy silently leaves the others wrong.

Install once, in editable mode, so notebooks can simply ``import cvlab``::

    C:\\cvlab_env\\Scripts\\python.exe -m pip install -e Stage_1
"""

__version__ = "1.0.0"

__all__ = ["paths", "data", "encoders", "features", "probe", "prototypes", "evaluation", "plotting"]
