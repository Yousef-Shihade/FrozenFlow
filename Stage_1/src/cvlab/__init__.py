"""
cvlab — shared library for the CVLAB Summer Project, Stage 1.

The project is organised as a **package + notebooks** pair:

* This package holds the *plumbing* — path resolution, dataset loading, the frozen
  encoders, the feature cache, the classifiers, and the plotting style. It is written
  once, imported everywhere, and is the only place a given piece of logic exists.
* The notebooks under ``Stage_1/Work/`` hold the *experiment* — what is being measured,
  why, and what the results mean. They read as a narrative and are what we present.

The split matters for grading: the assignment asks us to reproduce results and explain
the protocol. Duplicated helper code across five notebooks (as in the original Colab
version, where ``load_features`` and the k-shot sampler were pasted into three notebooks
each) makes both harder — a fix applied to one copy silently leaves the others wrong.

Install once, in editable mode, so notebooks can simply ``import cvlab``::

    C:\\cvlab_env\\Scripts\\python.exe -m pip install -e Stage_1
"""

__version__ = "1.0.0"

__all__ = ["paths", "data", "encoders", "features", "probe", "prototypes", "evaluation", "plotting"]
