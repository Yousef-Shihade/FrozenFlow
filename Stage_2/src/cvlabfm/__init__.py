"""
cvlabfm — the Flow Matching layer for Stage 2.

Stage 1 (:mod:`cvlab`) established two classification baselines on frozen features. Stage 2
inserts a learned transport step between the frozen feature and the prototype classifier: a
velocity network ``v(z, t)`` that moves an image feature toward its class prototype, after
which the *same* cosine-similarity rule from Stage 1 assigns the label.

This package holds only what is new in Stage 2. Everything shared with Stage 1 — the cached
features, prototype construction, k-shot subsets, L2 normalization — is imported from
:mod:`cvlab` rather than reimplemented, which is what keeps the Stage 1 baseline and the
Stage 2 comparison exactly aligned.

Layout
------
``cvlabfm.paths``
    Stage 2 path anchors, mirroring :mod:`cvlab.paths` but rooted at ``Stage_2/``. Also
    re-exports the Stage 1 locations Stage 2 reads from.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
