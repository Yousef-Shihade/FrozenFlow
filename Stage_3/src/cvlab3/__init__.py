"""
cvlab3 - Stage 3: a flow-matching transformation in front of a frozen linear classifier.

Stage 1 trained a linear probe on frozen encoder features. Stage 3 freezes that exact
classifier and learns a flow that reshapes the features feeding into it::

    z --[ FM, T Euler steps ]--> z_hat --[ frozen W, b ]--> logits

What this package adds is only what did not exist before: the frozen-classifier wrapper,
the identity initialisation the setup requires, and Stage 3's two training objectives.
The velocity network and the Euler integrator are imported from Stage 2 (``cvlabfm``) and
the features, k-shot subsets, probe and aggregation helpers from Stage 1 (``cvlab``), so a
fix applied once cannot drift between stages.
"""

from cvlab3.classifier import (
    COMBOS,
    K_SHOT,
    MAIN_COMBOS,
    SEEDS,
    T_STEPS,
    FrozenClassifier,
    identity_flow,
    transform,
    transform_and_classify,
)

__all__ = [
    "COMBOS",
    "K_SHOT",
    "MAIN_COMBOS",
    "SEEDS",
    "T_STEPS",
    "FrozenClassifier",
    "identity_flow",
    "transform",
    "transform_and_classify",
]
