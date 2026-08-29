"""
Feature-space visualization: frozen features before vs. after the standard-FM layer.

Continuity with Stage 1's own feature-viz (`Stage_1/Work/05_analysis`, section 6 --
the ResNet-18 vs. DINOv2 side-by-side on FGVC-Aircraft):

* same class subset: 9 systematically-diverse classes, an evenly spaced stride
  through the sorted class list (`pick_diverse_classes`, identical to Stage 1's --
  not `list(range(9))`, which on Aircraft is almost all 737 variants);
* same test images: every test-split image of those 9 classes, no subsampling;
* same style: `cvlab.plotting.apply_style`, the tab10 colour-per-class map, test
  points as dots and prototypes as black-edged stars;
* same normalize-before-projecting lesson: L2-normalize the features *and* the
  prototypes so both sit on the unit sphere before the joint fit. Stage 1
  section 6a showed that concatenating raw features (norm ~15-50) with unit-norm
  prototypes and projecting the mix pins every prototype into one corner.

What is new here:

* one PCA is fit JOINTLY across every view -- original features, FM-transported
  features, and the prototypes -- so all panels share a single projection and a
  single axis scale and are therefore directly comparable;
* panel 2 is the same features after transport through one trained standard-FM
  checkpoint (`cvlab.flow_matching.euler_inference`, `STEPS` Euler steps).

Adding the third panel (after rolled-out FM): once those checkpoints exist,
transport the same `raw_sel` features through one of them and add a single entry
to the `views` dict at the marked spot in `main()`. The joint PCA, shared axes,
colours, legend, and subplot layout all adapt to `len(views)` automatically.

Usage:
    C:\\cvlab_env\\Scripts\\python.exe Stage_2/Work/04_feature_space_viz/feature_space_viz.py
    (optional) --encoder dinov2 --dataset FGVC-Aircraft --k full --seed 0 --steps 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_STAGE1_SRC = _PROJECT_ROOT / "Stage_1" / "src"
if str(_STAGE1_SRC) not in sys.path:
    sys.path.insert(0, str(_STAGE1_SRC))

from cvlab.data import make_kshot_subset  # noqa: E402
from cvlab.features import load_features  # noqa: E402
from cvlab.flow_matching import VelocityNetwork, euler_inference  # noqa: E402
from cvlab.plotting import ArtifactWriter, apply_style  # noqa: E402
from cvlab.prototypes import compute_prototypes, l2_normalize  # noqa: E402

# --- Defaults: the DINOv2 / FGVC-Aircraft / full-data checkpoint, T=12 --------
#     (DINOv2/Aircraft is where standard FM helped most in the classification
#     eval, and it is one of the two encoders in Stage 1's own Aircraft figure.)
DEFAULT_ENCODER = "dinov2"
DEFAULT_DATASET = "FGVC-Aircraft"
DEFAULT_K = "full"
DEFAULT_SEED = 0
DEFAULT_STEPS = 12
N_CLASSES = 9
DEVICE = "cpu"

_MODELS_DIR = _PROJECT_ROOT / "Stage_2" / "results" / "flow_matching_models"
_ROLLED_MODELS_DIR = _PROJECT_ROOT / "Stage_2" / "results" / "rolled_out_fm_models"
_OUT_DIR = _PROJECT_ROOT / "Stage_2" / "results" / "feature_space_viz"

#: Stage 1's colour map, class i in selection order -> COLORS9[i].
COLORS9 = plt.cm.tab10.colors[:N_CLASSES]


def pick_diverse_classes(classes: list[str], n: int = N_CLASSES) -> np.ndarray:
    """Evenly spaced indices through the sorted class list -- identical to Stage 1's
    `05_analysis` helper. Spreads the selection across the whole family range
    instead of clustering on near-duplicate variants."""
    return np.linspace(0, len(classes) - 1, n).astype(int)


def rebuild_prototypes(encoder: str, dataset: str, k: object, seed: int) -> torch.Tensor:
    """The training-time prototypes for a checkpoint, via Stage 1's own functions
    (`make_kshot_subset` + `compute_prototypes`, same seed). For k == "full" this
    reproduces `Stage_1/results/full_prototypes.pt` exactly."""
    train_bundle = load_features(encoder, dataset, "train")
    features = train_bundle.features.to(DEVICE)
    labels = train_bundle.labels.to(DEVICE)
    if k == "full":
        sub_f, sub_y = features, labels
    else:
        sub_f, sub_y = make_kshot_subset(features, labels, int(k), int(seed))
    return compute_prototypes(sub_f, sub_y, train_bundle.n_classes).to(DEVICE)


def load_fm_model(checkpoint_path: Path) -> tuple[VelocityNetwork, dict]:
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = VelocityNetwork(ckpt["feature_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def plot_projection(ax, test_proj, proto_proj, labels_sel, selected_idxs, classes, title):
    """Stage 1's `05_analysis` `plot_projection`, unchanged: dots for test
    features, a big black-edged star for each class prototype."""
    for i, c in enumerate(selected_idxs):
        m = labels_sel == c
        ax.scatter(test_proj[m, 0], test_proj[m, 1], color=COLORS9[i], alpha=0.55, s=22,
                   label=classes[c][:18])
        ax.scatter(*proto_proj[i], color=COLORS9[i], marker="*", s=340,
                   edgecolor="black", linewidth=1.1, zorder=5)
    ax.set_title(title, fontsize=10.5)
    ax.set_xticks([])
    ax.set_yticks([])


def checkpoint_name(encoder: str, dataset: str, k: object, seed: int) -> str:
    slug = dataset.lower().replace("-", "_")
    ktok = "full" if k == "full" else str(int(k))
    return f"{encoder}__{slug}__K{ktok}__seed{seed}.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--k", default=DEFAULT_K, help='5, 10, or "full"')
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args = parser.parse_args()
    k = "full" if str(args.k) == "full" else int(args.k)

    apply_style()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    art = ArtifactWriter(_OUT_DIR, _OUT_DIR)

    # --- Class subset + the test images of those classes (raw features) -------
    test_bundle = load_features(args.encoder, args.dataset, "test")
    selected = pick_diverse_classes(test_bundle.classes, N_CLASSES)
    print(f"selected {args.dataset} classes: {[test_bundle.classes[i] for i in selected]}")

    mask = np.isin(test_bundle.labels.numpy(), selected)
    raw_sel = test_bundle.features[mask].to(DEVICE)
    labels_sel = test_bundle.labels[mask].numpy()
    print(f"{raw_sel.shape[0]} test images across {N_CLASSES} classes; "
          f"raw feature norm mean {raw_sel.norm(dim=1).mean():.2f}")

    prototypes = rebuild_prototypes(args.encoder, args.dataset, k, args.seed)
    proto_sel = prototypes[selected]

    # --- Transport the same raw features through one standard-FM checkpoint ---
    ckpt_path = _MODELS_DIR / checkpoint_name(args.encoder, args.dataset, k, args.seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found -- run Stage_2/Work/02_standard_fm/train_standard_fm.py first."
        )
    model, ckpt = load_fm_model(ckpt_path)
    transported_sel = euler_inference(model, raw_sel, steps=args.steps)
    print(f"transported through {ckpt_path.name} at T={args.steps}; "
          f"feature norm mean {raw_sel.norm(dim=1).mean():.2f} -> "
          f"{transported_sel.norm(dim=1).mean():.2f}")

    # --- Third panel: same features through the matching rolled-out-FM checkpoint.
    #   Rolled-out models live in a separate dir and are only valid at their own
    #   training T, so the filename carries T and must match args.steps. Added
    #   automatically when that checkpoint exists; the joint PCA, shared axes,
    #   colours, legend and layout all key off `views`.
    views: dict[str, torch.Tensor] = {
        "original features": raw_sel,
        "after standard FM": transported_sel,
    }
    slug = args.dataset.lower().replace("-", "_")
    ktok = "full" if k == "full" else str(int(k))
    rolled_path = _ROLLED_MODELS_DIR / f"{args.encoder}__{slug}__K{ktok}__T{args.steps}__seed{args.seed}.pt"
    if rolled_path.exists():
        ro_model, _ = load_fm_model(rolled_path)
        rolled_sel = euler_inference(ro_model, raw_sel, steps=args.steps)
        views["after rolled-out FM"] = rolled_sel
        print(f"transported through {rolled_path.name} at T={args.steps}; "
              f"feature norm mean -> {rolled_sel.norm(dim=1).mean():.2f}")
    else:
        print(f"(no rolled-out checkpoint at {rolled_path.name} -- 2-panel figure)")

    # --- One PCA, fit jointly across every view + the prototypes -------------
    #     L2-normalize everything first (Stage 1 section 6a's lesson): features
    #     and prototypes then share the unit-sphere scale the prototype method
    #     itself operates in.
    normed_views = {name: l2_normalize(v) for name, v in views.items()}
    proto_normed = l2_normalize(proto_sel)

    joint = torch.cat([*normed_views.values(), proto_normed], dim=0).cpu().numpy()
    pca = PCA(n_components=2, random_state=0).fit(joint)
    evr = pca.explained_variance_ratio_

    proto_proj = pca.transform(proto_normed.cpu().numpy())
    view_projs = {name: pca.transform(v.cpu().numpy()) for name, v in normed_views.items()}

    # Shared axis limits so every panel is at the identical scale.
    allpts = np.vstack([*view_projs.values(), proto_proj])
    (xmin, ymin), (xmax, ymax) = allpts.min(0), allpts.max(0)
    xpad, ypad = 0.05 * (xmax - xmin), 0.05 * (ymax - ymin)
    xlim, ylim = (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)

    # --- Plot ---------------------------------------------------------------
    n_panels = len(views)
    fig, axes = plt.subplots(1, n_panels, figsize=(7.0 * n_panels, 6.5), squeeze=False)
    for ax, (name, proj) in zip(axes[0], view_projs.items()):
        plot_projection(ax, proj, proto_proj, labels_sel, selected, test_bundle.classes, name)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
    axes[0][0].legend(fontsize=8, loc="best")

    panel_names = " vs. ".join(view_projs.keys())
    fig.suptitle(
        f"Feature space: {panel_names}  |  {args.encoder} / "
        f"{args.dataset}, K={k}, T={args.steps}\n"
        f"same {N_CLASSES} classes / test images / colours; one jointly-fit PCA "
        f"(explained variance {evr.sum():.1%}); features + prototypes L2-normalized",
        fontsize=12.5, fontweight="bold",
    )
    fig.tight_layout()
    art.figure(fig, "feature_space_viz")

    import pandas as pd
    art.table(
        pd.DataFrame({
            "selection_order": range(N_CLASSES),
            "class_index": selected,
            "class_name": [test_bundle.classes[i] for i in selected],
            "color_hex": [matplotlib.colors.to_hex(c) for c in COLORS9],
        }),
        "feature_space_viz_classes",
    )
    print(f"\nPCA explained variance (2D): {evr.sum():.1%}  "
          f"(PC1 {evr[0]:.1%}, PC2 {evr[1]:.1%})")


if __name__ == "__main__":
    main()
