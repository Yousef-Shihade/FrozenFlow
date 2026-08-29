"""
Flow trajectories: the path a feature takes through the FM layer, and its reverse.

Small multiples, because a single 2D projection cannot do several paths justice at
this explained-variance level. Every panel uses the *same construction* as
`feature_space_viz.py` -- L2-normalize every point onto the unit sphere, then one
PCA fit jointly across that panel's points (its trajectory + all 9 selected
prototypes + the relevant class's real test features) -- fit per panel so the
trajectory drives the projection instead of a background cloud.

**Forward flow (`flow_trajectories_forward.png`).** 4 individual real test images,
one per class, transported with `cvlab.flow_matching.euler_inference_trajectory`
so every intermediate Euler state is kept. Per panel: the class's test features
(faint), the 8 sibling prototypes (small grey stars) and the class's own
prototype (large coloured star), and the path -- a dot at every Euler step, the
quarter points labelled with their t value, the **t=0.5** midpoint as an open
diamond with a bold callout (the professor asked to see the midpoint, not just
the endpoint), the original feature (t=0) as a circle and the transported feature
(t=1) as a square, both black-edged. A dotted connector from t=1 to the own
prototype is labelled with the actual cosine at t=0 and t=1.

What it shows for the current standard-FM checkpoint: the path leaves the class
cloud and contracts, but cos(state, own prototype) *falls* along the way
(~0.8 -> ~0.4). Transport is helping classification by shrinking the sibling
prototypes' similarity faster than the true one's, not by landing on the
prototype -- so the square does not sit on the star.

**Reverse flow (`flow_trajectories_reverse.png`).** Starting *from* a class
prototype at t=1, integrate the same field with the velocity negated,
`w <- w - dt * v(w, t)`, for the same number of steps, down to t=0, overlaid on
that class's real test features. Panel title reports cos(point, mean real
feature) at the prototype vs. at the reverse endpoint -- if reversing traces back
toward where real data lives, that number goes up.

Usage:
    C:\\cvlab_env\\Scripts\\python.exe Stage_2/Work/04_feature_space_viz/flow_trajectories.py
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
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_STAGE1_SRC = _PROJECT_ROOT / "Stage_1" / "src"
for p in (str(_STAGE1_SRC), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cvlab.features import load_features  # noqa: E402
from cvlab.flow_matching import euler_inference_trajectory  # noqa: E402
from cvlab.plotting import ArtifactWriter, apply_style  # noqa: E402
from cvlab.prototypes import l2_normalize  # noqa: E402

from feature_space_viz import (  # noqa: E402
    COLORS9,
    DEFAULT_DATASET,
    DEFAULT_ENCODER,
    DEFAULT_K,
    DEFAULT_SEED,
    N_CLASSES,
    checkpoint_name,
    load_fm_model,
    pick_diverse_classes,
    rebuild_prototypes,
)

DEFAULT_STEPS = 12         # matches feature_space_viz.py; step count for BOTH directions
DEVICE = "cpu"
CLOUD_MAX = 40             # real test features per class drawn behind a path

_MODELS_DIR = _PROJECT_ROOT / "Stage_2" / "results" / "flow_matching_models"
_OUT_DIR = _PROJECT_ROOT / "Stage_2" / "results" / "feature_space_viz"

# Positions within the 9 selected classes (feature_space_viz.pick_diverse_classes).
# For DINOv2/Aircraft the 9 are:
#   0:707-320  1:747-300  2:A319  3:BAE 146-200  4:Cessna 525
#   5:DHC-8-300  6:Falcon 2000  7:MD-80  8:Yak-42
FORWARD_CLASS_SLOTS = (4, 5, 6, 0)   # 4 different manufacturer families
REVERSE_CLASS_SLOTS = (4, 5)         # Cessna 525, DHC-8-300


@torch.no_grad()
def reverse_trajectory_from_prototype(model, prototypes: torch.Tensor, steps: int) -> torch.Tensor:
    """Integrate ``w <- w - dt * v(w, t)`` from the prototype (t=1) back to t=0.

    Mirrors :func:`euler_inference_trajectory` but negates the step and runs time
    backwards. Returns ``(steps + 1, N, D)``: row 0 is ``prototypes`` unchanged,
    row ``k`` is at time ``t = 1 - k / steps``, row ``-1`` is the t=0 endpoint.
    """
    was_training = model.training
    model.eval()
    w = prototypes.clone()
    states = [w.clone()]
    dt = 1.0 / steps
    for k in range(steps):
        t = 1.0 - k * dt
        t_vec = torch.full((w.shape[0],), t, device=w.device, dtype=w.dtype)
        w = w - dt * model(w, t_vec)
        states.append(w)
    model.train(was_training)
    return torch.stack(states, dim=0)


def _fit_panel_pca(*parts: torch.Tensor) -> PCA:
    """One PCA per panel, fit on the L2-normalized concatenation of that panel's points."""
    joint = torch.cat([l2_normalize(p.reshape(-1, p.shape[-1])) for p in parts], dim=0)
    return PCA(n_components=2, random_state=0).fit(joint.cpu().numpy())


def _proj(pca: PCA, x: torch.Tensor) -> np.ndarray:
    flat = x.reshape(-1, x.shape[-1])
    return pca.transform(l2_normalize(flat).cpu().numpy()).reshape(*x.shape[:-1], 2)


def _square_limits(*arrays, margin=0.12):
    pts = np.vstack([a.reshape(-1, 2) for a in arrays])
    (xmin, ymin), (xmax, ymax) = pts.min(0), pts.max(0)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = (1 + margin) * max(xmax - xmin, ymax - ymin) / 2
    return (cx - half, cx + half), (cy - half, cy + half)


def _draw_steps(ax, path2d, t_values, color):
    """Dot at every Euler step; 't=…' at the quarter points; open diamond + callout at t=0.5."""
    ax.scatter(path2d[:, 0], path2d[:, 1], s=22, color=color, zorder=4,
               edgecolor="white", linewidth=0.4)
    for xy, t in zip(path2d, t_values):
        tf = float(t)
        if abs(tf - 0.25) < 1e-6 or abs(tf - 0.75) < 1e-6:
            ax.annotate(f"t={tf:g}", xy, textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color=color)
    half = int(np.argmin(np.abs(t_values - 0.5)))
    if abs(t_values[half] - 0.5) < 1e-6:
        ax.scatter(*path2d[half], s=150, facecolor="none", edgecolor=color,
                   linewidth=2.0, marker="D", zorder=6)
        ax.annotate("t = 0.5", path2d[half], textcoords="offset points", xytext=(11, -18),
                    fontsize=9, fontweight="bold", color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.0))


def _shape_legend(ax, extra=()):
    proxies = [
        plt.Line2D([], [], marker="o", color="0.35", ls="", ms=10, mec="black", label="original feature (t=0)"),
        plt.Line2D([], [], marker="s", color="0.35", ls="", ms=9, mec="black", label="endpoint (t=1 fwd / t=0 rev)"),
        plt.Line2D([], [], marker="*", color="0.35", ls="", ms=16, mec="black", label="own class prototype"),
        plt.Line2D([], [], marker="*", color="0.75", ls="", ms=11, label="sibling prototypes"),
        plt.Line2D([], [], marker="o", color="0.35", ls="", ms=4, label="Euler step"),
        plt.Line2D([], [], marker="D", color="0.35", ls="", ms=8, mfc="none", label="t = 0.5"),
        *extra,
    ]
    ax.legend(handles=proxies, fontsize=7.5, loc="best")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--k", default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args = parser.parse_args()
    k = "full" if str(args.k) == "full" else int(args.k)
    T = args.steps

    apply_style()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    art = ArtifactWriter(_OUT_DIR, _OUT_DIR)

    te = load_features(args.encoder, args.dataset, "test")
    selected = pick_diverse_classes(te.classes, N_CLASSES)
    labels_np = te.labels.numpy()
    prototypes = rebuild_prototypes(args.encoder, args.dataset, k, args.seed)
    proto9 = prototypes[selected]                       # the 9 selected, for context in every panel
    pn_all = l2_normalize(prototypes)

    ckpt_path = _MODELS_DIR / checkpoint_name(args.encoder, args.dataset, k, args.seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found -- run train_standard_fm.py first.")
    model, _ = load_fm_model(ckpt_path)

    t_forward = np.arange(T + 1) / T
    t_reverse = 1.0 - np.arange(T + 1) / T

    def class_rows(c):
        return np.where(labels_np == c)[0]

    def pick_representative_row(c):
        """The class member whose transported feature is most aligned with its own prototype
        -- a clean, working example rather than an arbitrary one."""
        rows = class_rows(c)
        zt = l2_normalize(euler_inference_trajectory(model, te.features[rows], steps=T)[-1])
        return int(rows[(zt @ pn_all[c]).argmax().item()])

    # ============================ FORWARD ============================
    fig_f, axes_f = plt.subplots(2, 2, figsize=(13.5, 12.5))
    fwd_manifest = []
    for ax, slot in zip(axes_f.flat, FORWARD_CLASS_SLOTS):
        c = int(selected[slot])
        color = COLORS9[slot]
        row = pick_representative_row(c)
        cloud = te.features[class_rows(c)[:CLOUD_MAX]]
        traj = euler_inference_trajectory(model, te.features[row:row + 1], steps=T)[:, 0, :]  # (T+1, D)

        pca = _fit_panel_pca(traj, proto9, cloud)
        traj2d = _proj(pca, traj)
        cloud2d = _proj(pca, cloud)
        proto9_2d = _proj(pca, proto9)
        own_2d = proto9_2d[slot]

        cos0 = float((l2_normalize(traj[0:1]) @ pn_all[c]).item())
        cos1 = float((l2_normalize(traj[-1:]) @ pn_all[c]).item())
        pred_c = int((l2_normalize(traj[-1:]) @ pn_all.T).argmax().item())

        ax.scatter(cloud2d[:, 0], cloud2d[:, 1], s=16, color=color, alpha=0.16, zorder=1)
        sib = [j for j in range(N_CLASSES) if j != slot]
        ax.scatter(proto9_2d[sib, 0], proto9_2d[sib, 1], s=90, color="0.75",
                   marker="*", edgecolor="0.4", linewidth=0.6, zorder=3)
        ax.plot(traj2d[:, 0], traj2d[:, 1], "-", color=color, lw=1.5, alpha=0.9, zorder=4)
        _draw_steps(ax, traj2d, t_forward, color)
        ax.plot([traj2d[-1, 0], own_2d[0]], [traj2d[-1, 1], own_2d[1]], ":", color=color,
                lw=1.1, alpha=0.8, zorder=3)
        ax.scatter(*traj2d[0], s=200, color=color, marker="o", edgecolor="black", linewidth=1.3, zorder=7)
        ax.scatter(*traj2d[-1], s=170, color=color, marker="s", edgecolor="black", linewidth=1.3, zorder=7)
        ax.scatter(*own_2d, s=430, color=color, marker="*", edgecolor="black", linewidth=1.3, zorder=7)

        xlim, ylim = _square_limits(traj2d, cloud2d, proto9_2d[[slot]])
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        ok = "correct" if pred_c == c else f"-> {te.classes[pred_c]}"
        ax.set_title(f"{te.classes[c]} (test row {row})\n"
                     f"cos to own prototype: {cos0:.2f} (t=0) -> {cos1:.2f} (t=1)   |   argmax/100: {ok}",
                     fontsize=9.5)
        fwd_manifest.append((c, te.classes[c], row, cos0, cos1, te.classes[pred_c]))
    _shape_legend(axes_f.flat[0], extra=[
        plt.Line2D([], [], ls=":", color="0.35", label="t=1 -> own prototype gap")])
    fig_f.suptitle(
        f"Forward flow: individual test features through the standard-FM layer, {T} Euler steps   "
        f"({args.encoder} / {args.dataset}, K={k})",
        fontsize=11.5, fontweight="bold", y=0.995,
    )
    fig_f.text(0.5, 0.945, "per-panel jointly-fit PCA on unit-normalized (trajectory + 9 prototypes "
               "+ class cloud); grey stars = sibling prototypes", ha="center", fontsize=8.5, color="0.35")
    fig_f.tight_layout(rect=[0, 0, 1, 0.935])
    art.figure(fig_f, "flow_trajectories_forward")

    # ============================ REVERSE ============================
    fig_r, axes_r = plt.subplots(1, 2, figsize=(14, 7.2))
    rev_manifest = []
    rev_class_idx = [int(selected[s]) for s in REVERSE_CLASS_SLOTS]
    rev_traj_all = reverse_trajectory_from_prototype(
        model, prototypes[rev_class_idx].to(DEVICE), steps=T)              # (T+1, R, D)
    for ax, slot, j in zip(axes_r.flat, REVERSE_CLASS_SLOTS, range(len(rev_class_idx))):
        c = rev_class_idx[j]
        color = COLORS9[slot]
        cloud = te.features[class_rows(c)[:CLOUD_MAX]]
        traj = rev_traj_all[:, j, :]                                       # (T+1, D)

        pca = _fit_panel_pca(traj, proto9, cloud)
        traj2d = _proj(pca, traj)
        cloud2d = _proj(pca, cloud)
        own_2d = _proj(pca, proto9)[slot]

        real_mean = l2_normalize(cloud.mean(0, keepdim=True))
        cos_proto = float((l2_normalize(traj[0:1]) @ real_mean.T).item())
        cos_end = float((l2_normalize(traj[-1:]) @ real_mean.T).item())

        ax.scatter(cloud2d[:, 0], cloud2d[:, 1], s=20, color=color, alpha=0.32, zorder=2,
                   label=f"{te.classes[c]} real test features")
        ax.plot(traj2d[:, 0], traj2d[:, 1], "--", color=color, lw=1.5, alpha=0.95, zorder=4)
        _draw_steps(ax, traj2d, t_reverse, color)
        ax.scatter(*traj2d[0], s=430, color=color, marker="*", edgecolor="black", linewidth=1.3, zorder=7)
        ax.scatter(*traj2d[-1], s=170, color=color, marker="s", edgecolor="black", linewidth=1.3, zorder=7)
        ax.annotate("t=1 (prototype)", traj2d[0], textcoords="offset points", xytext=(9, 9),
                    fontsize=8.5, fontweight="bold", color=color)
        ax.annotate("t=0", traj2d[-1], textcoords="offset points", xytext=(8, 6), fontsize=8, color=color)

        xlim, ylim = _square_limits(traj2d, cloud2d)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{te.classes[c]}   |   cos to mean real feature: "
                     f"{cos_proto:.2f} (prototype)  ->  {cos_end:.2f} (t=0 endpoint)", fontsize=9.5)
        ax.legend(fontsize=8, loc="best")
        rev_manifest.append((c, te.classes[c], cos_proto, cos_end))
    fig_r.suptitle(
        f"Reverse flow: from a class prototype, integrate -v_theta for {T} steps down to t=0   "
        f"({args.encoder} / {args.dataset}, K={k})",
        fontsize=11.5, fontweight="bold", y=0.995,
    )
    fig_r.text(0.5, 0.925, "per-panel jointly-fit PCA on unit-normalized (reverse trajectory + 9 "
               "prototypes + class cloud)", ha="center", fontsize=8.5, color="0.35")
    fig_r.tight_layout(rect=[0, 0, 1, 0.91])
    art.figure(fig_r, "flow_trajectories_reverse")

    art.table(
        pd.DataFrame(
            [("forward", c, n, r, f"{a:.3f}", f"{b:.3f}", p) for c, n, r, a, b, p in fwd_manifest]
            + [("reverse", c, n, -1, f"{a:.3f}", f"{b:.3f}", "") for c, n, a, b in rev_manifest],
            columns=["direction", "class_index", "class_name", "test_row",
                     "cos_start", "cos_end", "forward_argmax_pred"],
        ),
        "flow_trajectories_examples",
    )
    print("forward (cos to own prototype, t=0 -> t=1):")
    for c, n, r, a, b, p in fwd_manifest:
        print(f"  {n:<13} row {r:<5} {a:.2f} -> {b:.2f}   argmax/100: {p}")
    print("reverse (cos to mean real test feature, prototype -> t=0 endpoint):")
    for c, n, a, b in rev_manifest:
        print(f"  {n:<13} {a:.2f} -> {b:.2f}")


if __name__ == "__main__":
    main()
