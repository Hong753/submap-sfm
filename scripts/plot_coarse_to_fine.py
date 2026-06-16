#!/usr/bin/env python
"""Render the coarse-to-fine illustration from a cached cross-match grid.

Run cross_match_grid.py first (writes units/_grid/grid_<a>_<b>_n<n>.npz); this
loads that grid, runs the real select_keyframes on it (recording which cells the
coarse-to-fine pass evaluates), and plots the full grid + the sampling.
For intra (--a == --b) it just shows the grid (diagonal zeroed).

  python scripts/plot_coarse_to_fine.py --a hub_left --b hub_right --n 500
  python scripts/plot_coarse_to_fine.py --a hub_left --b hub_left  --n 500   # intra grid
"""
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

from submap_sfm.scene import Scene
from submap_sfm.pairs import list_images
from submap_sfm.keyframes import select_keyframes

# distinct shape per stride level (coarsest first); coarse = big hollow so it
# never covers the finer markers underneath
LEVEL_STYLE = [
    dict(marker="s", s=140, facecolors="none", edgecolors="#1f77b4", linewidths=1.7, zorder=4),
    dict(marker="D", s=46,  c="#2ca02c", zorder=3, alpha=0.9),
    dict(marker=".", s=34,  c="#d62728", zorder=2, alpha=0.85),
    dict(marker="^", s=24,  c="#9467bd", zorder=2, alpha=0.85),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--a", default="hub_left")
    ap.add_argument("--b", default="hub_right")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--contiguous", action="store_true")
    ap.add_argument("--a-start", type=int, default=0)
    ap.add_argument("--b-start", type=int, default=0)
    ap.add_argument("--max-stride", type=int, default=20)
    ap.add_argument("--min-stride", type=int, default=5)
    ap.add_argument("--min-matches", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    intra = (args.a == args.b)

    scene = Scene.load(args.config)
    root = Path(scene.root)
    image_root = root / "images"
    out_dir = root / "units" / "_grid"
    cache = out_dir / f"grid_{args.a}_{args.b}_n{args.n}.npz"
    if not cache.exists():
        raise SystemExit(f"no grid at {cache} — run cross_match_grid.py with the same --a/--b/--n first")
    out_png = Path(args.out) if args.out else out_dir / f"coarse_to_fine_{args.a}_{args.b}_n{args.n}.png"

    grid = np.load(cache)["grid"].astype(int)
    if intra:
        np.fill_diagonal(grid, 0)   # self-matches are trivially maximal, ignore them
    NA, NB = grid.shape

    a_all = list_images(os.path.join(image_root, args.a), root=image_root)
    b_all = list_images(os.path.join(image_root, args.b), root=image_root)
    if args.contiguous:
        a_names = a_all[args.a_start: args.a_start + args.n]
        b_names = b_all[args.b_start: args.b_start + args.n]
    else:
        a_names = a_all[:: max(1, len(a_all) // args.n)][: args.n]
        b_names = b_all[:: max(1, len(b_all) // args.n)][: args.n]
    if (len(a_names), len(b_names)) != (NA, NB):
        raise SystemExit(f"frame counts {(len(a_names), len(b_names))} != grid {(NA, NB)} — "
                         f"--contiguous/--a-start/--b-start must match the cross_match_grid run")

    # coarse-to-fine overlay only makes sense for the cross grid
    evaluated, sel_cells, n_eval, frac = {}, [], 0, 0.0
    if not intra:
        a_pos = {n: i for i, n in enumerate(a_names)}
        b_pos = {n: j for j, n in enumerate(b_names)}
        def score(pairs, stride):
            out = {}
            for p in pairs:
                na, nb = (p[0], p[1]) if p[0] in a_pos else (p[1], p[0])
                i, j = a_pos[na], b_pos[nb]
                out[p] = int(grid[i, j])
                evaluated.setdefault((i, j), stride)
            return out
        a_sel, b_sel = select_keyframes(a_names, b_names, score,
                                        max_stride=args.max_stride, min_stride=args.min_stride,
                                        min_inliers=args.min_matches)
        n_eval = len(evaluated)
        frac = 100 * n_eval / (NA * NB)
        sel_cells = [(i, j) for (i, j) in evaluated if grid[i, j] >= args.min_matches]
        print(f"coarse-to-fine evaluated {n_eval}/{NA*NB} = {frac:.1f}% | selected a:{len(a_sel)} b:{len(b_sel)}")

    # plot: one panel for intra (just the grid), two for cross
    norm = PowerNorm(0.45, vmin=0, vmax=max(1, int(np.percentile(grid, 99))))
    if intra:
        fig, ax0 = plt.subplots(1, 1, figsize=(9.5, 9), constrained_layout=True)
    else:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(19, 9.2), constrained_layout=True)

    im = ax0.imshow(grid, norm=norm, cmap="magma")
    ax0.set_xlabel(f"{args.b} frame", fontsize=12)
    ax0.set_ylabel(f"{args.a} frame", fontsize=12)
    ax0.set_aspect("equal")
    ax0.set_title(f"Full match grid — {NA}×{NB} = {NA*NB:,} pairs", fontsize=13)
    fig.colorbar(im, ax=ax0, shrink=0.8, label="matches")

    if not intra:
        ax1.imshow(grid, norm=norm, cmap="Greys", alpha=0.22)
        for k, s in enumerate(sorted({st for st in evaluated.values()}, reverse=True)):
            st = dict(LEVEL_STYLE[min(k, len(LEVEL_STYLE) - 1)])
            pts = [(j, i) for (i, j), v in evaluated.items() if v == s]
            xs, ys = zip(*pts)
            ax1.scatter(xs, ys, label=f"stride {s}", **st)
        if sel_cells:
            xs, ys = zip(*[(j, i) for (i, j) in sel_cells])
            ax1.scatter(xs, ys, marker="o", s=80, facecolors="none", edgecolors="k",
                        linewidths=1.3, zorder=5, label=f"selected (≥{args.min_matches})")
        ax1.set_xlim(-0.5, NB - 0.5)
        ax1.set_ylim(NA - 0.5, -0.5)
        ax1.set_xlabel(f"{args.b} frame", fontsize=12)
        ax1.set_aspect("equal")
        ax1.set_title(f"Coarse-to-fine — {n_eval:,} cells ({frac:.1f}%)", fontsize=13)
        ax1.legend(loc="upper right", fontsize=11, framealpha=0.95, markerscale=1.2)

    title = f"Intra match grid: {args.a}" if intra else f"Coarse-to-fine overlap discovery: {args.a} ↔ {args.b}"
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.savefig(out_png, dpi=150)
    print(f"figure -> {out_png}")


if __name__ == "__main__":
    main()