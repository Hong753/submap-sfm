#!/usr/bin/env python
"""Brute-force the cross-submap match grid (SuperPoint+LightGlue), resumable.
Writes the count grid to units/_grid/grid_<a>_<b>_n<n>.npz so
plot_coarse_to_fine.py can render it.

  python scripts/cross_match_grid.py --a hub_left --b hub_right --n 500

500x500 = 250k matches (~hours). Checkpoints every --flush cells; rerun resumes.
"""
import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from submap_sfm.scene import Scene
from submap_sfm.matching import load_matcher
from submap_sfm.pairs import list_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--a", default="hub_left")
    ap.add_argument("--b", default="hub_right")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--contiguous", action="store_true",
                    help="consecutive window (with --a-start/--b-start) instead of spanning the trajectory")
    ap.add_argument("--a-start", type=int, default=0)
    ap.add_argument("--b-start", type=int, default=0)
    ap.add_argument("--matcher", default="superpoint-lightglue")
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--flush", type=int, default=2000, help="checkpoint every N cells")
    args = ap.parse_args()

    scene = Scene.load(args.config)
    root = Path(scene.root)
    image_root = root / "images"
    out_dir = root / "units" / "_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f"grid_{args.a}_{args.b}_n{args.n}.npz"

    a_all = list_images(os.path.join(image_root, args.a), root=image_root)
    b_all = list_images(os.path.join(image_root, args.b), root=image_root)
    if args.contiguous:
        a_names = a_all[args.a_start: args.a_start + args.n]
        b_names = b_all[args.b_start: args.b_start + args.n]
    else:  # evenly spaced across the full trajectory, order preserved (band guaranteed visible)
        a_names = a_all[:: max(1, len(a_all) // args.n)][: args.n]
        b_names = b_all[:: max(1, len(b_all) // args.n)][: args.n]
    NA, NB = len(a_names), len(b_names)

    if cache.exists():
        d = np.load(cache)
        grid, done = d["grid"].astype(int), d["done"]
        if grid.shape != (NA, NB):
            raise SystemExit(f"cached grid {grid.shape} != {(NA, NB)} — delete {cache} or fix --n/--contiguous")
        print(f"resuming: {int(done.sum())}/{NA*NB} cells already done")
    else:
        grid = np.zeros((NA, NB), int)
        done = np.zeros((NA, NB), bool)

    m = load_matcher(args.matcher, device=args.device, img_size=args.img_size)
    todo = int((~done).sum())
    pbar = tqdm(total=todo, desc=f"{args.a} x {args.b}", unit="pair", smoothing=0.02)
    since = 0
    for i, a in enumerate(a_names):
        if done[i].all():
            continue
        img_a = m.load_image(str(image_root / a), resize=args.img_size)  # load A-row once
        for j, b in enumerate(b_names):
            if done[i, j]:
                continue
            img_b = m.load_image(str(image_root / b), resize=args.img_size)
            res = m(img_a, img_b)
            grid[i, j] = int(np.asarray(res["matched_kpts0"]).shape[0])
            done[i, j] = True
            pbar.update(1)
            since += 1
            if since >= args.flush:
                np.savez_compressed(cache, grid=grid, done=done)
                pbar.set_postfix(saved=int(done.sum()), max_m=int(grid.max()))
                since = 0
    pbar.close()
    np.savez_compressed(cache, grid=grid, done=done)
    print(f"done. grid -> {cache}  (max matches {grid.max()})")


if __name__ == "__main__":
    main()