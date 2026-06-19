#!/usr/bin/env python
"""Per-submap reconstruction (pure, no aug), end to end in one process.

Imports the submap_sfm modules directly -- no shelling out to other scripts --
and for each submap runs the three stages, timing each:

  1. pair definition   list_images + node_pairs (sequential intra)  -> units/<s>/pairs.txt
  2. matching          match_pairs -> build_database -> verify       -> units/<s>/database.db
  3. reconstruction    pycolmap incremental mapping                  -> units/<s>/sparse/<k>

Each submap is reconstructed from its OWN frames only, so it stays clean and
build_database's single-camera default is correct (one video == one camera).
Everything lands in  units/<submap>/  (no "_aug"). A per-stage timing summary
is printed at the end.

Run from the repo root:

  python scripts/run_submaps.py
  python scripts/run_submaps.py --config configs/default.yaml --submaps lounge hub_right hub_left
"""
import argparse
import time
from pathlib import Path

import yaml

from submap_sfm.scene import Scene
from submap_sfm.pairs import list_images, node_pairs, write_pairs
from submap_sfm import matching, colmap_db
from submap_sfm.reconstruct import reconstruct


def matcher_cfg(config):
    """Matcher block from the config (name / img_size / device), with defaults."""
    with open(config) as f:
        m = (yaml.safe_load(f) or {}).get("matcher", {})
    return (m.get("name", "superpoint-lightglue"),
            int(m.get("img_size", 1024)),
            m.get("device", "cuda"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--submaps", nargs="+", default=None,
                    help="subset to run (default: all submaps in the config)")
    ap.add_argument("--window", type=int, default=10,
                    help="sequential intra window for node_pairs; default 10")
    ap.add_argument("--matcher", default=None, help="override config matcher name")
    ap.add_argument("--img-size", type=int, default=None, help="override config img_size")
    ap.add_argument("--device", default=None, help="override config device")
    ap.add_argument("--min-matches", type=int, default=15)
    ap.add_argument("--merge-cell", type=float, default=1.0)
    ap.add_argument("--min-model-size", type=int, default=10)
    args = ap.parse_args()

    scene = Scene.load(args.config)
    root = Path(scene.root)
    image_root = root / "images"
    submaps = args.submaps or scene.submaps

    name, img_size, device = matcher_cfg(args.config)
    name = args.matcher or name
    img_size = args.img_size or img_size
    device = args.device or device

    print(f"scene_root = {root}")
    print(f"submaps    = {list(submaps)}")
    print(f"matcher    = {name}  img_size={img_size}  device={device}  window={args.window}\n")

    t = time.perf_counter()
    matcher = matching.load_matcher(name, device=device, img_size=img_size)
    t_load = time.perf_counter() - t
    print(f"matcher loaded in {t_load:.1f}s\n")

    rows = []
    for s in submaps:
        print(f"=== {s} ===")
        unit_dir = root / "units" / s
        pairs_path = unit_dir / "pairs.txt"
        db_path = unit_dir / "database.db"
        sparse_dir = unit_dir / "sparse"

        # 1. pair definition --------------------------------------------------
        t = time.perf_counter()
        names = list_images(image_root / s, root=image_root)
        pairs = sorted(node_pairs(names, window=args.window))
        write_pairs(pairs, pairs_path)
        t_pairs = time.perf_counter() - t
        print(f"  pairs: {len(names)} frames -> {len(pairs)} pairs  ({t_pairs:.1f}s)")

        # 2. matching: match -> build database -> geometric verification ------
        t = time.perf_counter()
        results = list(matching.match_pairs(matcher, image_root, pairs,
                                             img_size=img_size,
                                             min_matches=args.min_matches,
                                             desc=f"match {s}"))
        colmap_db.build_database(db_path, image_root, results, merge_cell=args.merge_cell)
        colmap_db.verify(db_path, pairs_path)
        t_match = time.perf_counter() - t
        print(f"  match: kept {len(results)}/{len(pairs)} pairs, db built + verified  ({t_match:.1f}s)")

        # 3. reconstruction ---------------------------------------------------
        t = time.perf_counter()
        maps = reconstruct(database_path=db_path, image_path=image_root,
                           output_path=sparse_dir, min_model_size=args.min_model_size)
        t_recon = time.perf_counter() - t
        if maps:
            best = max(maps, key=lambda i: maps[i].num_reg_images())
            reg, pts = maps[best].num_reg_images(), maps[best].num_points3D()
            print(f"  recon: {len(maps)} model(s); largest model {best} -> "
                  f"{reg} imgs / {pts} pts  ({t_recon:.1f}s)")
        else:
            reg = pts = 0
            print(f"  recon: no model produced  ({t_recon:.1f}s)")

        rows.append((s, len(names), len(pairs), len(results),
                     t_pairs, t_match, t_recon, reg, pts))
        print()

    # timing summary ----------------------------------------------------------
    W = 96
    print("=" * W)
    print(f"{'submap':<14}{'frames':>7}{'pairs':>8}{'kept':>7}"
          f"{'pair(s)':>10}{'match(s)':>11}{'recon(s)':>11}{'total(s)':>11}{'reg/pts':>17}")
    print("-" * W)
    nf_t = np_t = nk_t = 0
    tp_t = tm_t = tr_t = 0.0
    for (s, nf, npairs, nkept, tp, tm, tr, reg, pts) in rows:
        print(f"{s:<14}{nf:>7}{npairs:>8}{nkept:>7}"
              f"{tp:>10.1f}{tm:>11.1f}{tr:>11.1f}{tp + tm + tr:>11.1f}{f'{reg}/{pts}':>17}")
        nf_t += nf; np_t += npairs; nk_t += nkept
        tp_t += tp; tm_t += tm; tr_t += tr
    print("-" * W)
    print(f"{'TOTAL':<14}{nf_t:>7}{np_t:>8}{nk_t:>7}"
          f"{tp_t:>10.1f}{tm_t:>11.1f}{tr_t:>11.1f}{tp_t + tm_t + tr_t:>11.1f}")
    print(f"\nmatcher load: {t_load:.1f}s   |   wall total incl. load: "
          f"{t_load + tp_t + tm_t + tr_t:.1f}s")


if __name__ == "__main__":
    main()