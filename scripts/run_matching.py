#!/usr/bin/env python
"""Match a unit's pairs with vismatch and write a verified COLMAP database.

  python scripts/run_matching.py
  python scripts/run_matching.py --unit hub_left_aug
  python scripts/run_matching.py --unit hub_left_aug --pairs pairs_inter.txt --debug --limit 30
  python scripts/run_matching.py --matcher roma
"""
from __future__ import annotations

import argparse
from pathlib import Path

from submap_sfm.scene import Scene
from submap_sfm import matching, colmap_db
from submap_sfm.pairs import read_pairs

def units(scene: Scene) -> list[str]:
    return [f"{s}_aug" for s in scene.submaps] + ["full_scene"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--unit", default=None, help="default: all units")
    ap.add_argument("--pairs", default="pairs.txt")
    ap.add_argument("--matcher", default="superpoint-lightglue")
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-matches", type=int, default=15)
    ap.add_argument("--merge-cell", type=float, default=1.0)
    ap.add_argument("--debug", action="store_true",
                    help="visualize matches, print counts, skip db write unless --write")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    scene = Scene.load(args.config)
    root = Path(scene.root)
    todo = [args.unit] if args.unit else units(scene)
    matcher = matching.load_matcher(args.matcher, device=args.device, img_size=args.img_size)

    for unit in todo:
        unit_dir = root / unit
        pairs_path = unit_dir / args.pairs
        pairs = read_pairs(pairs_path)
        if args.limit:
            pairs = pairs[: args.limit]
        print(f"\n=== {unit}: {len(pairs)} pairs from {args.pairs} "
              f"[{args.matcher}, img_size={args.img_size}] ===")

        results = []
        debug_dir = unit_dir / "debug"
        for pm in matching.match_pairs(matcher, root, pairs, img_size=args.img_size,
                                       min_matches=args.min_matches):
            results.append(pm)
            if args.debug:
                print(f"  {pm.name0} <-> {pm.name1}: {pm.mkpts0.shape[0]} matches")
                if len(results) <= 20:
                    stem = f"{Path(pm.name0).stem}__{Path(pm.name1).stem}.png"
                    matching.visualize_pair(pm, root, debug_dir / stem)

        print(f"  kept {len(results)}, dropped {len(pairs) - len(results)} (< {args.min_matches})")
        if args.debug and not args.write:
            print(f"  [debug] viz -> {debug_dir} ; skipping db write")
            continue

        db_path = unit_dir / "database.db"
        colmap_db.build_database(db_path, root, results, merge_cell=args.merge_cell)
        colmap_db.verify(db_path, pairs_path)
        print(f"  wrote + verified {db_path}")


if __name__ == "__main__":
    main()