#!/usr/bin/env python
"""Match a unit's pairs with vismatch and write a verified COLMAP database.
  python scripts/run_matching.py --unit hub_left_aug
  python scripts/run_matching.py --unit hub_left_aug --pairs pairs_inter.txt --debug --viz-every 50
  python scripts/run_matching.py --submaps hub_left hub_right        # subset of full_scene, mapped as one unit
"""
from __future__ import annotations
import argparse
from pathlib import Path
from submap_sfm.scene import Scene
from submap_sfm import matching, colmap_db
from submap_sfm.pairs import read_pairs, write_pairs


def units(scene: Scene) -> list[str]:
    return [f"{s}_aug" for s in scene.submaps] + ["full_scene"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--unit", default=None, help="default: all units")
    ap.add_argument("--submaps", nargs="+", default=None,
                    help="filter full_scene pairs to this submap subset and map it as one unit")
    ap.add_argument("--pairs", default="pairs.txt")
    ap.add_argument("--matcher", default="superpoint-lightglue")
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-matches", type=int, default=15)
    ap.add_argument("--merge-cell", type=float, default=1.0)
    ap.add_argument("--min-inliers", type=int, default=0,
                    help="for inter pairs only: drop pairs with fewer verified inliers (0 = keep all)")
    ap.add_argument("--debug", action="store_true", help="print per-pair counts")
    ap.add_argument("--viz-every", type=int, default=0,
                    help="save an overlay every Nth matched pair (0 = none); works in full runs too")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N pairs (dev throttle)")
    ap.add_argument("--write", action="store_true", help="write db even in --debug")
    args = ap.parse_args()

    scene = Scene.load(args.config)
    root = Path(scene.root)
    image_root = root / "images"            # names are <submap>/<frame>.jpg relative to here

    def submap_of(name):
        for s in scene.submaps:
            if name.startswith(s + "/"):
                return s
        return None

    # subset mode: filter full_scene's pairs to the chosen submaps into a derived unit
    if args.submaps:
        subset = set(args.submaps)
        unknown = subset - set(scene.submaps)
        if unknown:
            ap.error(f"unknown submaps {sorted(unknown)}; choose from {scene.submaps}")
        unit = args.unit or "+".join(sorted(subset))
        src = root / "units" / "full_scene" / args.pairs
        kept_pairs = [p for p in read_pairs(src)
                      if submap_of(p[0]) in subset and submap_of(p[1]) in subset]
        (root / "units" / unit).mkdir(parents=True, exist_ok=True)
        write_pairs(sorted(kept_pairs), root / "units" / unit / args.pairs)
        print(f"filtered {args.pairs}: {len(kept_pairs)} pairs within {sorted(subset)} -> units/{unit}/")
        todo = [unit]
    else:
        todo = [args.unit] if args.unit else units(scene)

    matcher = matching.load_matcher(args.matcher, device=args.device, img_size=args.img_size)

    for unit in todo:
        unit_dir = root / "units" / unit    # derived artifacts live under units/
        pairs_path = unit_dir / args.pairs
        role = Path(args.pairs).stem  # "pairs_intra" | "pairs_inter" | "pairs"
        pairs = read_pairs(pairs_path)
        if args.limit:
            pairs = pairs[: args.limit]
        print(f"\n=== {unit}: {len(pairs)} pairs from {args.pairs} "
              f"[{args.matcher}, img_size={args.img_size}] ===")

        results = []
        viz_dir = unit_dir / "debug" / role  # namespaced by unit + pair role -> no collisions
        for pm in matching.match_pairs(matcher, image_root, pairs, img_size=args.img_size,
                                       min_matches=args.min_matches):
            results.append(pm)
            n = len(results)
            if args.debug:
                print(f"  [{role}] {pm.name0} <-> {pm.name1}: {pm.mkpts0.shape[0]} matches")
            if args.viz_every and (n - 1) % args.viz_every == 0:
                s0 = pm.name0.replace("/", "-").rsplit(".", 1)[0]
                s1 = pm.name1.replace("/", "-").rsplit(".", 1)[0]
                matching.visualize_pair(pm, image_root, viz_dir / f"{s0}__{s1}.png")

        print(f"  kept {len(results)}, dropped {len(pairs) - len(results)} (< {args.min_matches})")
        if args.viz_every:
            print(f"  overlays -> {viz_dir}")
        if args.debug and not args.write:
            print("  [debug] skipping db write")
            continue

        db_path = unit_dir / "database.db"
        colmap_db.build_database(db_path, image_root, results, merge_cell=args.merge_cell)
        colmap_db.verify(db_path, pairs_path)
        if role == "pairs_inter" and args.min_inliers > 0:
            kept = colmap_db.prune_by_inliers(db_path, args.min_inliers)
            print(f"  pruned inter pairs below {args.min_inliers} inliers; {kept} pairs remain verified")
        print(f"  wrote + verified {db_path}")


if __name__ == "__main__":
    main()