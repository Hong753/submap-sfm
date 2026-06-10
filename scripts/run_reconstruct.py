import argparse
import os

from submap_sfm.scene import Scene
from submap_sfm.reconstruct import reconstruct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--unit", default="full_scene")
    ap.add_argument("--min-model-size", type=int, default=10)
    args = ap.parse_args()

    scene = Scene.load(args.config)
    unit_dir = os.path.join(scene.root, args.unit)
    db_path = os.path.join(unit_dir, "database.db")
    sparse_dir = os.path.join(unit_dir, "sparse")

    print(f"=== {args.unit}: incremental mapping ===")
    print(f"  db    {db_path}")
    print(f"  imgs  {scene.root}")
    print(f"  out   {sparse_dir}")

    maps = reconstruct(
        database_path=db_path,
        image_path=scene.root,             # DB names are relative to SCENE_ROOT
        output_path=sparse_dir,
        min_model_size=args.min_model_size,
    )

    if not maps:
        print("  no reconstruction produced (nothing registered)")
        return

    for idx in sorted(maps):
        rec = maps[idx]
        print(f"  model {idx}: {rec.num_reg_images()} images, {rec.num_points3D()} points")
    best = max(maps, key=lambda i: maps[i].num_reg_images())
    print(f"  largest: model {best} -> {os.path.join(sparse_dir, str(best))} "
          f"({maps[best].num_reg_images()} registered)")


if __name__ == "__main__":
    main()