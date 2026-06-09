import os
from submap_sfm.scene import Scene
from submap_sfm.pairs import (
    list_images,
    build_scene_pairs_split,
    summarize,
    write_pairs,
    read_list,
)

CONFIG = "configs/default.yaml"
W = 20               # sequential window (matching-method param)

scene = Scene.load(CONFIG)
# ---------------------------------------------------------------------------

images = {
    s: list_images(os.path.join(scene.root, s, "images"), root=scene.root)
    for s in scene.submaps
}


def kf(aug):
    return read_list(os.path.join(scene.root, aug, "keyframes.txt"))


def emit(unit, fname, pairs):
    write_pairs(pairs, os.path.join(scene.root, unit, fname))


def write_unit(unit, intra, inter):
    emit(unit, "pairs_intra.txt", intra)                      # within-node
    emit(unit, "pairs_inter.txt", inter)                      # node-to-edge
    emit(unit, "pairs.txt", sorted(set(intra) | set(inter)))  # combined, for matching


# --- augmented submaps ---
splits = {}
for s in scene.submaps:
    aug = f"{s}_aug"
    groups, keyframes = [images[s]], kf(aug)
    parts = build_scene_pairs_split(groups, keyframes, window=W)
    splits[aug] = parts
    write_unit(aug, parts["intra"], parts["inter"])
    print(f"{aug:14s} {summarize(groups, keyframes, W)}")

# --- full-scene baseline: per-category union of the submaps ---
full_intra = sorted(set().union(*(set(p["intra"]) for p in splits.values())))
full_inter = sorted(set().union(*(set(p["inter"]) for p in splits.values())))
write_unit("full_scene", full_intra, full_inter)
print(f"{'full_scene':14s} intra={len(full_intra)} inter={len(full_inter)} "
      f"total={len(set(full_intra) | set(full_inter))}")