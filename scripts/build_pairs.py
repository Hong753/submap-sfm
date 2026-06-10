import os

from submap_sfm.scene import Scene
from submap_sfm.pairs import (
    list_images,
    node_pairs,
    keyframe_pairs,
    read_list,
    write_pairs,
)

CONFIG = "configs/default.yaml"
W = 10              # sequential window (node / intra)
INTER_STRIDE = 5    # keyframe -> local stride; 1 = every local frame, raise to speed up

# ---------------------------------------------------------------------------

scene = Scene.load(CONFIG)
images = {
    s: list_images(os.path.join(scene.root, s, "images"), root=scene.root)
    for s in scene.submaps
}


def kf(aug):
    return read_list(os.path.join(scene.root, aug, "keyframes.txt"))


def emit(unit, fname, pairs):
    write_pairs(sorted(pairs), os.path.join(scene.root, unit, fname))


# --- augmented submaps: sequential (intra) + manual keyframes (inter) ---
splits = {}
for s in scene.submaps:
    aug = f"{s}_aug"
    intra = node_pairs(images[s], window=W)
    inter = keyframe_pairs(kf(aug), images[s], stride=INTER_STRIDE)
    splits[aug] = (intra, inter)
    emit(aug, "pairs_intra.txt", intra)
    emit(aug, "pairs_inter.txt", inter)
    emit(aug, "pairs.txt", intra | inter)
    print(f"{aug:14s} intra={len(intra)} inter={len(inter)} total={len(intra | inter)}")

# --- full-scene baseline: de-duplicated union of the submap pair sets ---
full, naive = set(), 0
for intra, inter in splits.values():
    s = intra | inter
    full |= s
    naive += len(s)
emit("full_scene", "pairs.txt", full)
print(f"{'full_scene':14s} total={len(full)}  ({naive - len(full)} shared pairs deduped)")