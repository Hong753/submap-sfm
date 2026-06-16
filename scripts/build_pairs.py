import os
from collections import defaultdict
from submap_sfm.scene import Scene
from submap_sfm.pairs import (
    list_images,
    node_pairs,
    exhaustive_pairs,
    read_list,
    write_pairs,
)

CONFIG = "configs/default.yaml"
W = 10              # sequential window (node / intra)
# ---------------------------------------------------------------------------

scene = Scene.load(CONFIG)
images_root = os.path.join(scene.root, "images")
images = {
    s: list_images(os.path.join(images_root, s), root=images_root)
    for s in scene.submaps
}


def submap_of(name):
    for s in scene.submaps:
        if name.startswith(s + "/"):
            return s
    return None


def keyframes(aug):
    path = os.path.join(scene.root, "units", aug, "keyframes.txt")
    return set(read_list(path)) if os.path.exists(path) else set()


def emit(unit, fname, pairs):
    write_pairs(sorted(pairs), os.path.join(scene.root, "units", unit, fname))


# --- inter: keyframe <-> keyframe bridge (overlap frames on both sides) ---
# Per overlap (a, b), foreign keyframes live in the opposite unit:
#   b_ov = b-frames in a_aug, a_ov = a-frames in b_aug.
# bridge = a_ov x b_ov; both endpoints exist in both units, so both get the set.
inter = defaultdict(set)
for a, b in sorted({tuple(sorted(o)) for o in scene.overlaps}):
    a_aug, b_aug = f"{a}_aug", f"{b}_aug"
    b_ov = sorted(f for f in keyframes(a_aug) if submap_of(f) == b)
    a_ov = sorted(f for f in keyframes(b_aug) if submap_of(f) == a)
    bridge = exhaustive_pairs(a_ov, b_ov)
    inter[a_aug].update(bridge)
    inter[b_aug].update(bridge)
    print(f"{a}<->{b}: {len(a_ov)}x{len(b_ov)} = {len(bridge)} bridge pairs")

# --- per augmented unit: intra (sequential) + inter (keyframe bridge) ---
splits = {}
for s in scene.submaps:
    aug = f"{s}_aug"
    intra = node_pairs(images[s], window=W)
    itr = inter[aug]
    splits[aug] = (intra, itr)
    emit(aug, "pairs_intra.txt", intra)
    emit(aug, "pairs_inter.txt", itr)
    emit(aug, "pairs.txt", intra | itr)
    print(f"{aug:14s} intra={len(intra)} inter={len(itr)} total={len(intra | itr)}")

# --- full-scene baseline: de-duplicated union of the unit pair sets ---
full, naive = set(), 0
for intra, itr in splits.values():
    s = intra | itr
    full |= s
    naive += len(s)
emit("full_scene", "pairs.txt", full)
print(f"{'full_scene':14s} total={len(full)}  ({naive - len(full)} shared pairs deduped)")