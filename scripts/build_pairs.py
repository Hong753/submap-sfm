import os
from submap_sfm.pairs import (
    list_images,
    build_scene_pairs,
    summarize,
    write_pairs,
    read_list,
)

SCENE_ROOT = "/data/colmap_scenes/lg_science_park/hub_galaxy_test1"
SUBMAPS = [
    "hub_left",
    "hub_right",
]
W = 20
# ---------------------------------------------------------------------------

# image names relative to SCENE_ROOT, e.g. "hub_left/images/00000.jpg"
images = {
    s: list_images(os.path.join(SCENE_ROOT, s, "images"), root=SCENE_ROOT)
    for s in SUBMAPS
}


def kf(aug):
    return read_list(os.path.join(SCENE_ROOT, "keyframes", f"{aug}.txt"))

# --- augmented submaps: local trajectory + keyframes from the neighbour ---
submap_pairs = {}
for s in SUBMAPS:
    aug = f"{s}_aug"
    groups, keyframes = [images[s]], kf(aug)
    pairs = build_scene_pairs(groups, keyframes, window=W)
    submap_pairs[aug] = pairs
    write_pairs(pairs, os.path.join(SCENE_ROOT, aug, "pairs.txt"))
    print(f"{aug:14s} {summarize(groups, keyframes, W)}")

# --- full-scene baseline: union of the submap pair sets (== the merge's matches) ---
full = sorted(set().union(*submap_pairs.values()))
write_pairs(full, os.path.join(SCENE_ROOT, "full_scene", "pairs.txt"))
naive = sum(len(p) for p in submap_pairs.values())
print(f"{'full_scene':14s} total_pairs={len(full)}  ({naive - len(full)} shared pairs deduped)")

# ---------------------------------------------------------------------------