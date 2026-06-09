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

#----------------------------------------------------------------------------

images = {
    s: list_images(os.path.join(SCENE_ROOT, s, "images"), root=SCENE_ROOT)
    for s in SUBMAPS
}

def kf(aug):
    return read_list(os.path.join(SCENE_ROOT, "keyframes", f"{aug}.txt"))

# reconstruction units: name -> (sequential groups, keyframes)
units = {
    "hub_left_aug":  ([images["hub_left"]],  kf("hub_left_aug")),
    "hub_right_aug": ([images["hub_right"]], kf("hub_right_aug")),
}

# full scene: all trajectories + every bridge keyframe (deduped union)
all_kf = sorted(set(kf("hub_left_aug")) | set(kf("hub_right_aug")))
units["full_scene"] = ([images["hub_left"], images["hub_right"]], all_kf)

#----------------------------------------------------------------------------

for name, (groups, keyframes) in units.items():
    pairs = build_scene_pairs(groups, keyframes, window=W)
    out = os.path.join(SCENE_ROOT, name, "pairs.txt")
    write_pairs(pairs, out)
    print(f"{name:14s} {summarize(groups, keyframes, W)}")