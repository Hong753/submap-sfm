import os, random
from submap_sfm.pairs import list_images

SCENE_ROOT = "/data/colmap_scenes/lg_science_park/hub_galaxy_test1"
N_KF = 20
random.seed(0)

# each augmented scene draws keyframes from the OTHER submap(s)
neighbors = {
    "hub_left_aug":  ["hub_right"],
    "hub_right_aug": ["hub_left"],
}

os.makedirs(os.path.join(SCENE_ROOT, "keyframes"), exist_ok=True)
for aug, srcs in neighbors.items():
    pool = []
    for s in srcs:
        pool += list_images(os.path.join(SCENE_ROOT, s, "images"), root=SCENE_ROOT)
    kf = sorted(random.sample(pool, N_KF))
    out = os.path.join(SCENE_ROOT, "keyframes", f"{aug}.txt")
    with open(out, "w") as f:
        f.write("\n".join(kf) + "\n")
    print(f"{aug:14s} -> {out}  ({len(kf)} keyframes from {srcs})")