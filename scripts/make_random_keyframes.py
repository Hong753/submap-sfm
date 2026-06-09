import os, random
from submap_sfm.scene import Scene
from submap_sfm.pairs import list_images

CONFIG = "configs/default.yaml"
N_KF = 20
random.seed(0)

scene = Scene.load(CONFIG)

for s in scene.submaps:
    aug = f"{s}_aug"
    pool = []
    for nb in scene.neighbors(s):
        pool += list_images(os.path.join(scene.root, nb, "images"), root=scene.root)
    kf = sorted(random.sample(pool, N_KF))
    os.makedirs(os.path.join(scene.root, aug), exist_ok=True)
    out = os.path.join(scene.root, aug, "keyframes.txt")
    with open(out, "w") as f:
        f.write("\n".join(kf) + "\n")
    print(f"{aug:14s} -> {out}  ({len(kf)} keyframes from {scene.neighbors(s)})")