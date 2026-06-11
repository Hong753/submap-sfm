import os

from submap_sfm.scene import Scene
from submap_sfm.pairs import list_images
from submap_sfm.keyframes import select_keyframes

CONFIG = "configs/default.yaml"
STRIDE = 20
MIN_INLIERS = 100        # "significant overlap"; well above COLMAP's ~30 PnP floor
# ---------------------------------------------------------------------------

scene = Scene.load(CONFIG)
images_root = os.path.join(scene.root, "images")
images = {
    s: list_images(os.path.join(images_root, s), root=images_root)
    for s in scene.submaps
}


def score(pairs):
    """Verified inlier count for each cross pair.

    in:  [(name_a, name_b), ...]   names relative to {scene.root}/images
    out: {(name_a, name_b): num_verified_inliers}

    WIRE THIS TO matching.py. Either call a matching.py function that matches a
    pair list and hand back its verified match counts, or write `pairs` to a
    scratch pairs file, run your normal match+verify into a scratch database.db,
    and read the two-view inlier counts back out.
    """
    raise NotImplementedError("connect to matching.py")


def write_keyframes(names, unit):
    path = os.path.join(scene.root, "units", unit, "keyframes.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(names) + ("\n" if names else ""))


for a, b in {tuple(sorted(o)) for o in scene.overlaps}:
    a_ov, b_ov = select_keyframes(
        images[a], images[b], score, stride=STRIDE, min_inliers=MIN_INLIERS
    )
    # a-frames see b -> keyframes for b's aug unit; b-frames see a -> a's aug unit
    write_keyframes(b_ov, f"{a}_aug")
    write_keyframes(a_ov, f"{b}_aug")
    print(f"{a} <-> {b}: {len(a_ov)} from {a}, {len(b_ov)} from {b}")