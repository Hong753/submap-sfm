import os
import shutil
from pathlib import Path
from collections import defaultdict

import yaml
from tqdm import tqdm

from submap_sfm.scene import Scene
from submap_sfm.pairs import list_images
from submap_sfm.keyframes import select_keyframes
from submap_sfm.matching import load_matcher, match_pairs, visualize_pair

CONFIG = "configs/default.yaml"
STRIDE = 20
MIN_MATCHES = 200        # "significant overlap" — LightGlue match count
# ---------------------------------------------------------------------------

scene = Scene.load(CONFIG)
with open(CONFIG) as f:
    M = yaml.safe_load(f).get("matcher", {})
MATCHER = M.get("name", "superpoint-lightglue")
IMG_SIZE = M.get("img_size", 1024)
DEVICE = M.get("device", "cuda")

images_root = os.path.join(scene.root, "images")
images = {
    s: list_images(os.path.join(images_root, s), root=images_root)
    for s in scene.submaps
}

matcher = load_matcher(MATCHER, device=DEVICE, img_size=IMG_SIZE)

best_match = {}          # frame name -> (count, PairMatch), scoped to current overlap
_overlap = ""            # current overlap label, set per loop iteration


def score(pairs, stride):
    """Match count per (name_a, name_b); also remember each frame's best pair.

    Owns its own bar so it can show a live keyframe count (frames seen so far at
    >= MIN_MATCHES) in the postfix, which keeps moving during the long fine pass.
    """
    counts = {}
    pbar = tqdm(pairs, desc=f"  {_overlap} s={stride}", unit="pair",
                position=1, leave=False)
    for pm in match_pairs(matcher, Path(images_root), pbar, img_size=IMG_SIZE,
                          min_matches=0, progress=False):
        n = pm.mkpts0.shape[0]
        counts[(pm.name0, pm.name1)] = n
        for name in (pm.name0, pm.name1):
            if n > best_match.get(name, (0, None))[0]:
                best_match[name] = (n, pm)
        pbar.set_postfix(kf=sum(1 for c, _ in best_match.values() if c >= MIN_MATCHES))
    return counts


def write_keyframes(names, unit):
    path = os.path.join(scene.root, "units", unit, "keyframes.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(names) + ("\n" if names else ""))


def visualize_keyframes(names, unit):
    """Draw each keyframe's strongest match (this overlap) as it's done."""
    out = Path(scene.root) / "units" / unit / "debug" / "keyframes"
    out.mkdir(parents=True, exist_ok=True)
    for name in tqdm(names, desc=f"viz {unit}", unit="kf", position=1, leave=False):
        _, pm = best_match.get(name, (0, None))
        if pm is not None:
            visualize_pair(pm, Path(images_root), out / f"{name.replace('/', '_')}.png")


# clear previous overlays once up front (we append per overlap below)
for s in scene.submaps:
    shutil.rmtree(Path(scene.root) / "units" / f"{s}_aug" / "debug" / "keyframes",
                  ignore_errors=True)

# accumulate keyframes across overlaps; visualize each overlap's matches as it finishes
collected = defaultdict(set)
overlaps = sorted({tuple(sorted(o)) for o in scene.overlaps})
bar = tqdm(overlaps, unit="overlap", position=0)
for a, b in bar:
    bar.set_description(f"{a}<->{b}")
    _overlap = f"{a}<->{b}"
    best_match.clear()
    a_ov, b_ov = select_keyframes(
        images[a], images[b], score, stride=STRIDE, min_inliers=MIN_MATCHES
    )
    collected[f"{a}_aug"].update(b_ov)
    collected[f"{b}_aug"].update(a_ov)
    visualize_keyframes(b_ov, f"{a}_aug")
    visualize_keyframes(a_ov, f"{b}_aug")
    tqdm.write(f"{a} <-> {b}: {len(a_ov)} from {a}, {len(b_ov)} from {b}")

for unit, names in collected.items():
    write_keyframes(sorted(names), unit)
    tqdm.write(f"{unit}: {len(names)} keyframes -> units/{unit}/debug/keyframes/")