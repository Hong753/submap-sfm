import os
from submap_sfm.scene import Scene
from submap_sfm.pairs import list_images, node_pairs, keyframe_pairs, read_list, write_pairs

CONFIG = "configs/default.yaml"
W = 10
STRIDE = 10       # MUST match the edge stride so inter reuses the edge cache
# ---------------------------------------------------------------------------

scene = Scene.load(CONFIG)
images = {
    s: list_images(os.path.join(scene.root, s, "images"), root=scene.root)
    for s in scene.submaps
}

for s in scene.submaps:
    aug = f"{s}_aug"
    kf = read_list(os.path.join(scene.root, aug, "keyframes.txt"))
    intra = node_pairs(images[s], window=W)
    inter = keyframe_pairs(kf, images[s], stride=STRIDE)
    d = os.path.join(scene.root, aug)
    write_pairs(sorted(intra), os.path.join(d, "pairs_intra.txt"))
    write_pairs(sorted(inter), os.path.join(d, "pairs_inter.txt"))
    write_pairs(sorted(intra | inter), os.path.join(d, "pairs.txt"))
    print(f"{aug:14s} intra={len(intra)} inter={len(inter)} total={len(intra | inter)}")