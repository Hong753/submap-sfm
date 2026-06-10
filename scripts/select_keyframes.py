import os
from collections import defaultdict
import pycolmap
from submap_sfm.scene import Scene

CONFIG = "configs/default.yaml"
N_KF = 20                 # keyframes per augmented submap (top-scoring)
MIN_INLIERS = 100         # ignore weak cross geometries
MAX_PID = 2147483647      # COLMAP pair_id base: pid = id1*MAX_PID + id2
# ---------------------------------------------------------------------------

scene = Scene.load(CONFIG)


def edge_name(a: str, b: str) -> str:
    a, b = sorted((a, b))
    return f"{a}__{b}".replace("/", "-")


def submap_of(name: str):
    for s in scene.submaps:
        if name.startswith(s + "/"):
            return s
    return None


# scores[s_aug][neighbour_frame] = total verified cross-inliers bridging to s
scores = defaultdict(lambda: defaultdict(int))

for a, b in {tuple(sorted(e)) for e in scene.overlaps}:
    db_path = os.path.join(scene.root, "edges", edge_name(a, b), "database.db")
    db = pycolmap.Database.open(db_path)
    id_to_name = {im.image_id: im.name for im in db.read_all_images()}
    pair_ids, counts = db.read_two_view_geometry_num_inliers()
    db.close()

    for pid, n in zip(pair_ids, counts):
        if n < MIN_INLIERS:
            continue
        n1 = id_to_name.get(pid // MAX_PID)
        n2 = id_to_name.get(pid % MAX_PID)
        if n1 is None or n2 is None:
            continue
        s1, s2 = submap_of(n1), submap_of(n2)
        if s1 is None or s2 is None or s1 == s2:
            continue
        # frame from s2 bridges to s1 -> candidate keyframe for s1_aug, and vice versa
        scores[f"{s1}_aug"][n2] += n
        scores[f"{s2}_aug"][n1] += n

for aug, frame_scores in scores.items():
    ranked = sorted(frame_scores, key=frame_scores.get, reverse=True)[:N_KF]
    keyframes = sorted(ranked)
    os.makedirs(os.path.join(scene.root, aug), exist_ok=True)
    out = os.path.join(scene.root, aug, "keyframes.txt")
    with open(out, "w") as f:
        f.write("\n".join(keyframes) + "\n")
    top = [frame_scores[k] for k in ranked[:3]]
    print(f"{aug:14s} {len(keyframes)} keyframes  (top inlier scores: {top})")