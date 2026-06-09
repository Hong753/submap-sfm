"""Bridge vismatch per-pair coordinates into a COLMAP database via pycolmap.

merge per-image keypoints (quantized) -> indexed matches -> write db ->
geometric verification (pycolmap.verify_matches) to populate two_view_geometries.

Validated against pycolmap 4.0.4:
  - Database is an abstract interface; get the concrete backend via Database.open(path)
  - write_keypoints takes a raw [m, 4] float32 array (x, y, scale, orientation)
  - write_matches takes a raw [m, 2] uint32 array (no FeatureMatches conversion)
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pycolmap

from submap_sfm.matching import PairMatch, image_size

PIXEL_ORIGIN_SHIFT = 0.5  # COLMAP: center of upper-left pixel is (0.5, 0.5)

class KeypointTable:
    """Accumulate matched points per image, quantizing to a grid so the same physical
    point seen across different pairs collapses to one COLMAP keypoint index."""

    def __init__(self, cell: float = 1.0):
        self.cell = cell
        self._index: dict[str, dict[tuple[int, int], int]] = defaultdict(dict)
        self._coords: dict[str, list[tuple[float, float]]] = defaultdict(list)

    def add(self, name: str, xy: np.ndarray) -> np.ndarray:
        idx_map, coords = self._index[name], self._coords[name]
        out = np.empty(len(xy), dtype=np.uint32)
        for i, (x, y) in enumerate(xy):
            key = (int(round(x / self.cell)), int(round(y / self.cell)))
            j = idx_map.get(key)
            if j is None:
                j = len(coords)
                idx_map[key] = j
                coords.append((float(x), float(y)))
            out[i] = j
        return out

    def keypoints(self, name: str) -> np.ndarray:
        """Return [m, 4] float32: x, y, scale, orientation (only x,y used downstream)."""
        arr = np.asarray(self._coords.get(name, []), dtype=np.float32).reshape(-1, 2)
        arr = arr + PIXEL_ORIGIN_SHIFT
        out = np.zeros((len(arr), 4), dtype=np.float32)
        out[:, :2] = arr
        out[:, 2] = 1.0  # scale; orientation left 0
        return out


def build_database(db_path: Path, scene_root: Path, matches: list[PairMatch], *,
                   camera_model: str = "SIMPLE_RADIAL", focal_factor: float = 1.2,
                   merge_cell: float = 1.0, single_camera: bool = True) -> dict[str, int]:
    """Create database.db with cameras, images, keypoints, matches. Returns name -> image_id.
    Run verify() afterwards (mapper reads two_view_geometries only)."""
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    db = pycolmap.Database.open(str(db_path))

    table = KeypointTable(cell=merge_cell)
    indexed: list[tuple[str, str, np.ndarray]] = []
    names: set[str] = set()
    for pm in matches:
        i0 = table.add(pm.name0, pm.mkpts0)
        i1 = table.add(pm.name1, pm.mkpts1)
        indexed.append((pm.name0, pm.name1, np.column_stack([i0, i1]).astype(np.uint32)))
        names.update((pm.name0, pm.name1))

    image_ids: dict[str, int] = {}
    shared_cam_id = None
    for name in sorted(names):  # ids increase with name; pairs are name-sorted => id0 < id1
        w, h = image_size(scene_root / name)
        if single_camera and shared_cam_id is not None:
            cam_id = shared_cam_id
        else:
            f = focal_factor * max(w, h)
            cam = pycolmap.Camera(model=camera_model, width=w, height=h,
                                  params=[f, w / 2.0, h / 2.0, 0.0])  # SIMPLE_RADIAL: f,cx,cy,k
            cam_id = db.write_camera(cam)
            if single_camera:
                shared_cam_id = cam_id
        image_ids[name] = db.write_image(pycolmap.Image(name=name, camera_id=cam_id))

    for name, image_id in image_ids.items():
        db.write_keypoints(image_id, table.keypoints(name))  # raw [m,4] float32

    for name0, name1, m in indexed:
        id0, id1 = image_ids[name0], image_ids[name1]
        assert id0 < id1, (name0, name1)
        db.write_matches(id0, id1, m)  # raw [m,2] uint32

    db.close()
    return image_ids


def verify(db_path: Path, pairs_path: Path, max_num_trials: int = 20000,
           min_inlier_ratio: float = 0.1):
    """Populate two_view_geometries. Required — COLMAP reconstructs from this table only."""
    pycolmap.verify_matches(
        str(db_path), str(pairs_path),
        options=dict(ransac=dict(max_num_trials=max_num_trials,
                                 min_inlier_ratio=min_inlier_ratio)),
    )

def prune_by_inliers(db_path: Path, min_inliers: int) -> int:
    """Delete verified two-view geometries with fewer than min_inliers inlier matches.
    The mapper reads only two_view_geometries, so this removes weak pairs (e.g. keyframe
    bridges that 'verified' on repetitive/reflective structure) from reconstruction.
    Returns the number of pairs kept.
    """
    db = pycolmap.Database.open(str(db_path))
    pair_ids, num_inliers = db.read_two_view_geometry_num_inliers()
    kept = 0
    for pid, n in zip(pair_ids, num_inliers):
        if n >= min_inliers:
            kept += 1
            continue
        id1, id2 = pycolmap.pair_id_to_image_pair(pid)
        db.delete_two_view_geometry(id1, id2)
    db.close()
    return kept