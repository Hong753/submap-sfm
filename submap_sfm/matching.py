"""Run a vismatch matcher over a unit's pairs and yield per-pair matched coordinates.

vismatch is pair-level: each call returns matched point COORDINATES (not indices).
We return them in original-image pixel coordinates; the COLMAP-index bridge
(keypoint merging) lives in colmap_db.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from vismatch import get_matcher  # third_party/vismatch, installed editable


@dataclass
class PairMatch:
    name0: str          # image name relative to scene root, e.g. "hub_left/images/00000.jpg"
    name1: str
    mkpts0: np.ndarray  # (N, 2) float32, original-resolution pixel coords in image 0
    mkpts1: np.ndarray  # (N, 2) float32, original-resolution pixel coords in image 1


def _get(result: dict, *keys):
    for k in keys:
        if k in result and result[k] is not None:
            return result[k]
    raise KeyError(f"none of {keys} in matcher result (have {list(result)})")


def load_matcher(name: str = "superpoint-lightglue", device: str = "cuda", img_size: int = 1024):
    """Sparse/fast (high-volume intra): 'superpoint-lightglue', 'aliked-lightglue', 'disk-lightglue'.
       Dense/slow (hard inter, debug):  'roma', 'master'.
    Verify exact names before a long run; get_matcher() raises with the valid list on a bad name."""
    matcher = get_matcher(name, device=device)
    matcher.im_size = img_size
    return matcher


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (W, H)

def match_pairs(matcher, scene_root: Path, pairs, img_size: int = 1024,
                min_matches: int = 15, progress: bool = True) -> Iterator[PairMatch]:
    """Yield a PairMatch per pair clearing `min_matches`.

    Matched coords come back in the matcher's RESIZED frame (vismatch stretches to a
    square img_size x img_size), so we rescale per-axis back to original resolution.
    """
    it = pairs
    if progress:
        from tqdm import tqdm
        it = tqdm(pairs, desc="matching", unit="pair")

    for name0, name1 in it:
        img0 = matcher.load_image(str(scene_root / name0), resize=img_size)
        img1 = matcher.load_image(str(scene_root / name1), resize=img_size)
        result = matcher(img0, img1)
        mkpts0 = np.asarray(result["matched_kpts0"], dtype=np.float32)
        mkpts1 = np.asarray(result["matched_kpts1"], dtype=np.float32)
        (W0, H0), (W1, H1) = image_size(scene_root / name0), image_size(scene_root / name1)
        rH0, rW0 = tuple(img0.shape)[-2:]
        rH1, rW1 = tuple(img1.shape)[-2:]
        mkpts0 *= np.array([W0 / rW0, H0 / rH0], dtype=np.float32)
        mkpts1 *= np.array([W1 / rW1, H1 / rH1], dtype=np.float32)
        if mkpts0.shape[0] < min_matches:
            continue
        yield PairMatch(name0, name1, mkpts0, mkpts1)

def visualize_pair(pm: PairMatch, scene_root: Path, out_path: Path, max_draw: int = 10_000):
    """Debug: draw matches on the ORIGINAL-resolution images. If points land on
    corresponding content, coordinate handling is correct."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    im0 = np.asarray(Image.open(scene_root / pm.name0).convert("RGB"))
    im1 = np.asarray(Image.open(scene_root / pm.name1).convert("RGB"))
    h = max(im0.shape[0], im1.shape[0])
    canvas = np.zeros((h, im0.shape[1] + im1.shape[1], 3), dtype=np.uint8)
    canvas[: im0.shape[0], : im0.shape[1]] = im0
    canvas[: im1.shape[0], im0.shape[1] :] = im1
    off = im0.shape[1]

    n = min(max_draw, pm.mkpts0.shape[0])
    idx = np.linspace(0, pm.mkpts0.shape[0] - 1, n).astype(int)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(canvas)
    for i in idx:
        ax.plot([pm.mkpts0[i, 0], pm.mkpts1[i, 0] + off],
                [pm.mkpts0[i, 1], pm.mkpts1[i, 1]], "-", lw=0.4, alpha=0.5)
    ax.scatter(pm.mkpts0[idx, 0], pm.mkpts0[idx, 1], s=3, c="lime")
    ax.scatter(pm.mkpts1[idx, 0] + off, pm.mkpts1[idx, 1], s=3, c="lime")
    ax.set_title(f"{pm.name0}  <->  {pm.name1}   ({pm.mkpts0.shape[0]} matches)")
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)