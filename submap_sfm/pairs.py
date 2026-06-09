"""Pair generation for submap-sfm.

Two matching relationships, per the project design:

* Sequential (intra-trajectory): each image is matched to its `window`
  temporal neighbours. Captures the real overlap inside a continuous capture.
* Keyframe (exhaustive bridge): a small, manually selected set matched against
  every other image in the scene. These anchors connect non-sequential parts
  (left<->right) and are what the merge step uses to estimate the Sim(3).

Same builder for both settings:
  augmented submap -> one sequential group + keyframes
      hub_left_aug = [1941 sequential] + [20 keyframes]
  full scene       -> several sequential groups + keyframes
      [1941 left] + [2755 right] + [20 keyframes]

Pairs are unordered and de-duplicated, so any overlap between the sequential
and keyframe sets is removed automatically and the counts come out exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

Pair = tuple[str, str]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

def list_images(image_dir: str | Path, exts: Iterable[str] = IMAGE_EXTS) -> list[str]:
    """Image file names in a directory, lexicographically sorted.

    NOTE: sequential pairing assumes this order matches capture order, so
    frames must be zero-padded (frame_00001.jpg, ...). If they aren't, sort
    them yourself and pass explicit lists instead.
    """
    image_dir = Path(image_dir)
    exts = tuple(e.lower() for e in exts)
    return sorted(p.name for p in image_dir.iterdir() if p.suffix.lower() in exts)


def _canonical(a: str, b: str) -> Pair:
    return (a, b) if a <= b else (b, a)


def sequential_pairs(images: list[str], window: int) -> set[Pair]:
    """Pairs within one ordered trajectory: (i, j) for 0 < j - i <= window."""
    pairs: set[Pair] = set()
    n = len(images)
    for i in range(n):
        for j in range(i + 1, min(i + window + 1, n)):
            pairs.add(_canonical(images[i], images[j]))
    return pairs


def exhaustive_pairs(query: list[str], targets: list[str]) -> set[Pair]:
    """Each query image paired with every target image (no self-pairs)."""
    pairs: set[Pair] = set()
    for q in query:
        for t in targets:
            if q != t:
                pairs.add(_canonical(q, t))
    return pairs


def _scene_images(sequential_groups: list[list[str]], keyframes: list[str]) -> list[str]:
    seen: set[str] = set()
    scene: list[str] = []
    for group in sequential_groups:
        for name in group:
            if name not in seen:
                seen.add(name)
                scene.append(name)
    for kf in keyframes:
        if kf not in seen:
            seen.add(kf)
            scene.append(kf)
    return scene


def build_scene_pairs(
    sequential_groups: list[list[str]],
    keyframes: list[str] | None = None,
    window: int = 20,
) -> list[Pair]:
    """Build the full pair list for a scene.

    sequential_groups: one list of image names per trajectory; sequential
        matching with `window` is applied inside each.
    keyframes: names matched exhaustively against every image in the scene
        (and each other). May overlap with the groups (full-scene case) or be
        extra images (augmented submap) -- duplicates are handled either way.

    Returns a sorted list of unique, unordered (name0, name1) pairs.
    """
    keyframes = keyframes or []
    scene = _scene_images(sequential_groups, keyframes)

    pairs: set[Pair] = set()
    for group in sequential_groups:
        pairs |= sequential_pairs(group, window)
    if keyframes:
        pairs |= exhaustive_pairs(keyframes, scene)
    return sorted(pairs)


def summarize(
    sequential_groups: list[list[str]],
    keyframes: list[str] | None = None,
    window: int = 20,
) -> dict:
    """Pair-count breakdown for the report (sequential / keyframe / total)."""
    keyframes = keyframes or []
    scene = _scene_images(sequential_groups, keyframes)
    seq: set[Pair] = set()
    for group in sequential_groups:
        seq |= sequential_pairs(group, window)
    kf = exhaustive_pairs(keyframes, scene) if keyframes else set()
    total = seq | kf
    return {
        "images": len(scene),
        "sequential_pairs": len(seq),
        "keyframe_pairs": len(kf),
        "overlap_removed": len(seq) + len(kf) - len(total),
        "total_pairs": len(total),
    }


def write_pairs(pairs: list[Pair], path: str | Path) -> None:
    """Write pairs, one 'name0 name1' per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for a, b in pairs:
            f.write(f"{a} {b}\n")


def read_pairs(path: str | Path) -> list[Pair]:
    pairs: list[Pair] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                a, b = line.split()
                pairs.append((a, b))
    return pairs


def read_list(path: str | Path) -> list[str]:
    """Newline-separated names, e.g. your manually selected keyframes."""
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]