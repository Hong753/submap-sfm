"""Pair generation for submap-sfm.

Two matching relationships, per the project design:

* Sequential (intra-trajectory): each image is matched to its `window`
  temporal neighbours. Captures the real overlap inside a continuous capture.
* Keyframe (bridge): a small, manually selected set matched against the local
  trajectory images only (never against each other). These anchors connect
  non-sequential parts (left<->right) and are what the merge step uses to
  estimate the Sim(3).

Per augmented submap, build the pairs with one sequential group + keyframes:
    hub_left_aug = [1941 sequential] + [20 keyframes drawn from hub_right]

For the full-scene baseline, take the de-duplicated union of the submap pair
sets (`set(left_aug) | set(right_aug)`); that performs exactly the same matches
as the merge, in a single reconstruction.

Pairs are unordered (canonicalised) and stored in a set, so reverse-ordered
duplicates and any sequential/keyframe overlap are removed automatically and
the counts come out exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

Pair = tuple[str, str]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def list_images(
    image_dir: str | Path,
    root: str | Path | None = None,
    exts: Iterable[str] = IMAGE_EXTS,
) -> list[str]:
    """Image file names, lexicographically sorted.

    If `root` is given, each name is the file's path relative to `root` (POSIX),
    e.g. 'hub_left/images/00000.jpg'. This keeps names unique across submaps and
    lets you use `root` directly as COLMAP's image_path. NOTE: sequential pairing
    assumes lexicographic order matches capture order (zero-padded names).
    """
    image_dir = Path(image_dir)
    exts = tuple(e.lower() for e in exts)
    files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    if root is None:
        return [p.name for p in files]
    root = Path(root)
    return [p.relative_to(root).as_posix() for p in files]


def _canonical(a: str, b: str) -> Pair:
    return (a, b) if a <= b else (b, a)


def _unique(groups: Iterable[list[str]]) -> list[str]:
    """Flatten lists of names into a de-duplicated, order-preserving list."""
    seen: set[str] = set()
    out: list[str] = []
    for g in groups:
        for n in g:
            if n not in seen:
                seen.add(n)
                out.append(n)
    return out


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


def build_scene_pairs(
    sequential_groups: list[list[str]],
    keyframes: list[str] | None = None,
    window: int = 20,
) -> list[Pair]:
    """Build the pair list for a scene.

    sequential_groups: one list of image names per trajectory; sequential
        matching with `window` is applied inside each.
    keyframes: names matched exhaustively against the local trajectory images
        only (the members of sequential_groups), and never against each other.

    Returns a sorted list of unique, unordered (name0, name1) pairs.
    """
    keyframes = keyframes or []
    kf_set = set(keyframes)
    local = _unique(sequential_groups)
    targets = [x for x in local if x not in kf_set]  # local trajectory, no keyframes

    pairs: set[Pair] = set()
    for group in sequential_groups:
        pairs |= sequential_pairs(group, window)
    if keyframes:
        pairs |= exhaustive_pairs(keyframes, targets)
    return sorted(pairs)


def summarize(
    sequential_groups: list[list[str]],
    keyframes: list[str] | None = None,
    window: int = 20,
) -> dict:
    """Pair-count breakdown for the report (sequential / keyframe / total)."""
    keyframes = keyframes or []
    kf_set = set(keyframes)
    local = _unique(sequential_groups)
    targets = [x for x in local if x not in kf_set]
    scene = _unique([local, keyframes])

    seq: set[Pair] = set()
    for group in sequential_groups:
        seq |= sequential_pairs(group, window)
    kf = exhaustive_pairs(keyframes, targets) if keyframes else set()
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