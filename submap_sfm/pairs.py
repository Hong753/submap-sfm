"""Pair generation for submap-sfm.

Two pair categories per augmented submap:

* intra (node): pairs within one submap's own trajectory. Currently sequential
  matching with `window`; "intra" is the stable concept, the algorithm may change.
* inter (keyframe -> local): manually selected keyframes -- neighbour frames that
  overlap this submap -- matched against the local trajectory (never against each
  other). These bridge the submaps and anchor the Sim(3) merge.

Keyframes are authored by hand, one name per line, in {submap}_aug/keyframes.txt
(see README). The full-scene baseline is the de-duplicated union of the
augmented submaps' pair sets.

Pairs are unordered (canonicalised) and stored in sets, so reverse-ordered
duplicates and any intra/inter overlap collapse automatically.
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
    e.g. 'hub_left/images/00000.jpg'. NOTE: sequential pairing assumes
    lexicographic order matches capture order (zero-padded names).
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


def node_pairs(images: list[str], window: int = 20) -> set[Pair]:
    """Within-node (intra) pairs for one submap. Currently sequential."""
    return sequential_pairs(images, window)


def keyframe_pairs(keyframes: list[str], local: list[str], stride: int = 1) -> set[Pair]:
    """Inter pairs: keyframes matched against the local trajectory.

    Keyframes are neighbour frames overlapping this submap; `local` is this
    submap's own trajectory. Keyframes are never paired with each other.
    `stride` subsamples the local trajectory (1 = every frame); raise it to cut
    matching cost at some registration-robustness cost.
    """
    kf_set = set(keyframes)
    targets = [x for x in local if x not in kf_set]
    if stride > 1:
        targets = targets[::stride]
    return exhaustive_pairs(keyframes, targets)


def write_pairs(pairs: Iterable[Pair], path: str | Path) -> None:
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