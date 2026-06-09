"""Pair generation for submap-sfm.

Two pair categories:

* intra (within-node): pairs inside a single submap's trajectory. Currently
  produced by sequential matching with `window`, but the local method may
  change later -- the category is the stable concept, not the algorithm.
* inter (node-to-edge): keyframe images matched against the local trajectory
  only (never against each other). The local map is sequential, so consecutive
  frames are highly redundant; `inter_stride` subsamples the trajectory so each
  keyframe matches every Nth local frame instead of all of them. These bridge
  non-sequential parts (left<->right) and anchor the Sim(3) estimate at merge.

Per augmented submap, build the pairs with one sequential group + keyframes:
    hub_left_aug = [1941 sequential] + [20 keyframes drawn from hub_right]

For the full-scene baseline, take the de-duplicated per-category union of the
submap pair sets; that performs exactly the same matches as the merge, in a
single reconstruction.

Pairs are unordered (canonicalised) and stored in a set, so reverse-ordered
duplicates and any intra/inter overlap are removed automatically.
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


def _scene_pair_sets(
    sequential_groups: list[list[str]],
    keyframes: list[str] | None,
    window: int,
    inter_stride: int = 1,
) -> tuple[set[Pair], set[Pair]]:
    """Internal: (intra_set, inter_set) for a scene.

    inter_stride: subsample the local trajectory before matching it against
        keyframes. The local map is sequential, so consecutive frames are highly
        redundant -- each keyframe only needs to match every Nth local frame to
        register. 1 = every frame (exhaustive); 10 = every 10th frame.
    """
    keyframes = keyframes or []
    kf_set = set(keyframes)
    local = _unique(sequential_groups)
    targets = [x for x in local if x not in kf_set]  # local trajectory, no keyframes
    if inter_stride > 1:
        targets = targets[::inter_stride]            # sample every Nth local frame
    intra: set[Pair] = set()
    for group in sequential_groups:
        intra |= sequential_pairs(group, window)
    inter = exhaustive_pairs(keyframes, targets) if keyframes else set()
    return intra, inter


def build_scene_pairs(
    sequential_groups: list[list[str]],
    keyframes: list[str] | None = None,
    window: int = 20,
    inter_stride: int = 1,
) -> list[Pair]:
    """Combined pair list (intra + inter), sorted and de-duplicated."""
    intra, inter = _scene_pair_sets(sequential_groups, keyframes, window, inter_stride)
    return sorted(intra | inter)


def build_scene_pairs_split(
    sequential_groups: list[list[str]],
    keyframes: list[str] | None = None,
    window: int = 20,
    inter_stride: int = 1,
) -> dict[str, list[Pair]]:
    """Same pairs, split by category for separate inspection/visualization.

    Returns {"intra": [...], "inter": [...]} where:
      intra = within-node  (currently sequential, method may change)
      inter = node-to-edge (keyframe -> local trajectory, strided)
    """
    intra, inter = _scene_pair_sets(sequential_groups, keyframes, window, inter_stride)
    return {"intra": sorted(intra), "inter": sorted(inter)}


def summarize(
    sequential_groups: list[list[str]],
    keyframes: list[str] | None = None,
    window: int = 20,
    inter_stride: int = 1,
) -> dict:
    """Pair-count breakdown for the report."""
    keyframes = keyframes or []
    local = _unique(sequential_groups)
    scene = _unique([local, keyframes])
    intra, inter = _scene_pair_sets(sequential_groups, keyframes, window, inter_stride)
    total = intra | inter
    return {
        "images": len(scene),
        "intra_pairs": len(intra),
        "inter_pairs": len(inter),
        "overlap_removed": len(intra) + len(inter) - len(total),
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