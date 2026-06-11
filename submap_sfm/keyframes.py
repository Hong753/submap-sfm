"""Keyframe selection: find foreign frames that overlap a local submap.

Brute-forcing every A x B pair is wasteful. Both submaps are continuous
trajectories, so:

  1. Coarse pass — subsample each trajectory every `stride` frames and match
     that small grid. Overlap shows up as pairs whose verified inlier count
     clears `min_inliers`.
  2. Densify — a real overlap is a contiguous run, so the only other frames
     that can overlap are the ones skipped *around* the coarse hits. Re-match
     the +/- `stride` neighbourhood of every hit; consecutive hits' neighbour-
     hoods overlap, so each contiguous segment is recovered in full, and
     multiple separate segments are handled for free (each from its own clump
     of hits).
  3. Collect — every frame that appears in any pair over `min_inliers` is a
     keyframe. No fixed count; the threshold decides.

`score(pairs)` is injected. It takes [(name_a, name_b), ...] and returns
{(name_a, name_b): num_verified_inliers}; pairs that fail verification map to 0
or are omitted. See scripts/select_keyframes.py for wiring.
"""

from __future__ import annotations

from typing import Callable

Pair = tuple[str, str]
ScoreFn = Callable[[list[Pair]], dict[Pair, int]]


def _neighbourhood(idx: int, n: int, radius: int) -> range:
    return range(max(0, idx - radius), min(n, idx + radius + 1))

def select_keyframes(
    names_a: list[str],
    names_b: list[str],
    score: ScoreFn,
    stride: int = 20,
    min_inliers: int = 100,
) -> tuple[list[str], list[str]]:
    """Return (a_overlap, b_overlap): frames of each submap that overlap the other.

    a_overlap are A-frames that see B; b_overlap are B-frames that see A. By
    construction every scored pair is (A-frame, B-frame), so the two sides never
    get confused.
    """
    idx_a = {name: i for i, name in enumerate(names_a)}
    idx_b = {name: i for i, name in enumerate(names_b)}

    # 1. coarse grid over the subsampled trajectories
    coarse = [(a, b) for a in names_a[::stride] for b in names_b[::stride]]
    scores: dict[Pair, int] = dict(score(coarse))
    hits = [pair for pair, n in scores.items() if n >= min_inliers]

    # 2. densify the +/- stride neighbourhood around every hit (set dedups the
    #    overlapping blocks, so cost tracks the real overlap area, not #hits)
    fine: set[Pair] = set()
    for a, b in hits:
        for ia in _neighbourhood(idx_a[a], len(names_a), stride):
            for ib in _neighbourhood(idx_b[b], len(names_b), stride):
                pair = (names_a[ia], names_b[ib])
                if pair not in scores:
                    fine.add(pair)
    if fine:
        scores.update(score(sorted(fine)))

    # 3. collect every frame that appears in a pair over threshold
    a_overlap: set[str] = set()
    b_overlap: set[str] = set()
    for (a, b), n in scores.items():
        if n >= min_inliers:
            a_overlap.add(a)
            b_overlap.add(b)
    return sorted(a_overlap), sorted(b_overlap)