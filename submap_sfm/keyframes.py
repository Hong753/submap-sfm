"""Keyframe selection: find foreign frames that overlap a local submap.

The A x B grid of match counts is a smooth field — a near-zero background with
contiguous bumps where the trajectories overlap. So we sample coarse-to-fine:

  * Match a `stride`-subsampled grid of A x B.
  * Repeatedly halve the stride and re-test only the neighbourhoods of cells that
    matched, so off-band regions are only ever probed coarsely while the overlap
    bumps get refined down to full resolution.
  * Every frame in a pair over `min_inliers` is a keyframe (no fixed count);
    prune keyframes.txt by hand afterwards if a band gives more than you need.

`score(pairs, stride)`: match the pairs, return {(name_a, name_b): match_count}.
`stride` is the current grid level, passed only so the caller can label progress.
"""
from __future__ import annotations
from typing import Callable

Pair = tuple[str, str]
ScoreFn = Callable[[list[Pair], int], dict[Pair, int]]


def _grid(idx: int, n: int, radius: int, step: int) -> range:
    return range(max(0, idx - radius), min(n, idx + radius + 1), step)


def select_keyframes(
    names_a: list[str],
    names_b: list[str],
    score: ScoreFn,
    stride: int = 50,
    min_inliers: int = 100,
    explore: int | None = None,
) -> tuple[list[str], list[str]]:
    """Return (a_overlap, b_overlap): frames of each submap that overlap the other.

    `explore` is the looser bar for deciding *where to refine* (default half of
    `min_inliers`), so a weak-but-real band cell still gets drilled even if it
    wouldn't itself clear the keyframe threshold.
    """
    idx_a = {n: i for i, n in enumerate(names_a)}
    idx_b = {n: i for i, n in enumerate(names_b)}
    if explore is None:
        explore = max(1, min_inliers // 2)

    scores: dict[Pair, int] = {}

    def run(pairs, s):
        new = [p for p in pairs if p not in scores]
        if new:
            scores.update(score(new, s))

    level = [(a, b) for a in names_a[::stride] for b in names_b[::stride]]
    run(level, stride)

    while stride > 1:
        step = max(1, stride // 2)
        cand: set[Pair] = set()
        for a, b in level:
            if scores.get((a, b), 0) < explore:
                continue
            for ia in _grid(idx_a[a], len(names_a), stride, step):
                for ib in _grid(idx_b[b], len(names_b), stride, step):
                    cand.add((names_a[ia], names_b[ib]))
        level = list(cand)
        run(level, step)
        stride = step

    a_overlap = sorted({a for (a, b), n in scores.items() if n >= min_inliers})
    b_overlap = sorted({b for (a, b), n in scores.items() if n >= min_inliers})
    return a_overlap, b_overlap