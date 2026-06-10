"""Incremental sparse reconstruction for one unit (COLMAP mapper via pycolmap).

Reads a unit's database.db (cameras + images + verified two-view geometries) and
runs incremental mapping, writing the recovered model(s) to {output}/0, /1, ...
COLMAP can produce multiple disconnected models on hard scenes, so we return the
dict and let the caller pick the largest.
"""

from __future__ import annotations

from pathlib import Path

import pycolmap


def reconstruct(
    database_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    min_model_size: int = 10,
    num_threads: int = -1,
) -> dict[int, "pycolmap.Reconstruction"]:
    """Run incremental mapping. Returns {index: Reconstruction} (may be empty).

    image_path is the COLMAP image root; for this repo that is SCENE_ROOT, since
    the DB stores names relative to it (e.g. 'hub_left/images/00000.jpg').
    Models are written to {output_path}/0, /1, ...

    Requires a *verified* database (run_matching runs verification), since COLMAP
    triangulates only from two-view geometries.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    options = pycolmap.IncrementalPipelineOptions()
    options.min_model_size = min_model_size
    options.num_threads = num_threads

    return pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(image_path),
        output_path=str(output_path),
        options=options,
    )