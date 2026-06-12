# submap-sfm

Large-scale Structure-from-Motion by reconstructing submaps independently and
merging their poses (Sim(3) + global bundle adjustment). COLMAP backend,
[vismatch](https://github.com/gmberton/vismatch) for feature matching.

## Setup

```
# SSH
git clone git@github.com:Hong753/submap-sfm.git
cd submap-sfm
git submodule update --init --recursive

conda create -n submap-sfm python=3.12 -y
conda activate submap-sfm

# Optional
conda install spyder -y

# PyTorch — pick the ONE line matching your CUDA version
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
# or
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8

pip install -r requirements.txt
pip install -e third_party/vismatch --no-build-isolation
pip install -e . --no-build-isolation
```

## Repo structure

```
submap-sfm/
├── submap_sfm/
│   ├── __init__.py
│   ├── scene.py            # scene graph: submaps + overlaps
│   ├── pairs.py            # intra + inter pair generation
│   ├── keyframes.py        # automatic keyframe selection (overlap discovery)
│   ├── matching.py         # run a vismatch matcher over the pair list
│   ├── colmap_db.py        # write keypoints/matches into a COLMAP database
│   ├── reconstruct.py      # pycolmap incremental mapping per unit
│   └── merge.py            # model_merger + global BA
├── scripts/
│   ├── select_keyframes.py # discover overlaps -> keyframes.txt per aug unit
│   ├── review_keyframes.py # interactively keep/remove keyframes, per overlap
│   ├── build_pairs.py
│   ├── run_matching.py     # match a unit -> build + verify its database.db
│   ├── run_reconstruct.py  # incremental mapping -> sparse/ for a unit
│   └── run_merge.py
├── configs/
│   └── default.yaml        # scene definition
├── third_party/            # vismatch submodule
├── pyproject.toml
└── README.md
```

## Config

`configs/default.yaml` defines the scene graph — the submaps and which pairs of
them overlap (so they get bridged + merged):

```yaml
scene_root: /workspace/colmap_scenes/lg_science_park/full_scene
submaps: [lounge, hub_right, hub_left, hallway]
overlaps:
  - [lounge, hub_right]
  - [hub_right, hub_left]
  - [hub_left, hallway]
```

## Keyframes

Keyframes are the bridge frames tying two overlapping submaps together: frames
from one submap that view the region it shares with the other. They create the
cross-submap connections, so each one must register in *both* submaps. Discover
them automatically and prune by eye, or author them by hand. Either way the
result is one `keyframes.txt` per augmented unit under `units/`, one image name
per line, relative to `{SCENE_ROOT}/images` (e.g. `hub_right/000742.jpg`).

### Automatic selection (overlap discovery)

`scripts/select_keyframes.py` finds the overlapping frames for every pair in
`overlaps` without brute-forcing the full A×B grid. The grid of match counts is a
smooth field — near-zero almost everywhere, with contiguous bumps where the
trajectories overlap — so it's sampled coarse-to-fine:

1. **Coarse pass** — subsample both trajectories every `MAX_STRIDE` frames and
   match that small grid. Overlap shows up as pairs whose LightGlue match count
   clears `MIN_MATCHES`.
2. **Refine** — repeatedly halve the stride *down to `MIN_STRIDE`* (not to 1),
   re-matching only the neighbourhoods of cells already above a looser `explore`
   bar (half of `MIN_MATCHES`). Off-band regions stay coarsely probed; the
   overlap bumps get refined to `MIN_STRIDE` spacing.
3. **Collect** — every frame in a pair over `MIN_MATCHES` becomes a keyframe.
   Stopping at `MIN_STRIDE` caps density, so keyframes land ~`MIN_STRIDE` frames
   apart (≈ `band_length / MIN_STRIDE` per side) instead of one-per-frame
   near-duplicates.

Foreign frames go into the matching aug unit (B-frames overlapping A →
`A_aug/keyframes.txt`, and vice versa), accumulated across both neighbours for a
middle submap. Each pick is drawn as a match overlay in
`units/{unit}/debug/keyframes/` for review. Scoring reuses the same matcher
backend as `run_matching`.

```
python scripts/select_keyframes.py
```

Tunables at the top of the script: `MAX_STRIDE` (coarse grid; keep it ≤ the
shortest overlap you can't afford to miss), `MIN_STRIDE` (finest level / keyframe
spacing — raise to thin further, `1` resolves every frame), and `MIN_MATCHES`
(the "significant overlap" bar).

> Match count alone can't reject mirror reflections or repeated structure — they
> score as high as real overlap. Selection is a generous proposal; cull the false
> positives in review (below), and the final gate is whether each keyframe
> registers during reconstruction.

### Review (prune false positives)

`scripts/review_keyframes.py` steps through the overlays interactively, one
overlap at a time, so you can keep/remove each keyframe by eye. Nothing is
written until you confirm, so you can toggle freely.

```
python scripts/review_keyframes.py                      # list overlaps + counts
python scripts/review_keyframes.py --overlap hub_left hub_right
```

Keys: `←/→` navigate, `k` keep, `r` remove, `space` toggle, `q` finish (asks to
confirm — `y` applies and quits, anything else cancels). On confirm it strips the
removed frames from each `keyframes.txt` and deletes their overlay PNGs; other
overlaps' frames in a shared unit are left untouched. Needs an interactive
matplotlib backend (a display, Qt or Tk).

### Manual selection

Author each `keyframes.txt` by hand instead. A submap with more than one neighbor
in the overlap chain collects keyframes from each of them:

```
{SCENE_ROOT}/units/lounge_aug/keyframes.txt      # hub_right frames overlapping lounge
{SCENE_ROOT}/units/hub_right_aug/keyframes.txt   # lounge + hub_left frames overlapping hub_right
{SCENE_ROOT}/units/hub_left_aug/keyframes.txt    # hub_right + hallway frames overlapping hub_left
{SCENE_ROOT}/units/hallway_aug/keyframes.txt     # hub_left frames overlapping hallway
```

Example line: `hub_right/000742.jpg`

Selection guidelines:
- ~20 per overlap, all drawn from the shared region.
- Spread them across the overlap (not clustered or near-collinear).
- Avoid mirrors, glass, and blank/textureless walls — prefer stable, textured,
  static geometry.
- After matching, confirm each keyframe registers in both submaps with a healthy
  inlier count, and replace any that don't.

## Data layout (SCENE_ROOT)

Source frames are extracted from one video per submap at 1 fps (initial repeated
frames removed by hand):

```
ffmpeg -y -hwaccel auto -i hallway_20260513_075535.mp4 \
  -vf fps=1 -q:v 2 -start_number 0 \
  images/hallway/%06d.jpg
```

Inputs live under `images/`, all derived artifacts under `units/`:

```
{SCENE_ROOT}/
├── images/                         # source frames, one folder per submap
│   ├── lounge/000000.jpg ...
│   ├── hub_right/000000.jpg ...
│   ├── hub_left/000000.jpg ...
│   └── hallway/000000.jpg ...
└── units/                          # derived artifacts, one folder per reconstruction unit
    ├── lounge_aug/
    │   ├── keyframes.txt           # bridge frames from hub_right
    │   ├── pairs_intra.txt         # within-submap pairs
    │   ├── pairs_inter.txt         # keyframe -> local pairs
    │   └── pairs.txt               # combined (for matching)
    ├── hub_right_aug/
    │   └── ...                     # keyframes from lounge + hub_left
    ├── hub_left_aug/
    │   └── ...                     # keyframes from hub_right + hallway
    ├── hallway_aug/
    │   └── ...                     # keyframes from hub_left
    └── full_scene/
        └── pairs.txt
```

The COLMAP image root is `{SCENE_ROOT}/images`, so every image name — in the pair
lists, keyframes, and the database — is relative to it (e.g. `hub_left/000000.jpg`),
which keeps names unique across submaps. After matching, each unit also holds
`database.db` and `debug/pairs/` (match overlays); after reconstruction, `sparse/`.

## Usage

```
# 1. Select keyframes — automatically:
python scripts/select_keyframes.py
#    ...or author keyframes.txt by hand (see "Keyframes" above).

# 2. Review and prune the auto-selected keyframes, one overlap at a time:
python scripts/review_keyframes.py                      # lists overlaps
python scripts/review_keyframes.py --overlap hub_left hub_right

# 3. Build pair lists for each unit (prints intra / inter / total counts)
python scripts/build_pairs.py
```

### Matching

Match each unit on its combined `pairs.txt`. This builds and verifies
`units/{unit}/database.db` and saves overlays to `units/{unit}/debug/pairs/`.
`--viz-every N` saves every Nth matched pair, so set `N ≈ total_pairs / 500`
(totals from step 2) for ~500 overlays per unit. Run it for each aug unit plus
`full_scene`:

```
python scripts/run_matching.py --unit hub_right_aug --pairs pairs.txt --viz-every 50
python scripts/run_matching.py --unit hub_left_aug  --pairs pairs.txt --viz-every 50
python scripts/run_matching.py --unit full_scene    --pairs pairs.txt --viz-every 100
# ...likewise for lounge_aug and hallway_aug
```

Defaults: `--matcher superpoint-lightglue --img-size 1024 --device cuda
--min-matches 15`. Add `--debug` for per-pair match counts, or `--limit N` to
match only the first N pairs as a quick check.

### Reconstruction

Run incremental mapping on a verified unit. Models are written to
`units/{unit}/sparse/0`, `/1`, … (multiple models if COLMAP fragments); the
largest is the one to use:

```
python scripts/run_reconstruct.py --unit full_scene
```

Use `--min-model-size 3` while debugging a hard scene so small fragments survive
instead of being discarded.