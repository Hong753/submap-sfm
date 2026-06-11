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
│   ├── scene.py           # scene graph: submaps + overlaps
│   ├── pairs.py           # intra + inter pair generation
│   ├── matching.py        # run a vismatch matcher over the pair list
│   ├── colmap_db.py       # write keypoints/matches into a COLMAP database
│   ├── reconstruct.py     # pycolmap incremental mapping per unit
│   └── merge.py           # model_merger + global BA
├── scripts/
│   ├── build_pairs.py
│   ├── run_matching.py    # match a unit -> build + verify its database.db
│   ├── run_reconstruct.py # incremental mapping -> sparse/ for a unit
│   └── run_merge.py
├── configs/
│   └── default.yaml       # scene definition
├── third_party/           # vismatch submodule
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

## Keyframes (manual selection)

Keyframes are the bridge frames tying two overlapping submaps together: a handful
of frames from one submap that view the region it shares with the other. They
anchor the Sim(3) alignment at merge time, so each one must register in *both*
submaps.

Author one `keyframes.txt` per augmented submap by hand — one image name per
line, relative to `{SCENE_ROOT}/images`. A submap with more than one neighbor in
the overlap chain collects keyframes from each of them:

```
{SCENE_ROOT}/units/lounge_aug/keyframes.txt      # hub_right frames overlapping lounge
{SCENE_ROOT}/units/hub_right_aug/keyframes.txt   # lounge + hub_left frames overlapping hub_right
{SCENE_ROOT}/units/hub_left_aug/keyframes.txt    # hub_right + hallway frames overlapping hub_left
{SCENE_ROOT}/units/hallway_aug/keyframes.txt     # hub_left frames overlapping hallway
```

Example line: `hub_right/000742.jpg`

Selection guidelines:
- ~20 per overlap, all drawn from the shared region.
- Spread them across the overlap (not clustered or near-collinear) so scale and
  rotation are well constrained for the merge.
- Avoid mirrors, glass, and blank/textureless walls — prefer stable, textured,
  static geometry.
- After matching, confirm each keyframe actually registers in both submaps with
  a healthy inlier count, and replace any that don't (visually overlapping is
  not the same as matchable).

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
# 1. Author keyframes manually: create keyframes.txt in each augmented unit
#    under units/ (see "Keyframes" above).

# 2. Build pair lists for each unit (prints intra / inter / total counts)
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