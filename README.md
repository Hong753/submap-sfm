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
│   ├── scene.py          # scene graph: submaps + overlaps
│   ├── pairs.py          # intra + inter pair generation
│   ├── matching.py       # run a vismatch matcher over the pair list
│   ├── colmap_db.py      # write keypoints/matches into a COLMAP database
│   ├── reconstruct.py    # pycolmap incremental mapping per submap
│   └── merge.py          # model_merger + global BA
├── scripts/
│   ├── build_pairs.py
│   ├── run_matching.py   # match a unit -> build + verify its database.db
│   ├── run_submap.py
│   └── run_merge.py
├── configs/
│   └── default.yaml      # scene definition
├── third_party/          # vismatch submodule
├── pyproject.toml
└── README.md
```

## Config

`configs/default.yaml` defines the scene graph — the nodes (submaps) and which
ones overlap (so they get bridged + merged):

```yaml
scene_root: /data/colmap_scenes/lg_science_park/hub_galaxy_test1
submaps: [hub_left, hub_right]
overlaps:
  - [hub_left, hub_right]
```

## Keyframes (manual selection)

Keyframes are the bridge frames tying two submaps together: a handful of frames
from one submap that view the region it shares with the other. They anchor the
Sim(3) alignment at merge time, so each one must register in *both* submaps.

Author one `keyframes.txt` per augmented submap by hand — one image name per
line, relative to `SCENE_ROOT`:

```
{SCENE_ROOT}/hub_left_aug/keyframes.txt    # hub_right frames overlapping hub_left
{SCENE_ROOT}/hub_right_aug/keyframes.txt   # hub_left  frames overlapping hub_right
```

Example line: `hub_right/images/00742.jpg`

Selection guidelines:
- ~20 per submap, all drawn from the overlap region.
- Spread them across the overlap (not clustered or near-collinear) so scale and
  rotation are well constrained for the merge.
- Avoid mirrors, glass, and blank/textureless walls — prefer stable, textured,
  static geometry.
- After matching, confirm each keyframe actually registers in both submaps with
  a healthy inlier count, and replace any that don't (visually overlapping is
  not the same as matchable).

## Data layout (SCENE_ROOT)

```
{SCENE_ROOT}/
├── hub_left/images/00000.jpg ...      # source images
├── hub_right/images/00000.jpg ...     # source images
├── hub_left_aug/
│   ├── keyframes.txt                  # manual bridge frames (from hub_right)
│   ├── pairs_intra.txt                # within-node pairs
│   ├── pairs_inter.txt                # node-to-edge pairs (keyframe -> local)
│   └── pairs.txt                      # combined (for matching)
├── hub_right_aug/
│   └── ...                            # same files (keyframes from hub_left)
└── full_scene/
    └── pairs.txt
```

After matching, each unit also holds `database.db`, `debug/pairs/` (match
overlays), and later `sparse/`.

## Usage

```
# 1. Author keyframes manually: create keyframes.txt in each augmented submap
#    directory (see "Keyframes" above).

# 2. Build pair lists for each unit (prints intra / inter / total counts)
python scripts/build_pairs.py
```

### Matching

Match each unit on its combined `pairs.txt`. This builds and verifies
`{unit}/database.db` and saves overlays to `{unit}/debug/pairs/`. `--viz-every N`
saves every Nth matched pair, so set `N ≈ total_pairs / 500` (totals from step 2)
for ~500 overlays per unit:

```
python scripts/run_matching.py --unit hub_left_aug  --pairs pairs.txt --viz-every 50
python scripts/run_matching.py --unit hub_right_aug --pairs pairs.txt --viz-every 50
python scripts/run_matching.py --unit full_scene    --pairs pairs.txt --viz-every 100
```

Defaults: `--matcher superpoint-lightglue --img-size 1024 --device cuda
--min-matches 15`. Add `--debug` for per-pair match counts, or `--limit N` to
match only the first N pairs as a quick check.