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

pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install pycolmap pyyaml
pip install -e third_party/vismatch --no-build-isolation
pip install -e . --no-build-isolation
```

(The vismatch submodule and remaining dependencies are added later; once the
submodule exists, clone with `git clone --recursive`.)

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
│   ├── make_random_keyframes.py
│   ├── build_pairs.py
│   ├── run_submap.py
│   └── run_merge.py
├── configs/
│   └── default.yaml      # scene definition
├── third_party/          # vismatch submodule (added later)
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

## Data layout (SCENE_ROOT)

```
{SCENE_ROOT}/
├── hub_left/images/00000.jpg ...      # source images
├── hub_right/images/00000.jpg ...     # source images
├── hub_left_aug/
│   ├── keyframes.txt                  # bridge frames (from hub_right)
│   ├── pairs_intra.txt                # within-node pairs
│   ├── pairs_inter.txt                # node-to-edge pairs
│   └── pairs.txt                      # combined (for matching)
├── hub_right_aug/
│   └── ...                            # same files
└── full_scene/
    └── pairs.txt
```

Each reconstruction unit later also holds `database.db` and `sparse/`.

## Usage

```
# 1. select keyframes (random placeholder for now)
python scripts/make_random_keyframes.py

# 2. build pair lists for each unit
python scripts/build_pairs.py
```