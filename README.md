# submap-sfm
Large-scale Structure-from-Motion by reconstructing submaps independently and
merging their poses (Sim(3) + global BA). COLMAP backend, vismatch for matching.

```
# SSH
git clone git@github.com:Hong753/submap-sfm.git --recursive
cd submap-sfm

conda create -n submap-sfm python=3.12 -y
conda activate submap-sfm

# Optional
conda install spyder -y

pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128


pip install -e . --no-build-isolation
```

```
submap-sfm/
├── submap_sfm/
│   ├── __init__.py
│   ├── pairs.py          # pair generation (this is what we just wrote)
│   ├── matching.py       # run a vismatch matcher over the pair list
│   ├── colmap_db.py      # write keypoints/matches into a COLMAP database
│   ├── reconstruct.py    # pycolmap incremental mapping per submap
│   └── merge.py          # model_merger + global BA
├── scripts/
│   ├── build_pairs.py    # CLI: build the three pair lists
│   ├── run_submap.py
│   └── run_merge.py
├── configs/
│   └── .gitkeep
├── third_party/          # vismatch submodule goes here (not added yet)
├── pyproject.toml
├── README.md
└── .gitignore
```

```
{SCENE_ROOT}/
├── hub_left/images/00000.jpg ...
├── hub_right/images/00000.jpg ...
├── keyframes/
│   ├── hub_left_aug.txt     # prefixed names sourced from hub_right (+ any other neighbor)
│   └── hub_right_aug.txt    # prefixed names sourced from hub_left  (+ any other neighbor)
├── hub_left_aug/pairs.txt
├── hub_right_aug/pairs.txt
└── full_scene/pairs.txt
```