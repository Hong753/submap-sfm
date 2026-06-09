from submap_sfm.pairs import list_images, build_scene_pairs, summarize, write_pairs, read_list

SCENE_ROOT = "/data/colmap_scenes/lg_science_park/hub_galaxy_test1"
SUBMAPS_LIST = [
    "hub_left",
    "hub_right"
]

W = 20
left  = list_images("data/hub_left/images")    # 1941 names, sorted
right = list_images("data/hub_right/images")   # 2755 names, sorted
keyframes = read_list("data/keyframes.txt")    # your 20 manually selected names

# merge approach: two augmented submaps, SAME keyframe set in both
left_aug  = build_scene_pairs([left],  keyframes, window=W)
right_aug = build_scene_pairs([right], keyframes, window=W)
write_pairs(left_aug,  "outputs/pairs/hub_left_aug.txt")
write_pairs(right_aug, "outputs/pairs/hub_right_aug.txt")

# full scene baseline
full = build_scene_pairs([left, right], keyframes, window=W)
write_pairs(full, "outputs/pairs/full_scene.txt")

for name, groups in [("left_aug", [left]), ("right_aug", [right]), ("full", [left, right])]:
    print(name, summarize(groups, keyframes, W))