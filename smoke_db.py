import numpy as np, pycolmap
from pathlib import Path

p = Path("/tmp/smoke.db")
if p.exists():
    p.unlink()

db = pycolmap.Database.open(str(p))

cam = pycolmap.Camera(model="SIMPLE_RADIAL", width=640, height=480, params=[700, 320, 240, 0.0])
cid = db.write_camera(cam)

ids = [db.write_image(pycolmap.Image(name=f"img{i}.jpg", camera_id=cid)) for i in range(2)]

# keypoints: [m, 4] float32 -> x, y, scale, orientation (only x,y are used downstream)
kp = np.zeros((200, 4), dtype=np.float32)
kp[:, :2] = (np.random.rand(200, 2) * [640, 480]).astype(np.float32) + 0.5
kp[:, 2] = 1.0
for i in ids:
    db.write_keypoints(i, kp)

# matches: [m, 2] uint32, zero-based indices into each image's keypoints
m = np.column_stack([np.arange(200), np.arange(200)]).astype(np.uint32)
db.write_matches(ids[0], ids[1], m)

db.close()
print("write OK ->", p)