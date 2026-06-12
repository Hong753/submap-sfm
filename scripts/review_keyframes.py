"""Interactive review of auto-selected keyframes, one overlap at a time.

List overlaps:        python scripts/review_keyframes.py
Review one overlap:   python scripts/review_keyframes.py --overlap hub_left hub_right

Steps through that overlap's keyframe overlays; you mark keep/remove per frame.
NOTHING is written until you finish, so you can toggle freely. On quit it strips
removed frames from each keyframes.txt (other overlaps' frames in a shared unit
are left alone) and deletes their overlay PNGs.

Keys:  <- / ->  prev / next      k  keep      r  remove      space  toggle      q  finish

Needs an interactive matplotlib backend (a display, Qt or Tk). In Spyder set the
graphics backend to a separate window rather than inline.
"""
import os
import argparse
from collections import Counter, defaultdict

import matplotlib

if matplotlib.get_backend().lower() == "agg":          # grab a GUI backend if possible
    for _b in ("QtAgg", "TkAgg"):
        try:
            matplotlib.use(_b)
            break
        except Exception:
            pass
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# our handler owns the keyboard — clear matplotlib's defaults (q=quit, r/h=home,
# k/l=scale, left/right=back/forward, s=save, ...) so they don't fire alongside it
for _k in [k for k in plt.rcParams if k.startswith("keymap.")]:
    plt.rcParams[_k] = []

from submap_sfm.scene import Scene

CONFIG = "configs/default.yaml"
# ---------------------------------------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument("--overlap", nargs=2, metavar=("A", "B"),
                help="the two submaps to review, e.g. --overlap hub_left hub_right")
args = ap.parse_args()

scene = Scene.load(CONFIG)
units_root = os.path.join(scene.root, "units")
aug_units = sorted({f"{s}_aug" for o in scene.overlaps for s in o})


def _kpath(unit):
    return os.path.join(units_root, unit, "keyframes.txt")


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def _write(path, names):
    with open(path, "w") as f:
        f.write("\n".join(names) + ("\n" if names else ""))


# gather every keyframe across both sides of every overlap
items = []
for unit in aug_units:
    local = unit[:-4]
    dbg = os.path.join(units_root, unit, "debug", "keyframes")
    for name in _read(_kpath(unit)):
        foreign = name.split("/")[0]
        items.append({
            "unit": unit,
            "name": name,
            "png": os.path.join(dbg, name.replace("/", "_") + ".png"),
            "overlap": "<->".join(sorted([local, foreign])),
        })

by_overlap = Counter(it["overlap"] for it in items)
if not by_overlap:
    raise SystemExit("no keyframes found — run select_keyframes.py first")

if not args.overlap:
    print("keyframes per overlap — review one at a time:")
    for ov, n in sorted(by_overlap.items()):
        a, b = ov.split("<->")
        print(f"  {ov}: {n:5d}   ->  python scripts/review_keyframes.py --overlap {a} {b}")
    raise SystemExit

chosen = "<->".join(sorted(args.overlap))
if chosen not in by_overlap:
    raise SystemExit(f"no keyframes for '{chosen}'. available: {sorted(by_overlap)}")
items = [it for it in items if it["overlap"] == chosen]
items.sort(key=lambda it: (it["unit"], it["name"]))
print(f"reviewing {chosen}: {len(items)} keyframes")


class Reviewer:
    def __init__(self, items):
        self.items = items
        self.i = 0
        self.keep = [True] * len(items)               # decision per item, in memory only
        self.confirm = False                          # showing the quit prompt?
        self.confirmed = False                        # user pressed y -> apply
        self.fig, self.ax = plt.subplots(figsize=(16, 8))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

    def draw(self):
        it = self.items[self.i]
        self.ax.clear()
        try:
            self.ax.imshow(mpimg.imread(it["png"]))
        except Exception:
            self.ax.text(0.5, 0.5, "(overlay image missing)",
                         ha="center", va="center", transform=self.ax.transAxes)
        if self.confirm:
            n_remove = self.keep.count(False)
            title = (f"Quit and apply?  this removes {n_remove} keyframe(s)\n"
                     f"y = yes, apply      any other key = no, keep reviewing")
        else:
            state = "KEEP" if self.keep[self.i] else "REMOVE"
            title = (f"[{self.i + 1}/{len(self.items)}]  {it['overlap']}   "
                     f"{it['unit']} <- {it['name']}   [{state}]\n"
                     f"<-/->  navigate    k keep    r remove    space toggle    q finish")
        self.ax.set_title(title, fontsize=10)
        self.ax.axis("off")
        print(f"\rkeep {sum(self.keep)} / {len(self.items)}    ", end="", flush=True)
        self.fig.canvas.draw_idle()

    def on_key(self, e):
        if self.confirm:                              # answering the quit prompt
            if e.key == "y":
                self.confirmed = True
                plt.close(self.fig)
                return
            self.confirm = False                      # anything else cancels
            self.draw()
            return
        if e.key in ("right", "d"):
            self.i = min(self.i + 1, len(self.items) - 1)
        elif e.key in ("left", "a"):
            self.i = max(self.i - 1, 0)
        elif e.key == "k":
            self.keep[self.i] = True
        elif e.key == "r":
            self.keep[self.i] = False
        elif e.key in (" ", "x"):
            self.keep[self.i] = not self.keep[self.i]
        elif e.key == "q":
            self.confirm = True
            self.draw()
            return
        self.draw()

    def apply(self):
        removed = defaultdict(set)
        for it, k in zip(self.items, self.keep):
            if not k:
                removed[it["unit"]].add(it["name"])
        for unit in sorted({it["unit"] for it in self.items}):
            kept = [n for n in _read(_kpath(unit)) if n not in removed[unit]]
            _write(_kpath(unit), kept)
        for it, k in zip(self.items, self.keep):
            if not k:
                try:
                    os.remove(it["png"])
                except FileNotFoundError:
                    pass
        total = sum(len(v) for v in removed.values())
        print(f"\n--- {chosen} done: removed {total}, kept {len(self.items) - total} ---")


reviewer = Reviewer(items)
plt.show()
if reviewer.confirmed:
    reviewer.apply()
else:
    print("\ncancelled — nothing written")