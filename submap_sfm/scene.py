import yaml
from dataclasses import dataclass


@dataclass
class Scene:
    root: str
    submaps: list[str]
    overlaps: list[tuple[str, str]]

    @classmethod
    def load(cls, path: str) -> "Scene":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(
            root=d["scene_root"],
            submaps=d["submaps"],
            overlaps=[tuple(e) for e in d.get("overlaps", [])],
        )

    def neighbors(self, submap: str) -> list[str]:
        """Submaps that overlap `submap` (undirected)."""
        out: list[str] = []
        for a, b in self.overlaps:
            if a == submap and b not in out:
                out.append(b)
            elif b == submap and a not in out:
                out.append(a)
        return out