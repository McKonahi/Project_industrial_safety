from dataclasses import dataclass
from typing import List, Tuple, Dict
import yaml
import cv2
import numpy as np


@dataclass
class Zone:
    zone_id: str
    name: str
    polygon: List[Tuple[int, int]]  # list of (x, y)

    def contains(self, point: Tuple[float, float]) -> bool:
        # pointPolygonTest ожидает контур Nx1x2
        contour = np.array(self.polygon, dtype=np.int32).reshape((-1, 1, 2))
        return cv2.pointPolygonTest(contour, point, False) >= 0


def load_zones(path: str) -> List[Zone]:
    data = yaml.safe_load(open(path, "r", encoding="utf-8"))
    zones = []
    for z in data.get("zones", []):
        zones.append(Zone(
            zone_id=str(z["id"]),
            name=str(z.get("name", z["id"])),
            polygon=[(int(x), int(y)) for x, y in z["polygon"]],
        ))
    return zones