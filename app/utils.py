from dataclasses import dataclass
from typing import Tuple
import time
import numpy as np


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    cls: int = -1

    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def area(self) -> float:
        return self.width() * self.height()

    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def top_y(self) -> float:
        return self.y1

    def bottom_y(self) -> float:
        return self.y2

    def foot_point(self) -> Tuple[float, float]:
        # P_foot = bottom-center of bbox
        return (self.x1 + self.x2) / 2.0, self.y2


def iou(a: BBox, b: BBox) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = a.area() + b.area() - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def now_ts() -> float:
    return time.time()


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def to_int_tuple(pt):
    return int(pt[0]), int(pt[1])


def poly_to_np(poly):
    return np.array(poly, dtype=np.int32).reshape((-1, 1, 2))