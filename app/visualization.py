from typing import Dict, List, Tuple, Optional
import cv2
from .utils import BBox, to_int_tuple, poly_to_np
from .zones import Zone


def draw_bbox(frame, bbox: BBox, label: str):
    p1 = (int(bbox.x1), int(bbox.y1))
    p2 = (int(bbox.x2), int(bbox.y2))
    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
    cv2.putText(frame, label, (p1[0], max(0, p1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def draw_violation(frame, bbox: BBox, text: str):
    p1 = (int(bbox.x1), int(bbox.y1))
    p2 = (int(bbox.x2), int(bbox.y2))
    cv2.rectangle(frame, p1, p2, (0, 0, 255), 3)
    cv2.putText(frame, text, (p1[0], max(0, p1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def draw_zones(frame, zones: List[Zone]):
    for z in zones:
        cv2.polylines(frame, [poly_to_np(z.polygon)], True, (255, 255, 0), 2)
        # Метка рядом с первой точкой
        x, y = z.polygon[0]
        cv2.putText(frame, z.name, (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


def draw_foot_point(frame, pt: Tuple[float, float]):
    cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (255, 0, 255), -1)