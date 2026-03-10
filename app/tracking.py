from dataclasses import dataclass
from typing import List, Dict, Tuple
from .utils import BBox, iou, now_ts


@dataclass
class Track:
    track_id: int
    bbox: BBox
    last_seen: float


class IoUTracker:
    def __init__(self, iou_match: float = 0.3, ttl_sec: float = 1.5):
        self.iou_match = iou_match
        self.ttl_sec = ttl_sec
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, person_boxes: List[BBox]) -> Dict[int, BBox]:
        t = now_ts()

        # удаление истекших треков
        for tid in list(self.tracks.keys()):
            if t - self.tracks[tid].last_seen > self.ttl_sec:
                del self.tracks[tid]

        assigned = set()
        out: Dict[int, BBox] = {}

        # Сопоставление
        for pb in person_boxes:
            best_tid = None
            best_iou = 0.0
            for tid, tr in self.tracks.items():
                if tid in assigned:
                    continue
                v = iou(pb, tr.bbox)
                if v > best_iou:
                    best_iou = v
                    best_tid = tid
            if best_tid is not None and best_iou >= self.iou_match:
                self.tracks[best_tid].bbox = pb
                self.tracks[best_tid].last_seen = t
                assigned.add(best_tid)
                out[best_tid] = pb
            else:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(track_id=tid, bbox=pb, last_seen=t)
                assigned.add(tid)
                out[tid] = pb

        return out