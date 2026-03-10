from dataclasses import asdict
from typing import Deque, Optional, Tuple
from collections import deque
from pathlib import Path
import csv
import json
import cv2
import time

from .violations import ViolationEvent


class ClipBuffer:
    def __init__(self, max_frames: int):
        self.frames: Deque[Tuple[float, any]] = deque(maxlen=max_frames)  # (ts, frame)

    def push(self, ts: float, frame):
        self.frames.append((ts, frame))

    def get_last_seconds(self, seconds: float) -> list:
        if not self.frames:
            return []
        end_ts = self.frames[-1][0]
        start_ts = end_ts - seconds
        return [f for (t, f) in self.frames if t >= start_ts]


class Storage:
    def __init__(self, output_dir: str):
        self.root = Path(output_dir)
        self.snap_dir = self.root / "snapshots"
        self.clip_dir = self.root / "clips"
        self.log_dir = self.root / "logs"
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / "events.csv"
        self.jsonl_path = self.log_dir / "events.jsonl"

        self._init_csv()

    def _init_csv(self):
        if self.csv_path.exists():
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "event_id","timestamp","camera_id","track_id","violation_type",
                "severity","confidence","duration_sec","zone_id",
                "bbox_person","bbox_ppe","snapshot_path","clip_path"
            ])

    def save_snapshot(self, frame, event_id: str) -> str:
        path = self.snap_dir / f"{event_id}.jpg"
        cv2.imwrite(str(path), frame)
        return str(path)

    def save_clip(self, frames: list, fps: float, event_id: str, codec: str = "mp4v") -> Optional[str]:
        if not frames:
            return None
        h, w = frames[0].shape[:2]
        path = self.clip_dir / f"{event_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*codec)
        vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        for fr in frames:
            vw.write(fr)
        vw.release()
        return str(path)

    def log_event(self, ev: ViolationEvent):
        # CSV
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                ev.event_id, ev.timestamp, ev.camera_id, ev.track_id, ev.violation_type,
                ev.severity, ev.confidence, ev.duration_sec, ev.zone_id,
                str(ev.bbox_person), str(ev.bbox_ppe),
                ev.snapshot_path, ev.clip_path
            ])
        # JSONL
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")