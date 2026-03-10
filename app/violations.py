from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from .utils import BBox, iou, now_ts
from .zones import Zone


@dataclass
class ViolationEvent:
    event_id: str
    timestamp: float
    camera_id: str
    track_id: int
    violation_type: str
    severity: str
    confidence: float
    duration_sec: float
    zone_id: Optional[str]
    bbox_person: Tuple[float, float, float, float]
    bbox_ppe: Optional[Tuple[float, float, float, float]]
    snapshot_path: Optional[str]
    clip_path: Optional[str]


class TemporalGate:
    """Tracks how long a condition has been true per track_id."""
    def __init__(self):
        self.start_ts: Dict[int, float] = {}
        self.last_true: Dict[int, float] = {}

    def update(self, track_id: int, is_true: bool) -> float:
        t = now_ts()
        if is_true:
            if track_id not in self.start_ts:
                self.start_ts[track_id] = t
            self.last_true[track_id] = t
            return t - self.start_ts[track_id]
        else:
            # reset
            self.start_ts.pop(track_id, None)
            self.last_true.pop(track_id, None)
            return 0.0


class Cooldown:
    def __init__(self):
        self.last_fire: Dict[Tuple[int, str], float] = {}

    def can_fire(self, track_id: int, vtype: str, cooldown_sec: float) -> bool:
        t = now_ts()
        key = (track_id, vtype)
        last = self.last_fire.get(key, 0.0)
        if t - last >= cooldown_sec:
            self.last_fire[key] = t
            return True
        return False


def match_ppe_to_person(person: BBox, ppe_boxes: List[BBox], rule: Dict[str, Any], kind: str) -> Optional[BBox]:
    """Возвращае тнаилучшее соответствие PPE bbox для person bbox на основе геометрических ограничений."""
    min_iou = float(rule.get("min_iou", 0.01))

    px1, py1, px2, py2 = person.x1, person.y1, person.x2, person.y2
    p_w = person.width()
    p_h = person.height()

    best = None
    best_score = -1.0

    for bb in ppe_boxes:

        if iou(person, bb) < min_iou:
            continue

        cx, cy = bb.center()

        # --- helmet: верхняя часть ---
        if kind == "helmet":
            beta = float(rule.get("beta", 0.35))
            if cy >= py1 + beta * p_h:
                continue

        # --- goggles: верх головы/лицо (чуть ниже helmet) ---
        elif kind == "goggles":
            beta_top = float(rule.get("beta_top", 0.15))
            beta_bottom = float(rule.get("beta_bottom", 0.45))
            if not (py1 + beta_top * p_h <= cy <= py1 + beta_bottom * p_h):
                continue

        # --- vest: середина тела ---
        elif kind == "vest":
            beta_top = float(rule.get("beta_top", 0.25))
            beta_bottom = float(rule.get("beta_bottom", 0.75))
            if not (py1 + beta_top * p_h <= cy <= py1 + beta_bottom * p_h):
                continue

        # --- gloves: ближе к рукам ---
        elif kind == "gloves":
            beta_top = float(rule.get("beta_top", 0.35))
            beta_bottom = float(rule.get("beta_bottom", 0.85))
            if not (py1 + beta_top * p_h <= cy <= py1 + beta_bottom * p_h):
                continue

            # опционально: по X ближе к краям человека
            edge_x = float(rule.get("edge_x", 0.25))  # 0.25 = ближе к левому/правому краю
            left_edge = px1 + edge_x * p_w
            right_edge = px2 - edge_x * p_w
            if not (cx <= left_edge or cx >= right_edge):
                continue

        # --- boots: низ тела ---
        elif kind == "boots":
            beta = float(rule.get("beta", 0.90))
            if cy < py1 + beta * p_h:
                continue

        # Если неизвестный kind — ропускаем
        else:
            continue

        # выбираем по уверенности модели
        score = float(bb.score)
        if score > best_score:
            best_score = score
            best = bb

    return best

def severity_from_type(vtype: str) -> str:
    if vtype == "IN_DANGER_ZONE":
        return "HIGH"
    return "MEDIUM"