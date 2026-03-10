from typing import List, Optional
import cv2
import sys
from pathlib import Path
import numpy as np

from .config import AppConfig
from .detectors import YoloDetector
from .tracking import IoUTracker
from .zones import load_zones, Zone
from .utils import BBox, now_ts
from .violations import (
    TemporalGate, Cooldown, ViolationEvent,
    match_ppe_to_person, severity_from_type
)
from .storage import Storage, ClipBuffer
from .visualization import draw_bbox, draw_violation, draw_zones, draw_foot_point


def resolve_resource(base_dir: str, relative_path: str) -> str:

    p1 = Path(base_dir) / relative_path
    if p1.exists():
        return str(p1)

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        p2 = Path(sys._MEIPASS) / relative_path
        if p2.exists():
            return str(p2)

    return str(p1)


def reset_gate(gate: TemporalGate, track_id: int) -> None:
    """Полный сброс накопления времени для track_id (чтобы не было хвостов)."""
    gate.start_ts.pop(track_id, None)
    gate.last_true.pop(track_id, None)


def boots_checkable(person: BBox, frame_h: int) -> bool:
    """ Ботинки проверяем если человек реально почти в полный рост. Строго, чтобы не было ложных "NO_BOOTS" при обрезанной фигуре. """
    H = float(frame_h)
    h = person.height()
    w = max(person.width(), 1.0)
    ar = h / w

    if h < 0.72 * H:
        return False
    if ar < 1.8:
        return False
    if person.y1 > 0.18 * H:
        return False
    if person.y2 < 0.92 * H:
        return False
    return True


def gloves_checkable(person: BBox, frame_h: int, frame_w: int) -> bool:
    """
    Перчатки проверяем только если человек достаточно крупный и не обрезан по бокам/снизу.
    """
    H = float(frame_h)
    W = float(frame_w)
    h = person.height()
    w = max(person.width(), 1.0)
    ar = h / w

    if h < 0.55 * H:
        return False
    if ar < 1.4:
        return False

    margin = 18
    if person.x1 <= margin or person.x2 >= (W - margin):
        return False

    if person.y2 < 0.75 * H:
        return False

    return True


def face_visible_on_person(face_detector: cv2.CascadeClassifier, frame, person: BBox) -> bool:
    """
    Быстрый фильтр "человек лицом к камере":
    - вырезаем верхнюю часть bbox человека
    - ищем фронтальное лицо Haar cascade
    Если лицо не найдено -> считаем, что человек не фронтально (спина/профиль/слишком далеко).
    """
    x1, y1, x2, y2 = map(int, [person.x1, person.y1, person.x2, person.y2])
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, frame.shape[1] - 1)
    y2 = min(y2, frame.shape[0] - 1)

    h = max(y2 - y1, 1)
    # берем верхние ~45% bbox человека
    y_face2 = y1 + int(0.45 * h)
    roi = frame[y1:y_face2, x1:x2]
    if roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(20, 20)
    )
    return len(faces) > 0


class SafetyPipeline:
    """
    Hybrid pipeline:
    - Person: COCO model -> детект людей (cls=0)
    - PPE: кастомная модель -> Helmet/Vest/Goggles/Gloves/Boots
    """

    COCO_PERSON_ID = 0

    def __init__(self, cfg: AppConfig, zones_path: str, base_dir: str):
        self.cfg = cfg
        self.base_dir = base_dir

        self.zones: List[Zone] = load_zones(zones_path)

        # paths from config.yaml
        person_model_path = resolve_resource(self.base_dir, cfg.person_model_path)
        ppe_model_path = resolve_resource(self.base_dir, cfg.ppe_model_path)

        # пороги из yaml
        person_conf = float(cfg.person_conf_thres)
        ppe_conf = float(cfg.ppe_conf_thres)

        # PERSON detector — COCO
        self.detector_person = YoloDetector(
            model_path=person_model_path,
            device=cfg.device,
            imgsz=cfg.imgsz,
            conf=person_conf,
            iou_thres=cfg.iou_thres
        )

        # PPE detector
        self.detector_ppe = YoloDetector(
            model_path=ppe_model_path,
            device=cfg.device,
            imgsz=cfg.imgsz,
            conf=ppe_conf,
            iou_thres=cfg.iou_thres
        )

        self.tracker = IoUTracker(iou_match=0.3, ttl_sec=2.0)

        # Временные окна (temporal gates)
        self.gate_no_helmet = TemporalGate()
        self.gate_no_vest = TemporalGate()
        self.gate_no_goggles = TemporalGate()
        self.gate_no_gloves = TemporalGate()
        self.gate_no_boots = TemporalGate()
        self.gate_in_zone = TemporalGate()

        # Гейт "лицо видно" для фильтра NO_GOGGLES
        self.gate_face_visible = TemporalGate()

        self.cooldown = Cooldown()
        self.storage = Storage(cfg.output_dir)
        self.camera_id = "cam_1"

        # Буфер клипов
        pre_s = float(cfg.clip_cfg.get("pre_seconds", 3))
        fps_buf = max(float(cfg.processing_fps), 1.0)
        self.clip_pre_seconds = pre_s
        self.clip_post_seconds = float(cfg.clip_cfg.get("post_seconds", 3))
        self.clip_enabled = bool(cfg.clip_cfg.get("enabled", True))
        self.clip_codec = str(cfg.clip_cfg.get("codec", "mp4v"))
        self.clip_buffer = ClipBuffer(
            max_frames=int((pre_s + self.clip_post_seconds + 2) * fps_buf)
        )

        self.draw_cfg = cfg.draw_cfg

        # Детектор фронтального лица (OpenCV Haar)
        cascade_rel = "data/haarcascade_frontalface_default.xml"
        cascade_path = resolve_resource(self.base_dir, cascade_rel)

        self.face_detector = cv2.CascadeClassifier(cascade_path)
        if self.face_detector.empty():
            raise RuntimeError(
                f"Haar cascade не найден/не загрузился: {cascade_path}\n"
                f"Убедись, что файл лежит в папке рядом с exe: {cascade_rel}"
            )
        # сколько секунд подряд нужно видеть лицо, чтобы включить проверку goggles
        self.face_visible_sec = float(self.cfg.raw.get("face_visible_sec", 0.3))

        # per-class conf пороги (можешь дальше тюнить)
        self.ppe_conf_by_kind = {
            "helmet": 0.12,
            "vest": 0.12,
            "goggles": 0.04,
            "gloves": 0.10,
            "boots": 0.10,
        }

        # --- class ids from config.yaml ---
        self.PPE_HELMET_ID = int(self.cfg.classes.get("helmet", 3))
        self.PPE_VEST_ID = int(self.cfg.classes.get("vest", 5))
        self.PPE_GOGGLES_ID = int(self.cfg.classes.get("goggles", 2))
        self.PPE_GLOVES_ID = int(self.cfg.classes.get("gloves", 1))
        self.PPE_BOOTS_ID = int(self.cfg.classes.get("boots", 0))

        # флаги из UI (ставятся из main_window.py)
        self.zone_enabled = True
        self.zone_filter_enabled = False

    def _apply_zone_mask(self, frame):
        """
        Оставляет видимой только область внутри зоны (остальное чёрное).
        Работает только если zones не пустой.
        """
        if not self.zones:
            return frame
        z = self.zones[0]
        poly = np.array(z.polygon, dtype=np.int32)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        return cv2.bitwise_and(frame, frame, mask=mask)

    def process_frame(self, frame):
        """
        Обрабатывает один кадр и возвращает кадр с отрисовкой.
        """
        vis_frame = frame.copy()
        H, W = vis_frame.shape[:2]

        # Кладём кадр в буфер клипа (для pre_seconds)
        try:
            self.clip_buffer.push(now_ts(), frame.copy())
        except Exception:
            pass

        # Если включён фильтр по зоне — детектим только внутри маски
        use_frame = frame
        if getattr(self, "zone_filter_enabled", False):
            use_frame = self._apply_zone_mask(frame)

        # 1) COCO модель (люди)
        boxes_person = self.detector_person.detect(use_frame)
        persons = [b for b in boxes_person if int(b.cls) == self.COCO_PERSON_ID]
        tracks = self.tracker.update(persons)

        # 2) PPE пользовательская модель
        boxes_ppe = self.detector_ppe.detect(use_frame)

        helmets = [
            b for b in boxes_ppe
            if int(b.cls) == self.PPE_HELMET_ID and float(b.score) >= self.ppe_conf_by_kind["helmet"]
        ]
        vests = [
            b for b in boxes_ppe
            if int(b.cls) == self.PPE_VEST_ID and float(b.score) >= self.ppe_conf_by_kind["vest"]
        ]
        goggles = [
            b for b in boxes_ppe
            if int(b.cls) == self.PPE_GOGGLES_ID and float(b.score) >= self.ppe_conf_by_kind["goggles"]
        ]
        gloves = [
            b for b in boxes_ppe
            if int(b.cls) == self.PPE_GLOVES_ID and float(b.score) >= self.ppe_conf_by_kind["gloves"]
        ]
        boots = [
            b for b in boxes_ppe
            if int(b.cls) == self.PPE_BOOTS_ID and float(b.score) >= self.ppe_conf_by_kind["boots"]
        ]

        # Рисуем зону (если включено из UI)
        if self.draw_cfg.get("enabled", True) and getattr(self, "zone_enabled", True):
            draw_zones(vis_frame, self.zones)

        # Оценка каждого трека
        for track_id, pb in tracks.items():
            # что можно проверять на этом кадре
            can_boots = boots_checkable(pb, H)
            can_gloves = gloves_checkable(pb, H, W)

            # сбрасываем накопления, если проверка невозможна
            if not can_boots:
                reset_gate(self.gate_no_boots, track_id)
            if not can_gloves:
                reset_gate(self.gate_no_gloves, track_id)

            # goggles проверяем только когда человек лицом
            face_ok = face_visible_on_person(self.face_detector, frame, pb)
            dur_face = self.gate_face_visible.update(track_id, face_ok)
            can_check_goggles = dur_face >= self.face_visible_sec

            # если лицо не видно — сбрасываем накопление NO_GOGGLES
            if not can_check_goggles:
                reset_gate(self.gate_no_goggles, track_id)

            # PPE ассоциации
            h_match = match_ppe_to_person(pb, helmets, self.cfg.ppe_rules["helmet"], "helmet")
            v_match = match_ppe_to_person(pb, vests, self.cfg.ppe_rules["vest"], "vest")
            g_match = match_ppe_to_person(pb, goggles, self.cfg.ppe_rules.get("goggles", {}), "goggles")
            gl_match = match_ppe_to_person(pb, gloves, self.cfg.ppe_rules.get("gloves", {}), "gloves")
            b_match = match_ppe_to_person(pb, boots, self.cfg.ppe_rules.get("boots", {}), "boots")

            no_helmet = (h_match is None)
            no_vest = (v_match is None)

            # goggles проверяем только при "face gate"
            no_goggles = (g_match is None) if can_check_goggles else False

            no_gloves = (gl_match is None) if can_gloves else False
            no_boots = (b_match is None) if can_boots else False

            dur_h = self.gate_no_helmet.update(track_id, no_helmet)
            dur_v = self.gate_no_vest.update(track_id, no_vest)
            dur_g = self.gate_no_goggles.update(track_id, no_goggles)
            dur_gl = self.gate_no_gloves.update(track_id, no_gloves)
            dur_b = self.gate_no_boots.update(track_id, no_boots)

            # Проверка зоны по foot_point
            foot = pb.foot_point()
            in_zone_id = None
            for z in self.zones:
                if z.contains(foot):
                    in_zone_id = z.zone_id
                    break
            in_zone = in_zone_id is not None
            dur_z = self.gate_in_zone.update(track_id, in_zone)

            # если включён фильтр зоны, то события/оверлеи считаем только когда foot_point внутри зоны.
            if getattr(self, "zone_filter_enabled", False):
                if not in_zone:
                    reset_gate(self.gate_no_helmet, track_id)
                    reset_gate(self.gate_no_vest, track_id)
                    reset_gate(self.gate_no_goggles, track_id)
                    reset_gate(self.gate_no_gloves, track_id)
                    reset_gate(self.gate_no_boots, track_id)
                    reset_gate(self.gate_in_zone, track_id)
                    continue

            # draw
            if self.draw_cfg.get("enabled", True):
                draw_bbox(vis_frame, pb, f"person#{track_id}")
                draw_foot_point(vis_frame, foot)

                if h_match is not None:
                    draw_bbox(vis_frame, h_match, "helmet*")
                if v_match is not None:
                    draw_bbox(vis_frame, v_match, "vest*")
                if can_check_goggles and g_match is not None:
                    draw_bbox(vis_frame, g_match, "goggles*")
                if can_gloves and gl_match is not None:
                    draw_bbox(vis_frame, gl_match, "gloves*")
                if can_boots and b_match is not None:
                    draw_bbox(vis_frame, b_match, "boots*")

            # events (логирование/снапшоты/клипы)
            self._maybe_fire(vis_frame, pb, h_match, track_id, "NO_HELMET", dur_h, in_zone_id)
            self._maybe_fire(vis_frame, pb, v_match, track_id, "NO_VEST", dur_v, in_zone_id)

            if can_check_goggles:
                self._maybe_fire(vis_frame, pb, g_match, track_id, "NO_GOGGLES", dur_g, in_zone_id)

            if can_gloves:
                self._maybe_fire(vis_frame, pb, gl_match, track_id, "NO_GLOVES", dur_gl, in_zone_id)
            if can_boots:
                self._maybe_fire(vis_frame, pb, b_match, track_id, "NO_BOOTS", dur_b, in_zone_id)

            self._maybe_fire(vis_frame, pb, None, track_id, "IN_DANGER_ZONE", dur_z, in_zone_id)

            # overlays
            if self.draw_cfg.get("enabled", True):
                if dur_h >= float(self.cfg.violations["no_helmet"]["threshold_sec"]):
                    draw_violation(vis_frame, pb, "NO_HELMET")
                if dur_v >= float(self.cfg.violations["no_vest"]["threshold_sec"]):
                    draw_violation(vis_frame, pb, "NO_VEST")

                if can_check_goggles and dur_g >= float(self.cfg.violations["no_goggles"]["threshold_sec"]):
                    draw_violation(vis_frame, pb, "NO_GOGGLES")

                if can_gloves and dur_gl >= float(self.cfg.violations["no_gloves"]["threshold_sec"]):
                    draw_violation(vis_frame, pb, "NO_GLOVES")
                if can_boots and dur_b >= float(self.cfg.violations["no_boots"]["threshold_sec"]):
                    draw_violation(vis_frame, pb, "NO_BOOTS")

                if dur_z >= float(self.cfg.violations["in_danger_zone"]["threshold_sec"]):
                    draw_violation(vis_frame, pb, "IN_ZONE")

        return vis_frame

    def _parse_source(self):
        s = self.cfg.source
        if isinstance(s, str) and s.isdigit():
            return int(s)
        return s

    def run(self):
        """Консольный режим (OpenCV окно). Использует process_frame()."""
        cap = cv2.VideoCapture(self._parse_source())
        if not cap.isOpened():
            raise RuntimeError("Не удалось открыть источник видео")

        processing_fps = float(self.cfg.processing_fps)
        process_period = 1.0 / max(processing_fps, 1.0)
        last_process_time = 0.0

        show = bool(self.draw_cfg.get("show_window", True))
        window_name = str(self.draw_cfg.get("window_name", "Industrial Safety CV"))

        if show:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        last_vis_frame = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t = now_ts()
            do_process = (t - last_process_time) >= process_period

            if do_process:
                last_process_time = t
                vis_frame = self.process_frame(frame)
                last_vis_frame = vis_frame
            else:
                vis_frame = last_vis_frame if last_vis_frame is not None else frame

            if show:
                cv2.imshow(window_name, vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

        cap.release()
        if show:
            cv2.destroyAllWindows()

    def _maybe_fire(
        self,
        frame,
        person_bbox: BBox,
        ppe_bbox: Optional[BBox],
        track_id: int,
        vtype: str,
        duration_sec: float,
        zone_id: Optional[str]
    ):
        vtype_to_cfgkey = {
            "NO_HELMET": "no_helmet",
            "NO_VEST": "no_vest",
            "NO_GOGGLES": "no_goggles",
            "NO_GLOVES": "no_gloves",
            "NO_BOOTS": "no_boots",
            "IN_DANGER_ZONE": "in_danger_zone",
        }

        cfg_key = vtype_to_cfgkey.get(vtype)
        if cfg_key is None:
            return

        vcfg = self.cfg.violations[cfg_key]
        threshold = float(vcfg["threshold_sec"])
        cooldown_sec = float(vcfg["cooldown_sec"])

        if duration_sec < threshold:
            return

        if vtype == "IN_DANGER_ZONE" and zone_id is None:
            return

        if not self.cooldown.can_fire(track_id, vtype, cooldown_sec):
            return

        ts = now_ts()
        event_id = f"{int(ts)}_{self.camera_id}_{track_id}_{vtype}"

        snapshot_path = self.storage.save_snapshot(frame, event_id)

        clip_path = None
        if self.clip_enabled:
            pre_frames = self.clip_buffer.get_last_seconds(self.clip_pre_seconds)
            clip_path = self.storage.save_clip(
                frames=pre_frames,
                fps=float(self.cfg.processing_fps),
                event_id=event_id,
                codec=self.clip_codec
            )

        ev = ViolationEvent(
            event_id=event_id,
            timestamp=ts,
            camera_id=self.camera_id,
            track_id=track_id,
            violation_type=vtype,
            severity=severity_from_type(vtype),
            confidence=float(person_bbox.score),
            duration_sec=float(duration_sec),
            zone_id=zone_id,
            bbox_person=(person_bbox.x1, person_bbox.y1, person_bbox.x2, person_bbox.y2),
            bbox_ppe=(ppe_bbox.x1, ppe_bbox.y1, ppe_bbox.x2, ppe_bbox.y2) if ppe_bbox else None,
            snapshot_path=snapshot_path,
            clip_path=clip_path
        )
        self.storage.log_event(ev)