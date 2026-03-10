from dataclasses import dataclass
from typing import Dict, Any
import yaml
from pathlib import Path


@dataclass
class AppConfig:
    raw: Dict[str, Any]

    # source / models
    @property
    def source(self) -> str:
        return str(self.raw.get("source", "0"))

    @source.setter
    def source(self, value: str) -> None:
        self.raw["source"] = value

    # Старое поле
    @property
    def model_path(self) -> str:
        return str(self.raw.get("model_path", ""))

    # Новые поля
    @property
    def person_model_path(self) -> str:
        # если не задано — fallback на yolov8n.pt
        return str(self.raw.get("person_model_path", "yolov8n.pt"))

    @property
    def ppe_model_path(self) -> str:
        # если не задано — fallback на weights/best.pt
        return str(self.raw.get("ppe_model_path", "weights/best.pt"))

    # device / image size
    @property
    def device(self) -> str:
        return str(self.raw.get("device", "cpu"))

    @property
    def imgsz(self) -> int:
        return int(self.raw.get("imgsz", 640))

    # fps
    @property
    def processing_fps(self) -> float:
        return float(self.raw.get("processing_fps", 10))

    @property
    def camera_fps_fallback(self) -> float:
        return float(self.raw.get("camera_fps_fallback", 25))

    # Пороги
    @property
    def conf_thres(self) -> float:
        """
        Для совместимости:
        - если в yaml есть person_conf_thres -> используем его
        - иначе conf_thres
        """
        if "person_conf_thres" in self.raw:
            return float(self.raw.get("person_conf_thres", 0.25))
        return float(self.raw.get("conf_thres", 0.25))

    @property
    def person_conf_thres(self) -> float:
        return float(self.raw.get("person_conf_thres", self.conf_thres))

    @property
    def ppe_conf_thres(self) -> float:
        return float(self.raw.get("ppe_conf_thres", 0.05))

    @property
    def iou_thres(self) -> float:
        return float(self.raw.get("iou_thres", 0.5))

    # mapping / rules
    @property
    def classes(self) -> Dict[str, int]:
        return dict(self.raw.get("classes", {}))

    @property
    def ppe_rules(self) -> Dict[str, Any]:
        return dict(self.raw.get("ppe_rules", {}))

    @property
    def violations(self) -> Dict[str, Any]:
        return dict(self.raw.get("violations", {}))

    # clip/output/draw
    @property
    def clip_cfg(self) -> Dict[str, Any]:
        return dict(self.raw.get("clip", {}))

    @property
    def output_dir(self) -> str:
        return str(self.raw.get("output_dir", "output"))

    @property
    def draw_cfg(self) -> Dict[str, Any]:
        return dict(self.raw.get("draw", {}))


def load_config(path: str) -> AppConfig:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return AppConfig(raw=data)
