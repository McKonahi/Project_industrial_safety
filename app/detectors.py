from typing import List, Dict
from ultralytics import YOLO
from .utils import BBox


class YoloDetector:
    def __init__(self, model_path: str, device: str, imgsz: int, conf: float, iou_thres: float):
        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou_thres = iou_thres

    def detect(self, frame) -> List[BBox]:
        # Ultralytics обрабатывает NMS внутренне; iou_thres для NMS в predict
        res = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou_thres,
            device=self.device,
            verbose=False
        )
        boxes: List[BBox] = []
        if not res:
            return boxes
        r0 = res[0]
        if r0.boxes is None:
            return boxes
        for b in r0.boxes:
            xyxy = b.xyxy[0].tolist()
            score = float(b.conf[0].item()) if b.conf is not None else 1.0
            cls = int(b.cls[0].item()) if b.cls is not None else -1
            boxes.append(BBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3], score=score, cls=cls))
        return boxes