import os
import sys
import cv2
import yaml
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, QPoint
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QComboBox, QCheckBox, QFileDialog
)

from app.config import load_config
from app.pipeline import SafetyPipeline
from pathlib import Path


def get_app_base_dir() -> str:
    # При exe: sys._MEIPASS — это внутренняя папка, но base_dir для записи передаём из launcher.py
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return str(Path(__file__).resolve().parents[1])


def resolve_resource(base_dir: str, relative_path: str) -> str:
    p1 = Path(base_dir) / relative_path
    if p1.exists():
        return str(p1)

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        p2 = Path(sys._MEIPASS) / relative_path
        if p2.exists():
            return str(p2)

    return str(p1)


def cv_to_qimage(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)


def load_zones_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {"zones": []}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"zones": []}


def save_zone_polygon(path: str, zone_id: str, polygon_xy: list[list[int]]):
    data = {
        "zones": [{
            "id": zone_id,
            "name": "Danger_zone",
            "polygon": polygon_xy
        }]
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)


class VideoLabel(QLabel):
    """QLabel, который принимает клики для редактирования зоны."""
    clicked = Signal(int, int, str)  # x, y, "left"/"right"

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(int(event.position().x()), int(event.position().y()), "left")
        elif event.button() == Qt.RightButton:
            self.clicked.emit(int(event.position().x()), int(event.position().y()), "right")
        super().mousePressEvent(event)


class VideoWorker(QThread):
    frame_ready = Signal(np.ndarray)
    status = Signal(str)

    def __init__(self, base_dir: str, source: str):
        super().__init__()
        self.base_dir = base_dir
        self.source = source
        self._stop = False

        # управляющие флаги (меняются из UI)
        self.zone_draw_enabled = True
        self.zone_filter_enabled = False

        # для hot-reload зоны
        self._zones_reload_requested = False

    def stop(self):
        self._stop = True

    def request_reload_zones(self):
        self._zones_reload_requested = True

    def run(self):
        try:
            cfg_path = resolve_resource(self.base_dir, "config.yaml")
            zones_path = resolve_resource(self.base_dir, "zones.yaml")

            # покажем, откуда реально читаем zones
            self.status.emit(f"zones_path = {zones_path}")

            cfg = load_config(cfg_path)
            cfg.source = self.source

            # выключаем OpenCV окно
            cfg.draw_cfg["show_window"] = False

            pipeline = SafetyPipeline(cfg, zones_path, base_dir=self.base_dir)

            # источник (камера или файл)
            src = str(self.source)
            if not src.isdigit():
                src = resolve_resource(self.base_dir, src)

            cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
            if not cap.isOpened():
                self.status.emit("Не удалось открыть источник видео")
                return

            # FPS/задержка считаем один раз
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 1:
                fps = float(cfg.camera_fps_fallback)
            frame_delay = max(1, int(1000 / fps))

            self.status.emit("Запущено")

            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    break

                # задержка только для видеофайла (для камеры не ставим)
                if not str(self.source).isdigit():
                    QThread.msleep(frame_delay)

                # применяем флаги из UI
                pipeline.zone_enabled = bool(self.zone_draw_enabled)
                pipeline.zone_filter_enabled = bool(self.zone_filter_enabled)

                # hot-reload zones.yaml (подхватываем нарисованную зону сразу)
                if self._zones_reload_requested:
                    zones_write_path = str(Path(self.base_dir) / "zones.yaml")
                    from app.zones import load_zones
                    pipeline.zones = load_zones(zones_write_path)
                    self._zones_reload_requested = False

                out = pipeline.process_frame(frame)
                self.frame_ready.emit(out)

            cap.release()
            self.status.emit("Остановлено")

        except Exception as e:
            self.status.emit(f"Ошибка: {e}")


class MainWindow(QMainWindow):
    def __init__(self, base_dir: str | None = None):
        super().__init__()
        self.base_dir = base_dir or get_app_base_dir()
        self.setWindowTitle("Industrial Safety CV")

        # zones.yaml мы сохраняем рядом с exe (base_dir)
        self.zones_path = str(Path(self.base_dir) / "zones.yaml")

        self.edit_mode = False
        # edit_points храним в координатах SCALED-изображения (без полей QLabel)
        self.edit_points: list[tuple[int, int]] = []
        self.zone_id = "zone_1"

        self.video_label = VideoLabel("Нет видео")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(960, 540)
        self.video_label.clicked.connect(self.on_video_click)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["0", "1"])

        self.btn_open_file = QPushButton("Открыть видео")
        self.btn_open_file.clicked.connect(self.open_video_file)

        self.btn_start = QPushButton("Старт")
        self.btn_stop = QPushButton("Стоп")
        self.btn_stop.setEnabled(False)

        self.chk_show_zone = QCheckBox("Показывать danger_zone")
        self.chk_show_zone.setChecked(True)

        self.chk_filter_zone = QCheckBox("Детектить только внутри danger_zone")
        self.chk_filter_zone.setChecked(False)

        self.btn_edit_zone = QPushButton("Настроить зону")
        self.btn_edit_zone.setCheckable(True)

        self.status_label = QLabel("Готово.")

        top = QHBoxLayout()
        top.addWidget(QLabel("Источник:"))
        top.addWidget(self.source_combo)
        top.addWidget(self.btn_open_file)
        top.addWidget(self.btn_start)
        top.addWidget(self.btn_stop)
        top.addWidget(self.chk_show_zone)
        top.addWidget(self.chk_filter_zone)
        top.addWidget(self.btn_edit_zone)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.video_label, 1)
        layout.addWidget(self.status_label)

        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

        self.worker = None

        # для правильного сохранения зоны (offset + размеры)
        self._last_frame_shape = None          # (H, W)
        self._last_pixmap_size = None          # (pw, ph)
        self._pixmap_offset = (0, 0)           # (offx, offy)

        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.chk_show_zone.stateChanged.connect(self.on_ui_flags_changed)
        self.chk_filter_zone.stateChanged.connect(self.on_ui_flags_changed)
        self.btn_edit_zone.toggled.connect(self.toggle_edit_mode)

    def open_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видео",
            "",
            "Видео файлы (*.mp4 *.avi *.mov);;Все файлы (*)"
        )

        if file_path:
            self.source_combo.addItem(file_path)
            self.source_combo.setCurrentText(file_path)

    def start(self):
        if self.worker is not None:
            return

        source = self.source_combo.currentText().strip()
        if not source:
            self.status_label.setText("Выберите источник видео (камера или файл).")
            return

        self.worker = VideoWorker(base_dir=self.base_dir, source=source)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status.connect(self.on_status)

        # применяем флаги сразу
        self.worker.zone_draw_enabled = self.chk_show_zone.isChecked()
        self.worker.zone_filter_enabled = self.chk_filter_zone.isChecked()

        self.worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop(self):
        if self.worker is None:
            return
        self.worker.stop()
        self.worker.wait(2000)
        self.worker = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def closeEvent(self, event):
        self.stop()
        event.accept()

    def on_status(self, text: str):
        self.status_label.setText(text)

    def on_ui_flags_changed(self):
        if self.worker is None:
            return
        self.worker.zone_draw_enabled = self.chk_show_zone.isChecked()
        self.worker.zone_filter_enabled = self.chk_filter_zone.isChecked()

    def toggle_edit_mode(self, enabled: bool):
        self.edit_mode = enabled
        self.edit_points = []
        if enabled:
            self.status_label.setText("Режим зоны: ЛКМ добавить точку, ПКМ удалить, Enter сохранить, Esc отмена")
        else:
            self.status_label.setText("Готово.")

    def on_video_click(self, x, y, btn: str):
        if not self.edit_mode:
            return

        if self._last_pixmap_size is None:
            return

        offx, offy = self._pixmap_offset
        pw, ph = self._last_pixmap_size

        xr = int(x - offx)
        yr = int(y - offy)

        # клики вне реальной картинки игнорируем
        if xr < 0 or yr < 0 or xr >= pw or yr >= ph:
            return

        if btn == "left":
            self.edit_points.append((xr, yr))
        elif btn == "right":
            if self.edit_points:
                self.edit_points.pop()

    def keyPressEvent(self, event):
        if not self.edit_mode:
            return super().keyPressEvent(event)

        if event.key() == Qt.Key_Escape:
            self.edit_points = []
            self.btn_edit_zone.setChecked(False)
            return

        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if len(self.edit_points) < 3:
                self.status_label.setText("Нужно минимум 3 точки для зоны.")
                return

            if self._last_frame_shape is not None and self._last_pixmap_size is not None:
                H, W = self._last_frame_shape
                pw, ph = self._last_pixmap_size

                sx = W / float(pw)
                sy = H / float(ph)

                polygon = [[int(px * sx), int(py * sy)] for (px, py) in self.edit_points]
                save_zone_polygon(self.zones_path, self.zone_id, polygon)

                if self.worker is not None:
                    self.worker.request_reload_zones()

                self.status_label.setText("Зона сохранена в zones.yaml")
                self.btn_edit_zone.setChecked(False)
            else:
                self.status_label.setText("Нет кадра для расчёта масштаба. Запусти видео и попробуй снова.")
            return

        return super().keyPressEvent(event)

    def on_frame(self, frame_bgr: np.ndarray):
        H, W = frame_bgr.shape[:2]
        self._last_frame_shape = (H, W)

        img = cv_to_qimage(frame_bgr)
        pix = QPixmap.fromImage(img)

        scaled = pix.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        pw, ph = scaled.width(), scaled.height()
        self._last_pixmap_size = (pw, ph)

        # поля внутри QLabel из-за KeepAspectRatio
        lw, lh = self.video_label.width(), self.video_label.height()
        self._pixmap_offset = ((lw - pw) // 2, (lh - ph) // 2)

        # рисуем точки поверх scaled pixmap
        if self.edit_mode and self.edit_points:
            from PySide6.QtGui import QPainter, QPen
            painter = QPainter(scaled)
            pen = QPen(Qt.blue, 3)
            painter.setPen(pen)

            for i, (x, y) in enumerate(self.edit_points):
                painter.drawEllipse(QPoint(x, y), 4, 4)
                if i > 0:
                    x0, y0 = self.edit_points[i - 1]
                    painter.drawLine(x0, y0, x, y)

            if len(self.edit_points) >= 3:
                x1, y1 = self.edit_points[-1]
                x0, y0 = self.edit_points[0]
                painter.drawLine(x1, y1, x0, y0)

            painter.end()

        self.video_label.setPixmap(scaled)