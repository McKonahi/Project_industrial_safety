from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QPen
import yaml
from pathlib import Path


class ZoneEditor(QDialog):
    """ Редактор зоны:
    -клики мышкой добавляют точки полигона
    -Save записывает в zones.yaml
    """

    def __init__(self, zones_path: str, zone_id: str = "zone_1", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zone Editor")
        self.resize(800, 600)

        self.zones_path = Path(zones_path)
        self.zone_id = zone_id

        self.points = []  # list[QPoint]

        self.lbl = QLabel("Кликните мышкой по окну: добавьте точки полигона.\n"
                          "ПКМ — удалить последнюю точку.")
        self.btn_save = QPushButton("Save zone")
        self.btn_clear = QPushButton("Clear")

        self.btn_save.clicked.connect(self.save_zone)
        self.btn_clear.clicked.connect(self.clear_zone)

        layout = QVBoxLayout()
        layout.addWidget(self.lbl)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_clear)
        self.setLayout(layout)

    def clear_zone(self):
        self.points = []
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.points.append(event.pos())
            self.update()
        elif event.button() == Qt.RightButton:
            if self.points:
                self.points.pop()
                self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.points:
            return

        painter = QPainter(self)
        pen = QPen(Qt.blue, 2, Qt.SolidLine)
        painter.setPen(pen)

        # точки
        for p in self.points:
            painter.drawEllipse(p, 4, 4)

        # линии
        for i in range(len(self.points) - 1):
            painter.drawLine(self.points[i], self.points[i + 1])

        # замыкаем полигон
        if len(self.points) >= 3:
            painter.drawLine(self.points[-1], self.points[0])

    def save_zone(self):
        if len(self.points) < 3:
            self.lbl.setText("Нужно минимум 3 точки, чтобы сохранить полигон.")
            return

        polygon = [[int(p.x()), int(p.y())] for p in self.points]

        # если файла нет — создаём структуру
        if self.zones_path.exists():
            data = yaml.safe_load(self.zones_path.read_text(encoding="utf-8")) or {}
        else:
            data = {}

        if "zones" not in data:
            data["zones"] = []

        # обновляем или добавляем zone_id
        updated = False
        for z in data["zones"]:
            if z.get("id") == self.zone_id:
                z["polygon"] = polygon
                updated = True
                break

        if not updated:
            data["zones"].append({
                "id": self.zone_id,
                "name": "Опасная зона",
                "polygon": polygon
            })

        self.zones_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        self.accept()