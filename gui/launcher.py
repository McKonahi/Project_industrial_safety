import os
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow

def get_base_dir() -> str:
    if getattr(sys, "frozen", False):
        # exe
        return str(Path(sys.executable).resolve().parent)
    # обычный запуск: gui/launcher.py - корень на 1 уровень выше gui
    return str(Path(__file__).resolve().parents[1])

def main():
    app = QApplication(sys.argv)
    base_dir = get_base_dir()

    window = MainWindow(base_dir=base_dir)
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()