from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QSizePolicy, QVBoxLayout, QWidget

from ui.video_task_studio import VideoTaskStudio


class MainWindow(QWidget):
    """Task-optimized host for Video Scene Graph, VQA, and captioning."""

    def __init__(self, logger=None):
        super().__init__()
        self._app_title = "IMPACT VQA"
        self.setWindowTitle(self._app_title)
        self.setGeometry(100, 80, 1100, 720)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        try:
            screen = QApplication.primaryScreen()
            geom = screen.availableGeometry() if screen else None
            if geom:
                min_w = min(800, max(640, geom.width() - 80))
                min_h = min(600, max(480, geom.height() - 80))
                self.setMinimumSize(min_w, min_h)
                target_w = min(1360, geom.width() - 60)
                target_h = min(840, geom.height() - 60)
                self.resize(max(min_w, target_w), max(min_h, target_h))
            else:
                self.setMinimumSize(800, 600)
        except Exception:
            self.setMinimumSize(800, 600)
        try:
            self.setWindowFlag(Qt.MSWindowsFixedSizeDialogHint, False)
        except Exception:
            pass

        root = QVBoxLayout(self)
        self.studio = VideoTaskStudio(self)
        root.addWidget(self.studio)
