from __future__ import annotations

import os
import sys


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from app import _bootstrap_qt_runtime

    _bootstrap_qt_runtime()

    from PyQt5.QtWidgets import QApplication
    from ui.main_window import MainWindow

    app = QApplication([])
    win = MainWindow()
    win.resize(1460, 920)
    try:
        studio = win.studio
        studio.video_name_label.setText("factory_line_demo.mp4")
        studio.video_path = os.path.join(repo_root, "test_data", "demo.mp4")
        studio._set_status("Workspace preview rendered with the SAM3 commercial UI theme.", status_type="success")
        studio._update_workspace_header()
    except Exception:
        pass
    win.show()
    app.processEvents()

    out_path = os.path.join(repo_root, "tmp", "ui_preview.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pixmap = win.grab()
    ok = pixmap.save(out_path)
    print(out_path)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
