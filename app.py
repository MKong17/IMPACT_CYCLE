import argparse
import os
import sys


def _looks_like_cv2_qt_path(path: str) -> bool:
    raw = str(path or "").replace("\\", "/").lower()
    return "/site-packages/cv2/qt/plugins" in raw or raw.endswith("/cv2/qt/plugins")


def _first_existing(paths: list[str]) -> str:
    for path in paths:
        p = os.path.abspath(os.path.expanduser(str(path or "").strip()))
        if p and os.path.isdir(p):
            return p
    return ""


def _bootstrap_qt_runtime() -> None:
    """
    Force Qt to use the PyQt5-bundled runtime/plugins.
    This avoids PATH/plugin conflicts in mixed environments (e.g. Anaconda + other Qt installs).
    """
    try:
        import PyQt5  # delay QtWidgets import until plugin paths are normalized
        from PyQt5.QtCore import QLibraryInfo
    except Exception:
        return

    pyqt_root = os.path.dirname(PyQt5.__file__)
    qt_root = os.path.join(pyqt_root, "Qt5")
    pyqt_plugin_root = os.path.join(qt_root, "plugins")
    pyqt_platform_root = os.path.join(pyqt_plugin_root, "platforms")
    pyqt_bin = os.path.join(qt_root, "bin")

    conda_root = sys.prefix
    if os.name == "nt":
        conda_plugin_candidates = [os.path.join(conda_root, "Library", "plugins")]
        conda_bin_candidates = [os.path.join(conda_root, "Library", "bin")]
    else:
        conda_plugin_candidates = [
            os.path.join(conda_root, "lib", "qt", "plugins"),
            os.path.join(conda_root, "lib", "qt5", "plugins"),
            os.path.join(conda_root, "plugins"),
        ]
        conda_bin_candidates = [os.path.join(conda_root, "bin")]

    qli_plugin_root = ""
    try:
        qli_plugin_root = str(QLibraryInfo.location(QLibraryInfo.PluginsPath) or "").strip()
    except Exception:
        qli_plugin_root = ""

    plugin_candidates = [qli_plugin_root, *conda_plugin_candidates, pyqt_plugin_root]
    plugin_root = _first_existing(plugin_candidates)
    platform_root = _first_existing([os.path.join(plugin_root, "platforms"), pyqt_platform_root])
    qt_bin = _first_existing([*conda_bin_candidates, pyqt_bin])

    # Clear and rebuild Qt plugin paths to avoid cv2 Qt plugin conflicts.
    os.environ.pop("QT_PLUGIN_PATH", None)
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    if plugin_root and not _looks_like_cv2_qt_path(plugin_root):
        os.environ["QT_PLUGIN_PATH"] = plugin_root
    if platform_root and not _looks_like_cv2_qt_path(platform_root):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = platform_root

    if os.path.isdir(qt_bin):
        cur_path = os.environ.get("PATH", "")
        parts = [p.strip() for p in cur_path.split(os.pathsep) if p.strip()]
        norm_parts = {os.path.normcase(os.path.normpath(p)) for p in parts}
        norm_qt_bin = os.path.normcase(os.path.normpath(qt_bin))
        if norm_qt_bin not in norm_parts:
            os.environ["PATH"] = qt_bin + os.pathsep + cur_path
        # Python 3.8+ on Windows: ensure dependent DLL lookup includes Qt bin.
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(qt_bin)
            except Exception:
                pass


_bootstrap_qt_runtime()

from utils.feature_env import load_feature_env_defaults  # noqa: E402
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
load_feature_env_defaults(repo_root=_REPO_ROOT)
# Some EAST/torch kernels need this before torch-dependent modules load.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from PyQt5.QtGui import QIcon  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402
from utils.op_logger import OperationLogger  # noqa: E402


def _set_windows_app_id(app_id: str) -> None:
    if os.name != "nt" or not app_id:
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass


def _resolve_app_icon() -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(root, "icon.ico"),
        os.path.join(root, "icon.png"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oplog",
        action="store_true",
        help="Enable operation logging (writes alongside annotation file).",
    )
    args, _ = parser.parse_known_args()

    _set_windows_app_id("cvhci.video.annotation.impact_vqa")
    # Re-apply Qt plugin paths after importing app modules; cv2 can overwrite them.
    _bootstrap_qt_runtime()
    app = QApplication(sys.argv)
    icon_path = _resolve_app_icon()
    if icon_path:
        icon = QIcon(icon_path)
        app.setWindowIcon(icon)
    logger = OperationLogger(enabled=bool(args.oplog))
    w = MainWindow(logger=logger)
    if icon_path:
        w.setWindowIcon(QIcon(icon_path))
    w.show()
    sys.exit(app.exec_())


# Subtitle conversion feature is not implemented
# Bbox Genera1tor feature is not implemented
# Segmantaion Assistant feature is not implemented
# Interactive Segmentation feature is not implemented
