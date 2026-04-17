from PyQt5.QtGui import QColor, QPixmap, QPainter, QIcon
from PyQt5.QtCore import Qt, QSize

# Built-in palette
PRESET_COLORS = {
    "Red": QColor(220, 53, 69),
    "Orange": QColor(255, 159, 64),
    "Yellow": QColor(255, 205, 86),
    "Green": QColor(40, 167, 69),
    "Teal": QColor(32, 201, 151),
    "Cyan": QColor(23, 162, 184),
    "Blue": QColor(0, 123, 255),
    "Indigo": QColor(102, 16, 242),
    "Purple": QColor(111, 66, 193),
    "Pink": QColor(232, 62, 140),
    "Gray": QColor(108, 117, 125),
}

# timeline render & interaction
MARKER_WIDTH_PX = 10
ROW_HEIGHT = 32
EDGE_TOLERANCE_PX = 6

# snapping & exclusivity
SNAP_RADIUS_FRAMES = 10
# playhead (red current-frame marker) snap radius, kept smaller than generic snap
CURRENT_FRAME_SNAP_RADIUS_FRAMES = 6
PREFER_FORWARD = True
EDGE_SNAP_FRAMES = 5

# default timeline span
DEFAULT_VIEW_SPAN = 600
MIN_VIEW_SPAN = 100

# Special label names
EXTRA_LABEL_NAME = "Interaction"
EXTRA_ALIASES = {EXTRA_LABEL_NAME, "Extra", "extra"}


def is_extra_label(name: str) -> bool:
    if name is None:
        return False
    txt = str(name)
    return txt in EXTRA_ALIASES or txt.lower() == "extra"


def color_from_key(key: str) -> QColor:
    """Accept preset key or 'custom:#RRGGBB'."""
    if key.startswith("custom:"):
        hexcode = key.split(":", 1)[1].strip()
        return QColor(hexcode)
    return PRESET_COLORS.get(key, QColor(0, 123, 255))


def make_color_icon(color: QColor, size: QSize = QSize(14, 14)) -> QIcon:
    pm = QPixmap(size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setBrush(color)
    p.setPen(Qt.gray)
    # p.drawRect(0, 0, size.width()-1, size.height()-1)
    p.drawEllipse(0, 0, size.width() - 1, size.height() - 1)
    p.end()
    return QIcon(pm)
