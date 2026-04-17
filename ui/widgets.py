from PyQt5.QtCore import Qt, QSize, QRect
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush
from PyQt5.QtWidgets import QAbstractButton, QListWidget, QStyleOptionViewItem, QStyle


class ToggleSwitch(QAbstractButton):
    """Compact on/off switch with a sliding thumb."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self._margin = 2
        self._track_off = QColor("#d0d5dd")
        self._track_on = QColor("#12b76a")
        self._track_border = QColor("#98a2b3")
        self._thumb = QColor("#ffffff")
        self.setFixedSize(42, 22)

    def sizeHint(self) -> QSize:
        return QSize(42, 22)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        track = QRect(
            self._margin, self._margin, w - 2 * self._margin, h - 2 * self._margin
        )
        radius = track.height() / 2
        p.setPen(QPen(self._track_border))
        p.setBrush(QBrush(self._track_on if self.isChecked() else self._track_off))
        p.drawRoundedRect(track, radius, radius)

        thumb_d = track.height() - 4
        thumb_y = track.top() + 2
        if self.isChecked():
            thumb_x = track.right() - thumb_d - 1
        else:
            thumb_x = track.left() + 1
        thumb_rect = QRect(int(thumb_x), int(thumb_y), int(thumb_d), int(thumb_d))
        p.setPen(QPen(QColor("#ffffff")))
        p.setBrush(QBrush(self._thumb))
        p.drawEllipse(thumb_rect)


class ClickToggleList(QListWidget):
    """A checklist that toggles by clicking the item body, not only the checkbox."""

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if item is None:
            return super().mousePressEvent(event)
        option = QStyleOptionViewItem()
        option.initFrom(self)
        option.rect = self.visualItemRect(item)
        indicator = self.style().subElementRect(
            QStyle.SE_ItemViewItemCheckIndicator, option, self
        )
        if indicator.contains(event.pos()):
            return super().mousePressEvent(event)
        item.setCheckState(
            Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
        )
