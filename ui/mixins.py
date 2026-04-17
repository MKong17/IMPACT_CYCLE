class FrameControlMixin:
    """Small helper to update frame-related controls with signals blocked."""

    @staticmethod
    def _set_value_blocked(widget, value: int):
        widget.blockSignals(True)
        widget.setValue(int(value))
        widget.blockSignals(False)

    def _set_frame_controls(self, frame: int):
        self._set_value_blocked(self.spin_jump, frame)
        self._set_value_blocked(self.slider, frame)
