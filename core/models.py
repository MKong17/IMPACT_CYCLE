from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import bisect

Delta = Tuple[int, Optional[str], Optional[str]]  # (frame, old_label, new_label)
DeltaRun = Tuple[
    int, int, Optional[str], Optional[str]
]  # (start, end, old_label, new_label)
TxnDelta = Union[Delta, DeltaRun]


@dataclass
class LabelDef:
    name: str
    color_name: str  # key in color dict, or "custom:#RRGGBB"
    id: int  # action id used in export (e.g., 0,1,2,...)


@dataclass
class EntityDef:
    name: str
    id: int


@dataclass
class AnnotationStore:
    """
    Global single-frame, single-label storage.
    Invariant: each frame_id appears at most once (globally).
    """

    # frame_id -> label_name
    frame_to_label: Dict[int, str] = field(default_factory=dict)
    # label_name -> sorted list of frames
    label_to_frames: Dict[str, List[int]] = field(default_factory=dict)
    # transaction bookkeeping
    _txn_active: bool = field(default=False, init=False, repr=False)
    _txn_buffer: List[TxnDelta] = field(default_factory=list, init=False, repr=False)
    _last_deltas: Optional[List[TxnDelta]] = field(default=None, init=False, repr=False)

    def _log(
        self, frame: int, old_label: Optional[str], new_label: Optional[str]
    ) -> None:
        self._append_txn_delta(int(frame), int(frame), old_label, new_label)

    def _append_txn_delta(
        self, start: int, end: int, old_label: Optional[str], new_label: Optional[str]
    ) -> None:
        if not self._txn_active:
            return
        if end < start:
            start, end = end, start
        if self._txn_buffer:
            last = self._txn_buffer[-1]
            if len(last) == 3:
                lf, lold, lnew = last
                if (
                    lold == old_label
                    and lnew == new_label
                    and start <= lf + 1
                    and end >= lf - 1
                ):
                    ns = min(int(lf), start)
                    ne = max(int(lf), end)
                    self._txn_buffer[-1] = (
                        (ns, ne, old_label, new_label)
                        if ns != ne
                        else (ns, old_label, new_label)
                    )
                    return
            else:
                ls, le, lold, lnew = last
                if (
                    lold == old_label
                    and lnew == new_label
                    and start <= int(le) + 1
                    and end >= int(ls) - 1
                ):
                    ns = min(int(ls), start)
                    ne = max(int(le), end)
                    self._txn_buffer[-1] = (
                        (ns, ne, old_label, new_label)
                        if ns != ne
                        else (ns, old_label, new_label)
                    )
                    return
        self._txn_buffer.append(
            (start, end, old_label, new_label)
            if start != end
            else (start, old_label, new_label)
        )

    def is_occupied(self, frame: int) -> bool:
        return frame in self.frame_to_label

    def label_at(self, frame: int) -> Optional[str]:
        return self.frame_to_label.get(frame)

    def add(self, label: str, frame: int) -> bool:
        if frame in self.frame_to_label:
            # allow idempotent add for same label
            if self.frame_to_label[frame] == label:
                return True
            return False
        self.frame_to_label[frame] = label
        lst = self.label_to_frames.setdefault(label, [])
        bisect.insort(lst, frame)
        self._log(frame, None, label)
        return True

    def move(self, old_frame: int, new_frame: int) -> bool:
        if old_frame not in self.frame_to_label:
            return False
        if new_frame in self.frame_to_label:
            return False
        label = self.frame_to_label.pop(old_frame)
        lst = self.label_to_frames.get(label, [])
        try:
            lst.remove(old_frame)
        except ValueError:
            pass
        self.frame_to_label[new_frame] = label
        bisect.insort(self.label_to_frames.setdefault(label, []), new_frame)
        self._log(old_frame, label, None)
        self._log(new_frame, None, label)
        return True

    def remove_at(self, frame: int) -> bool:
        if frame not in self.frame_to_label:
            return False
        label = self.frame_to_label.pop(frame)
        lst = self.label_to_frames.get(label, [])
        try:
            lst.remove(frame)
        except ValueError:
            pass
        self._log(frame, label, None)
        return True

    def remove_range(self, label: str, start: int, end: int) -> int:
        """
        Remove frames in [start, end] for a given label using slice operations.
        This is much faster than calling remove_at() frame-by-frame on long segments.
        """
        if end < start:
            start, end = end, start
        frames = self.label_to_frames.get(label, [])
        if not frames:
            return 0
        i0 = bisect.bisect_left(frames, int(start))
        i1 = bisect.bisect_right(frames, int(end))
        if i0 >= i1:
            return 0
        removed = frames[i0:i1]
        if self._txn_active:
            for rs, re in self.frames_to_runs(removed):
                self._append_txn_delta(int(rs), int(re), label, None)
        for f in removed:
            self.frame_to_label.pop(int(f), None)
        del frames[i0:i1]
        return len(removed)

    def remove_all_of_label(self, label: str) -> None:
        frames = list(self.label_to_frames.get(label, []))
        for f in frames:
            self.frame_to_label.pop(f, None)
        self.label_to_frames.pop(label, None)

    def rename_label(self, old: str, new: str) -> None:
        """Rename a label globally: move its frames to the new label name."""
        if old == new:
            return
        frames = self.label_to_frames.pop(old, [])
        if not frames:
            return
        # ensure target list exists
        lst = self.label_to_frames.setdefault(new, [])
        import bisect

        for f in frames:
            # update the reverse map
            self.frame_to_label[f] = new
            # keep sorted & unique in target list
            if not lst or lst[-1] != f:
                bisect.insort(lst, f)

    def frames_of(self, label: str) -> List[int]:
        return list(self.label_to_frames.get(label, []))

    def nearest_unlabeled(
        self, center: int, radius: int, prefer_forward=True
    ) -> Optional[int]:
        if prefer_forward:
            for d in range(0, radius + 1):
                f = center + d
                if f not in self.frame_to_label:
                    return f
            for d in range(1, radius + 1):
                f = center - d
                if f not in self.frame_to_label:
                    return f
        else:
            for d in range(0, radius + 1):
                f = center - d
                if f not in self.frame_to_label:
                    return f
            for d in range(1, radius + 1):
                f = center + d
                if f not in self.frame_to_label:
                    return f
        return None

    # ===== transaction API =====
    def begin_txn(self) -> None:
        self._txn_active = True
        self._txn_buffer.clear()

    def end_txn(self) -> None:
        if not self._txn_active:
            return
        self._txn_active = False
        # expose the deltas from this transaction (append if previous not yet consumed)
        if self._last_deltas:
            self._last_deltas.extend(self._txn_buffer)
        else:
            self._last_deltas = list(self._txn_buffer)
        self._txn_buffer.clear()

    def cancel_txn(self) -> None:
        self._txn_active = False
        self._txn_buffer.clear()
        self._last_deltas = None

    # MainWindow fetches the last transaction deltas (one-shot)
    def consume_last_deltas(self) -> List[TxnDelta]:
        if not self._last_deltas:
            return []
        ds = self._last_deltas
        self._last_deltas = None
        return ds

    # ===== command application (Undo/Redo) =====
    def apply_deltas(self, deltas: List[TxnDelta]) -> None:
        # apply without re-logging
        was_active = self._txn_active
        self._txn_active = False
        try:
            for d in deltas:
                if len(d) == 3:
                    frame, old_label, new_label = d
                    frame_iter = range(int(frame), int(frame) + 1)
                else:
                    s, e, old_label, new_label = d
                    s = int(s)
                    e = int(e)
                    if old_label is not None and new_label is None:
                        # Remove only frames that still match the expected old label.
                        self.remove_range(old_label, s, e)
                        continue
                    frame_iter = range(min(s, e), max(s, e) + 1)
                for frame in frame_iter:
                    cur = self.frame_to_label.get(frame)
                    # Enforce the expected pre-state to avoid destructive overwrites.
                    if old_label is None:
                        if cur is not None and cur != new_label:
                            continue
                    else:
                        if cur != old_label:
                            if new_label is not None and cur == new_label:
                                continue
                            continue
                    if cur is not None and cur != new_label:
                        self.remove_at(frame)
                    if new_label is not None and self.frame_to_label.get(frame) != new_label:
                        self.add(new_label, frame)
        finally:
            self._txn_active = was_active

    def apply_deltas_reverse(self, deltas: List[TxnDelta]) -> None:
        # reverse apply by swapping (old,new)
        rev: List[TxnDelta] = []
        for d in deltas:
            if len(d) == 3:
                frame, old, new = d
                rev.append((int(frame), new, old))
            else:
                s, e, old, new = d
                rev.append((int(s), int(e), new, old))
        self.apply_deltas(list(reversed(rev)))

    @staticmethod
    def frames_to_runs(frames: List[int]) -> List[Tuple[int, int]]:
        """sorted frames -> contiguous [start,end] (inclusive)"""
        if not frames:
            return []
        runs = []
        s = e = frames[0]
        for f in frames[1:]:
            if f == e + 1:
                e = f
            else:
                runs.append((s, e))
                s = e = f
        runs.append((s, e))
        return runs
