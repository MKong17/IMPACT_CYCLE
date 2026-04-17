import csv
from datetime import datetime
from typing import Dict, Any, List
import os


class OperationLogger:
    """
    Lightweight operation logger.
    - enabled: toggle logging.
    - call log(event, **fields) to append an entry.
    - call flush(path) to write CSV to disk.
    """

    def __init__(self, enabled: bool = False, max_rows: int = 20000):
        self.enabled = enabled
        self.max_rows = max(1, int(max_rows))
        self._rows: List[Dict[str, Any]] = []
        self.started_at = datetime.now().isoformat(timespec="seconds")

    def log(self, event: str, **fields):
        if not self.enabled:
            return
        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
        }
        row.update(fields)
        self._rows.append(row)
        if len(self._rows) > self.max_rows:
            del self._rows[: len(self._rows) - self.max_rows]

    def flush(self, path: str):
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # collect all keys for consistent columns
        keys = ["timestamp", "event"]
        for r in self._rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in self._rows:
                writer.writerow({k: r.get(k, "") for k in keys})

    def clear(self):
        self._rows.clear()
