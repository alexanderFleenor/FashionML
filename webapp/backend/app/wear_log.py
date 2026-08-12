"""Simple JSON wear log with one entry per worn outfit."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


class WearLog:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        if not self.path.exists():
            self.path.write_text("[]")

    def _load(self) -> List[Dict]:
        try:
            return json.loads(self.path.read_text() or "[]")
        except json.JSONDecodeError:
            return []

    def append(self, item_ids: List[str]) -> Dict:
        entry = {
            "worn_at": datetime.now(timezone.utc).isoformat(),
            "item_ids": item_ids,
        }
        with self._lock:
            entries = self._load()
            entries.append(entry)
            self.path.write_text(json.dumps(entries, indent=2))
        return entry

    def recent(self, days: int = 14) -> List[Dict]:
        """Return entries from the last N days."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        out = []
        for entry in self._load():
            try:
                ts = datetime.fromisoformat(entry["worn_at"])
                if ts >= cutoff:
                    out.append(entry)
            except (KeyError, ValueError):
                continue
        return out

    def all(self) -> List[Dict]:
        return self._load()
