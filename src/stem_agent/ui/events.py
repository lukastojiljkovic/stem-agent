"""Append-only JSONL event log; one event per line; UTF-8.

Used to feed the optional Streamlit dashboard and to enable post-hoc replay.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLog:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def session_log(session_id: str, runs_root: str | os.PathLike) -> EventLog:
    runs_root = Path(runs_root)
    return EventLog(runs_root / session_id / "event_log.jsonl")
