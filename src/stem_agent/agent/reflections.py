"""Minimal reflections store: short text snippets keyed by (domain, capability).

When a candidate pipeline fails or scores well below its peers, the runner can
record a one-paragraph "lesson" here. Future sessions read the most recent
reflections for the relevant cell and inject them into the agent's system prompt
alongside the domain brief, mimicking Voyager's "compress lessons into the
context" pattern.

This is intentionally tiny: a JSON file mapping
``"<domain>/<capability>" -> [{ts, text, score, session}, ...]``. We cap at the
last N reflections per cell so the file never bloats.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

_MAX_PER_CELL = 8


class ReflectionsStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, list[dict[str, Any]]] = {}
        self.load()

    @staticmethod
    def _key(domain: str | None, capability: str | None) -> str:
        return f"{domain or 'general'}/{capability or 'generic'}"

    def load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8")) or {}
        except Exception:
            self._data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    def add(self, *, domain: str | None, capability: str | None,
            text: str, score: float, session: str) -> None:
        if not text or not text.strip():
            return
        k = self._key(domain, capability)
        bucket = self._data.setdefault(k, [])
        bucket.append({
            "ts": dt.datetime.utcnow().isoformat(),
            "session": session,
            "score": score,
            "text": text.strip()[:600],
        })
        # Keep only the most recent N per cell.
        bucket.sort(key=lambda e: e.get("ts", ""))
        del bucket[:-_MAX_PER_CELL]

    def for_cell(self, domain: str | None, capability: str | None) -> list[dict[str, Any]]:
        return list(self._data.get(self._key(domain, capability), []))

    def render_for_prompt(self, domain: str | None, capability: str | None,
                          k: int = 3) -> str:
        entries = self.for_cell(domain, capability)
        if not entries:
            return ""
        recent = entries[-k:]
        lines = [f"Recent reflections for {self._key(domain, capability)} cell:"]
        for e in recent:
            lines.append(f"- (score={e.get('score', 0):.2f}) {e.get('text','')}")
        return "\n".join(lines)
