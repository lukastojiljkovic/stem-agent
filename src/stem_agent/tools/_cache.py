"""Tiny on-disk JSON cache used by retrieval tools.

Key is a stable SHA-256 of (namespace, args_dict). Value is JSON-serializable.
TTL in seconds; 0 disables expiry. No locking — assumes single-process agent.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


class JsonCache:
    def __init__(self, namespace: str, ttl_s: int = 7 * 24 * 3600,
                 root: str | Path | None = None):
        base = Path(root) if root else Path(os.environ.get("STEM_CACHE_ROOT", ".cache"))
        self.dir = base / namespace
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_s

    @staticmethod
    def _key(payload: dict[str, Any]) -> str:
        s = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]

    def get(self, payload: dict[str, Any]) -> Any | None:
        path = self.dir / f"{self._key(payload)}.json"
        if not path.exists():
            return None
        if self.ttl > 0 and (time.time() - path.stat().st_mtime) > self.ttl:
            try: path.unlink()
            except Exception: pass
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def put(self, payload: dict[str, Any], value: Any) -> None:
        path = self.dir / f"{self._key(payload)}.json"
        try:
            path.write_text(json.dumps(value, ensure_ascii=False, default=str), encoding="utf-8")
        except Exception:
            pass
