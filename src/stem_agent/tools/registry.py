"""Persistent ToolLibrary with typed (symbolic) + embedding-based retrieval.

File layout under tool_library/:
    primitives.json         -- list of primitive Tool dicts (informational; the live
                               objects are constructed at import time and registered)
    composites.json         -- list of composite tool dicts (graduated pipelines)
    archive/<session-id>/   -- per-session snapshots
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import Tool, ToolKind
from ..types import TypeName, is_compatible


_MODEL = None


def _embed_text(text: str) -> list[float]:
    """Embed a description with sentence-transformers MiniLM. Lazy import; CPU."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return [0.0] * 384
    global _MODEL
    if _MODEL is None:
        try:
            _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            return [0.0] * 384
    try:
        return _MODEL.encode([text], normalize_embeddings=True)[0].tolist()
    except Exception:
        return [0.0] * 384


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    s = sum(x*y for x, y in zip(a, b))
    return float(s)


class ToolLibrary:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._tools: dict[str, Tool] = {}
        self._embeds: dict[str, list[float]] = {}
        self.composites: dict[str, dict[str, Any]] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        text = f"{tool.name}: {tool.description}"
        self._embeds[tool.name] = _embed_text(text)

    def register_composite(self, comp: dict[str, Any]) -> None:
        if not comp.get("embedding"):
            comp["embedding"] = _embed_text(comp.get("description") or comp.get("id", ""))
        self.composites[comp["id"]] = comp
        # Also expose the composite as a callable Tool so evolution can compose
        # composites with each other (composite-of-composites). The wrapper
        # delegates to Pipeline.execute on the underlying typed steps.
        try:
            self._register_composite_as_tool(comp)
        except Exception:
            # Wrapper registration is best-effort; missing it doesn't break the run.
            pass

    def _register_composite_as_tool(self, comp: dict[str, Any]) -> None:
        cid = comp["id"]
        # Resolve types from string back to TypeName.
        try:
            in_t = TypeName(comp["input_type"])
            out_t = TypeName(comp["output_type"])
        except Exception:
            return
        steps = comp.get("steps") or []
        if not steps:
            return
        # Reject self-reference at registration time (prevents infinite recursion
        # when the agent picks a composite as a step in itself).
        if any(s.get("tool") == cid for s in steps):
            return
        registry_self = self

        def _runner(**kwargs: Any) -> Any:
            from ..agent.pipeline import Pipeline, PipelineStep, execute as _execute
            # Hard depth cap on composite-of-composites recursion. Caps at 3
            # nested composites, more than enough at the scale we operate.
            import threading
            tls = getattr(_runner, "_tls", None)
            if tls is None:
                tls = threading.local(); _runner._tls = tls  # type: ignore[attr-defined]
            depth = getattr(tls, "depth", 0)
            if depth >= 3:
                return None
            tls.depth = depth + 1
            try:
                inner = Pipeline([PipelineStep(s["tool"], dict(s.get("params") or {}))
                                  for s in steps])
                preferred = ["text", "query", "doc", "document", "documents",
                             "time_series", "filing", "data", "output", "project"]
                initial = None
                for p in preferred:
                    if p in kwargs:
                        initial = kwargs[p]; break
                if initial is None and kwargs:
                    initial = next(iter(kwargs.values()))
                res = _execute(inner, registry_self._tools, initial)
                return res.final if res.success else None
            finally:
                tls.depth = depth

        wrapper = Tool(
            name=cid,
            description=f"[composite] {comp.get('description') or cid}",
            input_type=in_t,
            output_type=out_t,
            kind=ToolKind.COMPOSITE,
            domain=comp.get("domain"),
            subdomain=comp.get("subdomain"),
            capability_tag=comp.get("capability_tag"),
            cost=0.20,
            function=_runner,
        )
        self._tools[cid] = wrapper
        self._embeds[cid] = comp.get("embedding") or _embed_text(comp.get("description") or cid)

    def __contains__(self, name: str) -> bool:
        return name in self._tools or name in self.composites

    def has(self, name: str) -> bool:
        """Public membership check for primitives (composites live in `composites`)."""
        return name in self._tools

    def get(self, name: str) -> Tool:
        if name in self._tools:
            return self._tools[name]
        raise KeyError(name)

    def all_primitives(self) -> list[Tool]:
        return [t for t in self._tools.values() if t.kind == ToolKind.PRIMITIVE]

    def evolution_candidates(self, producer_type: TypeName | None = None,
                             domain: str | None = None) -> list[Tool]:
        out: list[Tool] = []
        for t in self._tools.values():
            if t.kind == ToolKind.UNIVERSAL:
                continue
            if producer_type is not None and not is_compatible(producer_type, t.input_type):
                continue
            if domain and t.domain and t.domain != domain and t.domain != "general":
                continue
            out.append(t)
        return out

    def compatible_consumers(self, producer_type: TypeName) -> list[Tool]:
        return [t for t in self._tools.values()
                if t.kind != ToolKind.UNIVERSAL and is_compatible(producer_type, t.input_type)]

    def search_by_description(self, query: str, k: int = 5) -> list[Tool]:
        q = _embed_text(query)
        scored: list[tuple[float, Tool]] = []
        for n, e in self._embeds.items():
            if self._tools[n].kind == ToolKind.UNIVERSAL:
                continue
            scored.append((_cosine(q, e), self._tools[n]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:k]]

    @property
    def composites_path(self) -> Path:
        return self.root / "composites.json"

    @property
    def primitives_meta_path(self) -> Path:
        return self.root / "primitives.json"

    @property
    def embeddings_path(self) -> Path:
        return self.root / "embeddings.json"

    def save(self) -> None:
        # Write composites without their inline 384-d embeddings so the
        # composites.json file stays human-skimmable. Embeddings live alongside
        # in a sidecar file keyed by composite id.
        primitives_meta = [t.to_dict() for t in self._tools.values() if t.kind == ToolKind.PRIMITIVE]
        self.primitives_meta_path.write_text(
            json.dumps(primitives_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        slim_composites: list[dict[str, Any]] = []
        embeddings: dict[str, list[float]] = {}
        for c in self.composites.values():
            slim = {k: v for k, v in c.items() if k != "embedding"}
            slim_composites.append(slim)
            emb = c.get("embedding")
            if emb:
                embeddings[c["id"]] = list(emb)
        self.composites_path.write_text(
            json.dumps(slim_composites, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.embeddings_path.write_text(
            json.dumps(embeddings, ensure_ascii=False), encoding="utf-8"
        )

    def load(self) -> None:
        if self.composites_path.exists():
            data = json.loads(self.composites_path.read_text(encoding="utf-8"))
            embeddings: dict[str, list[float]] = {}
            if self.embeddings_path.exists():
                try:
                    embeddings = json.loads(self.embeddings_path.read_text(encoding="utf-8")) or {}
                except Exception:
                    embeddings = {}
            self.composites = {}
            for c in data:
                if "embedding" not in c and c["id"] in embeddings:
                    c["embedding"] = embeddings[c["id"]]
                # Re-register composites (also sets up the composite-as-tool wrapper).
                self.register_composite(c)

    def snapshot(self, session_id: str) -> Path:
        target = self.root / "archive" / session_id
        target.mkdir(parents=True, exist_ok=True)
        (target / "primitives.json").write_text(
            json.dumps([t.to_dict() for t in self._tools.values()], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (target / "composites.json").write_text(
            json.dumps(list(self.composites.values()), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return target
