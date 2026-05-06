"""Tool base class + `@tool` decorator + frozen-universal registry.

A Tool is a typed callable with metadata sufficient to be serialized into
the ToolLibrary, retrieved by typed signature or embedding similarity, and
composed into a Pipeline.

Universal frozen tools (latex_*, grammar_check, pdf_compile, report_finalize)
are tagged with kind=UNIVERSAL and the registry refuses to consider them as
candidates during evolution.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable

from ..types import TypeName


class ToolKind(str, Enum):
    PRIMITIVE = "primitive"
    COMPOSITE = "composite"
    UNIVERSAL = "universal"


FROZEN_UNIVERSAL: set[str] = {
    "latex_init",
    "latex_section",
    "latex_table",
    "latex_chart",
    "grammar_check",
    "pdf_compile",
    "report_finalize",
}


@dataclass
class Tool:
    name: str
    description: str
    input_type: TypeName
    output_type: TypeName
    kind: ToolKind = ToolKind.PRIMITIVE
    domain: str | None = None
    subdomain: str | None = None
    capability_tag: str | None = None
    parameters_schema: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.05
    function: Callable[..., Any] | None = field(default=None, repr=False)

    def run(self, **kwargs) -> Any:
        if self.function is None:
            raise RuntimeError(f"Tool {self.name} has no bound function.")
        return self.function(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["input_type"] = self.input_type.value
        d["output_type"] = self.output_type.value
        d["kind"] = self.kind.value
        d.pop("function", None)
        return d


def tool(
    *,
    name: str,
    description: str,
    input_type: TypeName,
    output_type: TypeName,
    kind: ToolKind = ToolKind.PRIMITIVE,
    domain: str | None = None,
    subdomain: str | None = None,
    capability_tag: str | None = None,
    cost: float = 0.05,
    parameters_schema: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: wrap a python callable into a Tool with metadata."""
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fn)
        ps = parameters_schema or _infer_param_schema(sig)
        meta = Tool(
            name=name,
            description=description,
            input_type=input_type,
            output_type=output_type,
            kind=kind,
            domain=domain,
            subdomain=subdomain,
            capability_tag=capability_tag,
            parameters_schema=ps,
            cost=cost,
            function=fn,
        )
        if kind == ToolKind.UNIVERSAL and name not in FROZEN_UNIVERSAL:
            FROZEN_UNIVERSAL.add(name)
        fn.tool = meta  # type: ignore[attr-defined]
        return fn
    return deco


def _infer_param_schema(sig: inspect.Signature) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pname, p in sig.parameters.items():
        if pname == "self":
            continue
        out[pname] = {
            "type": _annotation_to_str(p.annotation),
            "required": p.default is inspect.Parameter.empty,
            "default": None if p.default is inspect.Parameter.empty else _safe_default(p.default),
        }
    return out


def _annotation_to_str(ann: Any) -> str:
    if ann is inspect.Parameter.empty:
        return "Any"
    return getattr(ann, "__name__", None) or str(ann)


def _safe_default(v: Any) -> Any:
    try:
        import json
        json.dumps(v, default=str)
        return v
    except Exception:
        return str(v)
