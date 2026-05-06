"""Grammar/style check via language-tool-python (local, requires Java) with online fallback.

The tool reports issues but does NOT mutate the .tex source - that is left
to a future revision. Used as a quality signal in the final report stage.
"""
from __future__ import annotations

from typing import Any

from ...types import TypeName
from ...ui.console import log_report, log_warn
from ..base import tool, ToolKind


_LT = None


def _lt():
    global _LT
    if _LT is not None: return _LT
    try:
        import language_tool_python as ltp
        try:
            _LT = ltp.LanguageTool("en-US")
        except Exception:
            log_warn("language_tool: local Java engine unavailable; using public API")
            _LT = ltp.LanguageToolPublicAPI("en-US")
        return _LT
    except Exception as e:
        log_warn(f"language_tool unavailable: {e}")
        return None


@tool(
    name="grammar_check",
    description="Run grammar/style check on the TexProject's prose. Returns list of {message,offset,length,replacements}.",
    input_type=TypeName.TEX_PROJECT,
    output_type=TypeName.ISSUES_LIST,
    kind=ToolKind.UNIVERSAL,
)
def grammar_check(project: dict[str, Any]) -> list[dict[str, Any]]:
    log_report("grammar_check")
    lt = _lt()
    if lt is None:
        return []
    pieces: list[str] = []
    for s in project["sections"]:
        if s.get("level") == "raw":
            continue
        body = s.get("body","")
        text_only = body.replace(r"\textbf", "").replace(r"\textit", "")
        pieces.append(text_only)
    text = "\n\n".join(pieces)
    if not text.strip(): return []
    issues = []
    try:
        for m in lt.check(text)[:50]:
            issues.append({
                "message": m.message,
                "offset": m.offset,
                "length": m.errorLength,
                "replacements": m.replacements[:3],
                "rule_id": m.ruleId,
            })
    except Exception as e:
        log_warn(f"grammar_check failed: {e}")
    return issues
