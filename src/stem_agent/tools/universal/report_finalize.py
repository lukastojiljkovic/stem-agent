"""Orchestrates universal tools to produce the agent's final answer PDF.

Inputs: a free-form `answer_md` (markdown), an optional `metrics_table` (rows),
plus arbitrary `figures` to embed. Outputs the absolute path to the generated PDF.

This is the ONLY function the agent's outer driver should invoke directly to
materialize a final answer; sub-tools are still invokable from inside but
are typed-coerced compatible.
"""
from __future__ import annotations

from typing import Any

from ...types import TypeName
from ...ui.console import log_report
from ..base import tool, ToolKind
from .latex_builder import latex_init, latex_section, latex_table, latex_chart
from .grammar import grammar_check
from .pdf import pdf_compile


@tool(
    name="report_finalize",
    description="Build a TexProject from answer_md + optional metrics_table + figures, run grammar check, compile PDF.",
    input_type=TypeName.STRUCTURED_DATA,
    output_type=TypeName.PDF_PATH,
    kind=ToolKind.UNIVERSAL,
)
def report_finalize(payload: dict[str, Any]) -> str:
    title = payload.get("title", "Stem Agent — Answer")
    body_md = payload.get("answer_md", "")
    metrics = payload.get("metrics_table") or []
    figures = payload.get("figures") or []
    out_path = payload.get("out_path") or "runs/_unfiled/agent_answer.pdf"
    archive_dir = payload.get("archive_tex_dir")

    p = latex_init(title)
    p = latex_section(p, "Answer", body_md, level="section")
    if metrics:
        p = latex_table(p, metrics, caption="Metrics", long=False)
    for fig in figures:
        p = latex_chart(p, fig.get("data", {}), kind=fig.get("kind", "bar"),
                        caption=fig.get("caption",""), out_dir=fig.get("out_dir","report/figures"))
    issues = grammar_check(p)
    if issues:
        log_report(f"grammar issues: {len(issues)} (informational only)")
    return pdf_compile(p, out_path, archive_tex_dir=archive_dir)
