"""LaTeX project builder. A 'TexProject' is a dict {main, sections, figures, table_count}.

Tools here NEVER execute pdflatex — that is `pdf_compile`'s job. They only
build the .tex source string. This separation keeps each tool small and
unit-testable without a LaTeX install.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ...types import TypeName
from ...ui.console import log_report
from ..base import tool, ToolKind


_LATEX_SPECIALS = str.maketrans({
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
})


def _escape(s: str) -> str:
    return (s or "").translate(_LATEX_SPECIALS)


_PDFLATEX_PREAMBLE = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{microtype}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\hypersetup{colorlinks=true, urlcolor=blue, linkcolor=black}
"""


@tool(
    name="latex_init",
    description="Start a TexProject. Returns project dict with title/preamble/sections placeholder.",
    input_type=TypeName.TEXT,
    output_type=TypeName.TEX_PROJECT,
    kind=ToolKind.UNIVERSAL,
)
def latex_init(title: str, author: str = "Stem Agent",
               engine: str = "pdflatex",
               tex_dir: str | None = None) -> dict[str, Any]:
    """Start a TexProject. ``tex_dir`` is the directory the .tex file will be
    written to; downstream tools (latex_chart) use it to compute portable
    relative include paths instead of leaking absolute system paths."""
    log_report(f"latex_init engine={engine} title={title!r}")
    project: dict[str, Any] = {
        "engine": engine,
        "title": _escape(title),
        "author": _escape(author),
        "preamble": _PDFLATEX_PREAMBLE,
        "sections": [],
        "figures": [],
        "table_count": 0,
        "tex_dir": tex_dir,
    }
    return project


def _md_to_tex(md: str) -> str:
    """Minimal Markdown -> LaTeX. Bullets, italic, bold, inline code, paragraphs."""
    lines = (md or "").splitlines()
    out: list[str] = []
    in_list = False
    for line in lines:
        if re.match(r"\s*[-*]\s+", line):
            if not in_list:
                out.append(r"\begin{itemize}[leftmargin=*]"); in_list = True
            content = re.sub(r"^\s*[-*]\s+", "", line)
            out.append(r"\item " + _inline_md_to_tex(content))
        else:
            if in_list:
                out.append(r"\end{itemize}"); in_list = False
            out.append(_inline_md_to_tex(line))
    if in_list:
        out.append(r"\end{itemize}")
    return "\n".join(out)


def _inline_md_to_tex(s: str) -> str:
    s = _escape(s)
    s = re.sub(r"\\\*\\\*(.+?)\\\*\\\*", r"\\textbf{\1}", s)
    s = re.sub(r"\\\*(.+?)\\\*", r"\\textit{\1}", s)
    s = re.sub(r"`([^`]+)`", r"\\texttt{\1}", s)
    return s


@tool(
    name="latex_section",
    description="Append a section (with markdown body) to the TexProject.",
    input_type=TypeName.TEX_PROJECT,
    output_type=TypeName.TEX_PROJECT,
    kind=ToolKind.UNIVERSAL,
)
def latex_section(project: dict[str, Any], name: str, content_md: str,
                  level: str = "section") -> dict[str, Any]:
    log_report(f"latex_section ({level}) {name!r}")
    body = _md_to_tex(content_md or "")
    project["sections"].append({"level": level, "name": _escape(name), "body": body})
    return project


@tool(
    name="latex_table",
    description="Append a table from a list of rows (first row = header). Uses booktabs.",
    input_type=TypeName.TEX_PROJECT,
    output_type=TypeName.TEX_PROJECT,
    kind=ToolKind.UNIVERSAL,
)
def latex_table(project: dict[str, Any], rows: list[list[str]],
                caption: str = "", label: str | None = None,
                long: bool = False) -> dict[str, Any]:
    log_report(f"latex_table rows={len(rows)} caption={caption!r}")
    if not rows: return project
    project["table_count"] += 1
    label = label or f"tab:t{project['table_count']}"
    n_cols = len(rows[0])
    align = "l" + "r" * (n_cols - 1)
    if long:
        block = [r"\begin{longtable}{" + align + "}",
                 r"\caption{" + _escape(caption) + r"} \\",
                 r"\toprule",
                 " & ".join(_escape(str(c)) for c in rows[0]) + r" \\",
                 r"\midrule",
                 r"\endfirsthead",
                 r"\toprule",
                 " & ".join(_escape(str(c)) for c in rows[0]) + r" \\",
                 r"\midrule",
                 r"\endhead",
                 r"\bottomrule",
                 r"\endfoot",
                 ]
        for r in rows[1:]:
            block.append(" & ".join(_escape(str(c)) for c in r) + r" \\")
        block.append(r"\end{longtable}")
        project["sections"].append({"level": "raw", "name": "", "body": "\n".join(block)})
    else:
        block = [r"\begin{table}[h]", r"\centering",
                 r"\begin{tabular}{" + align + "}", r"\toprule",
                 " & ".join(_escape(str(c)) for c in rows[0]) + r" \\",
                 r"\midrule"]
        for r in rows[1:]:
            block.append(" & ".join(_escape(str(c)) for c in r) + r" \\")
        block += [r"\bottomrule", r"\end{tabular}",
                  r"\caption{" + _escape(caption) + r"}",
                  r"\label{" + label + r"}", r"\end{table}"]
        project["sections"].append({"level": "raw", "name": "", "body": "\n".join(block)})
    return project


@tool(
    name="latex_chart",
    description="Render a matplotlib chart to PDF and append \\includegraphics. data is dict by chart kind.",
    input_type=TypeName.TEX_PROJECT,
    output_type=TypeName.TEX_PROJECT,
    kind=ToolKind.UNIVERSAL,
)
def latex_chart(project: dict[str, Any], data: dict[str, Any],
                kind: str = "bar", caption: str = "",
                out_dir: str = "report/figures") -> dict[str, Any]:
    log_report(f"latex_chart kind={kind} caption={caption!r}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        fname = f"chart_{len(project['figures'])+1:02d}.pdf"
        path = out / fname
        fig = plt.figure(figsize=(6, 3.5))
        ax = fig.add_subplot(111)
        if kind == "bar":
            xs = data.get("labels") or []; ys = data.get("values") or []
            ax.bar(xs, ys); ax.set_ylabel(data.get("ylabel",""))
        elif kind == "line":
            xs = data.get("x") or list(range(len(data.get("y") or [])))
            ax.plot(xs, data.get("y") or [])
            ax.set_xlabel(data.get("xlabel","")); ax.set_ylabel(data.get("ylabel",""))
        else:
            ax.text(0.5, 0.5, f"unknown chart kind: {kind}", ha="center")
        ax.set_title(data.get("title","") or caption)
        fig.tight_layout(); fig.savefig(path); plt.close(fig)
        # Resolve a portable include path: if a tex_dir is set on the project,
        # store the figure path *relative to* the .tex file's directory so the
        # generated .tex never contains absolute system paths (privacy + portability).
        tex_dir = project.get("tex_dir")
        try:
            include_path = (
                Path(path).resolve().relative_to(Path(tex_dir).resolve()).as_posix()
                if tex_dir else Path(path).as_posix()
            )
        except ValueError:
            # Path isn't under tex_dir; use just the bare filename rather than leak
            # an absolute path. The caller is responsible for ensuring figures end
            # up alongside the .tex when tex_dir isn't a parent of out_dir.
            include_path = Path(path).name
        project["figures"].append((str(path), caption))
        block = [r"\begin{figure}[h]", r"\centering",
                 r"\includegraphics[width=0.85\linewidth]{" + include_path + r"}",
                 r"\caption{" + _escape(caption) + r"}", r"\end{figure}"]
        project["sections"].append({"level":"raw","name":"","body":"\n".join(block)})
    except Exception as e:
        project["sections"].append({"level":"raw","name":"",
                                    "body": r"\textit{[chart render failed: " + _escape(str(e)) + r"]}"})
    return project


def render_tex(project: dict[str, Any]) -> str:
    parts: list[str] = [project["preamble"],
                        r"\title{" + project["title"] + r"}",
                        r"\author{" + project["author"] + r"}",
                        r"\date{\today}",
                        r"\begin{document}",
                        r"\maketitle"]
    for s in project["sections"]:
        if s["level"] == "raw":
            parts.append(s["body"])
        else:
            cmd = {"section": r"\section", "subsection": r"\subsection",
                   "subsubsection": r"\subsubsection"}.get(s["level"], r"\section")
            parts.append(cmd + "{" + s["name"] + "}")
            parts.append(s["body"])
    parts.append(r"\end{document}")
    return "\n".join(parts)
