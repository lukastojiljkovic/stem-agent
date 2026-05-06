"""Compile a TexProject to PDF using `latexmk -pdf` (MiKTeX provides this on Windows).

Behavior:
- Writes `<out_path>.tex` next to the requested PDF.
- Runs latexmk in batchmode; if missing falls back to `pdflatex` x2.
- After successful compile, deletes auxiliary files (.aux, .log, .out, .toc, .fls,
  .fdb_latexmk, .synctex.gz, .bbl, .blg).
- The original .tex is COPIED to <archive_tex_dir> as required by the spec.
- Returns absolute path to the produced PDF.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from ...types import TypeName
from ...ui.console import log_report, log_warn, log_error
from ..base import tool, ToolKind
from .latex_builder import render_tex


_AUX_EXTS = {".aux",".log",".out",".toc",".fls",".fdb_latexmk",".synctex.gz",".bbl",".blg",".lof",".lot"}


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


@tool(
    name="pdf_compile",
    description="Compile a TexProject to PDF, archive the source .tex, clean aux files. Returns PDF path.",
    input_type=TypeName.TEX_PROJECT,
    output_type=TypeName.PDF_PATH,
    kind=ToolKind.UNIVERSAL,
)
def pdf_compile(project: dict[str, Any], out_path: str,
                archive_tex_dir: str | None = None) -> str:
    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tex_path = out.with_suffix(".tex")
    tex_path.write_text(render_tex(project), encoding="utf-8")
    log_report(f"pdf_compile -> {out}")

    cwd = tex_path.parent
    ok = False
    if _have("latexmk"):
        cmd = ["latexmk","-pdf","-interaction=nonstopmode","-halt-on-error",
               "-quiet", str(tex_path.name)]
        try:
            r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=180)
            ok = (r.returncode == 0 and out.exists())
            if not ok: log_warn(f"latexmk rc={r.returncode}\n{(r.stderr or r.stdout)[-1500:]}")
        except Exception as e:
            log_warn(f"latexmk threw: {e}")
    if not ok and _have("pdflatex"):
        try:
            r = None
            for _ in range(2):
                r = subprocess.run(
                    ["pdflatex","-interaction=nonstopmode","-halt-on-error", str(tex_path.name)],
                    cwd=cwd, capture_output=True, text=True, timeout=180,
                )
            ok = out.exists()
            if not ok and r is not None: log_warn(f"pdflatex rc={r.returncode}")
        except Exception as e:
            log_error(f"pdflatex failed: {e}")

    if not ok:
        return str(tex_path)

    if archive_tex_dir:
        archive_dir = Path(archive_tex_dir); archive_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(tex_path, archive_dir / tex_path.name)
        except Exception as e:
            log_warn(f"tex archive failed: {e}")

    for f in cwd.iterdir():
        if f == out or f == tex_path: continue
        if f.suffix.lower() in _AUX_EXTS:
            try: f.unlink()
            except Exception: pass
    return str(out)
