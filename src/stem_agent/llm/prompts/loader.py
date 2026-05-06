"""Read prompt markdown from this directory and apply simple `{{var}}` substitution."""
from __future__ import annotations

import re
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, **vars: str) -> str:
    text = load_prompt(name)
    def repl(m: re.Match[str]) -> str:
        key = m.group(1).strip()
        return str(vars.get(key, m.group(0)))
    return re.sub(r"\{\{\s*(\w+)\s*\}\}", repl, text)
