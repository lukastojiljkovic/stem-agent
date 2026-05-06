"""Sandboxed Python exec via RestrictedPython, with timeout and output capture.

Permissions:
- No network, no filesystem write outside an explicit /tmp work dir.
- Importable modules: math, statistics, json, re, datetime, itertools.
- Variables passed in via `ctx` dict; result must be assigned to `result`.
"""
from __future__ import annotations

import io
import math
import multiprocessing
import statistics
import json
import re
import datetime
import itertools
from contextlib import redirect_stdout
from typing import Any

from RestrictedPython import compile_restricted, safe_builtins
from RestrictedPython.Eval import default_guarded_getitem
from RestrictedPython.Guards import safe_globals as rp_safe_globals, guarded_iter_unpack_sequence

from ..types import TypeName
from ..ui.console import log_tool
from .base import tool, ToolKind


_ALLOWED_IMPORTS = {"math","statistics","json","re","datetime","itertools"}


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name not in _ALLOWED_IMPORTS:
        raise ImportError(f"Module {name!r} is not whitelisted")
    return __import__(name, globals, locals, fromlist, level)


def _safe_globals(ctx: dict[str, Any]) -> dict[str, Any]:
    g = dict(rp_safe_globals)
    g["__builtins__"] = {**safe_builtins, "__import__": _restricted_import,
                         "len": len, "range": range, "min": min, "max": max,
                         "sum": sum, "abs": abs, "round": round, "sorted": sorted,
                         "list": list, "dict": dict, "tuple": tuple, "set": set,
                         "str": str, "int": int, "float": float, "bool": bool,
                         "enumerate": enumerate, "zip": zip, "any": any, "all": all,
                         "print": print}
    g["_getitem_"] = default_guarded_getitem
    g["_getiter_"] = iter
    g["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
    g["math"] = math; g["statistics"] = statistics; g["json"] = json; g["re"] = re
    g["datetime"] = datetime; g["itertools"] = itertools
    g.update(ctx)
    return g


def _runner(code: str, ctx: dict[str, Any], q):
    try:
        compiled = compile_restricted(code, filename="<exec>", mode="exec")
        env = _safe_globals(ctx)
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(compiled, env)
        q.put({"ok": True, "result": env.get("result"), "stdout": buf.getvalue()[:4000]})
    except Exception as e:
        q.put({"ok": False, "error": f"{type(e).__name__}: {e}", "stdout": ""})


@tool(
    name="python_exec",
    description="Execute restricted Python with `ctx` available; assign final value to `result`.",
    input_type=TypeName.STRUCTURED_DATA,
    output_type=TypeName.EXEC_RESULT,
    kind=ToolKind.PRIMITIVE,
    cost=0.10,
)
def python_exec(code: str, ctx: dict[str, Any] | None = None, timeout: float = 10.0) -> dict[str, Any]:
    log_tool(f"python_exec timeout={timeout}")
    ctx = ctx or {}
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_runner, args=(code, ctx, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(1.0)
        return {"ok": False, "error": f"Timeout after {timeout}s", "stdout": "", "result": None}
    try:
        return q.get_nowait()
    except Exception:
        return {"ok": False, "error": "no_result", "stdout": "", "result": None}
