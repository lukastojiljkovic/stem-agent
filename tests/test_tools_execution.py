import pytest

from stem_agent.tools.execution import python_exec


def test_basic_arithmetic():
    out = python_exec("result = 2 + 2", ctx={})
    assert out["ok"] is True
    assert out["result"] == 4


def test_uses_ctx_vars():
    out = python_exec("result = sum(xs)", ctx={"xs": [1,2,3,4]})
    assert out["ok"] is True
    assert out["result"] == 10


def test_blocks_unwhitelisted_import():
    out = python_exec("import os\nresult = 1", ctx={})
    assert out["ok"] is False
    assert "not whitelisted" in (out.get("error") or "")


def test_timeout_kills_long_loop():
    out = python_exec("while True: pass", ctx={}, timeout=1.0)
    assert out["ok"] is False
    assert "Timeout" in (out.get("error") or "")
