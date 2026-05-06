"""Tests for JSONL event log writer."""
import json
from pathlib import Path

import pytest

from stem_agent.ui.events import EventLog


def test_event_log_appends(tmp_path: Path):
    log_file = tmp_path / "event_log.jsonl"
    log = EventLog(log_file)
    log.emit("session.start", session_id="abc")
    log.emit("tool.call", tool="web_search", query="hello")
    log.close()

    lines = log_file.read_text().splitlines()
    assert len(lines) == 2

    a = json.loads(lines[0])
    assert a["event"] == "session.start"
    assert a["session_id"] == "abc"
    assert "ts" in a

    b = json.loads(lines[1])
    assert b["event"] == "tool.call"
    assert b["tool"] == "web_search"


def test_event_log_safe_for_unicode(tmp_path: Path):
    log = EventLog(tmp_path / "e.jsonl")
    log.emit("legal.note", text="GDPR Čl. 5 — uslovi obrade")
    log.close()
    lines = (tmp_path / "e.jsonl").read_text(encoding="utf-8").splitlines()
    obj = json.loads(lines[0])
    assert obj["text"].startswith("GDPR")


def test_event_log_does_not_lose_data_on_repeated_open(tmp_path: Path):
    f = tmp_path / "e.jsonl"
    EventLog(f).emit("a", x=1)
    EventLog(f).emit("b", x=2)
    lines = f.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "a"
    assert json.loads(lines[1])["event"] == "b"
