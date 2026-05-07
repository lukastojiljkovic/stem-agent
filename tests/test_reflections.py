"""Tests for the reflections store."""
import json
from pathlib import Path

from stem_agent.agent.reflections import ReflectionsStore


def test_add_persists_and_reloads(tmp_path: Path):
    p = tmp_path / "reflections.json"
    s1 = ReflectionsStore(p)
    s1.add(domain="legal", capability="clause_extraction",
           text="raising min_conf above 0.6 dropped recall sharply", score=0.42, session="abc")
    s1.save()

    s2 = ReflectionsStore(p)
    rendered = s2.render_for_prompt("legal", "clause_extraction")
    assert "min_conf" in rendered
    assert "score=0.42" in rendered


def test_caps_per_cell_to_max(tmp_path: Path):
    p = tmp_path / "r.json"
    s = ReflectionsStore(p)
    for i in range(20):
        s.add(domain="legal", capability="x",
              text=f"reflection {i}", score=0.1 * i, session="s")
    s.save()

    raw = json.loads(p.read_text(encoding="utf-8"))
    assert len(raw["legal/x"]) == 8
    # Most-recent entries kept.
    assert raw["legal/x"][-1]["text"] == "reflection 19"


def test_for_cell_empty_returns_empty_list(tmp_path: Path):
    s = ReflectionsStore(tmp_path / "r.json")
    assert s.for_cell("nope", "nada") == []
    assert s.render_for_prompt("nope", "nada") == ""
