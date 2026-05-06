"""Tests for ToolLibrary persistence + typed retrieval + embedding fallback."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from stem_agent.tools.base import Tool, ToolKind
from stem_agent.tools.registry import ToolLibrary
from stem_agent.types import TypeName


def make_primitive(name: str, in_t=TypeName.TEXT, out_t=TypeName.TEXT, domain=None) -> Tool:
    return Tool(
        name=name,
        description=f"Test tool {name}",
        input_type=in_t,
        output_type=out_t,
        kind=ToolKind.PRIMITIVE,
        domain=domain,
    )


def test_register_and_lookup(tmp_path: Path):
    lib = ToolLibrary(tmp_path)
    t = make_primitive("alpha")
    lib.register(t)
    assert "alpha" in lib
    assert lib.get("alpha").name == "alpha"


def test_compatible_consumers_typed(tmp_path: Path):
    lib = ToolLibrary(tmp_path)
    lib.register(make_primitive("a", out_t=TypeName.DOCUMENTS))
    lib.register(make_primitive("b", in_t=TypeName.TEXT))
    lib.register(make_primitive("c", in_t=TypeName.SCORE))
    consumers = [t.name for t in lib.compatible_consumers(producer_type=TypeName.DOCUMENTS)]
    assert "b" in consumers
    assert "c" not in consumers


def test_excludes_universal_from_evolution_candidates(tmp_path: Path):
    lib = ToolLibrary(tmp_path)
    u = Tool(name="latex_init", description="frozen", input_type=TypeName.TEXT,
             output_type=TypeName.TEX_PROJECT, kind=ToolKind.UNIVERSAL)
    p = make_primitive("usable")
    lib.register(u)
    lib.register(p)
    cands = [t.name for t in lib.evolution_candidates(producer_type=TypeName.TEXT)]
    assert "latex_init" not in cands
    assert "usable" in cands


def test_persist_and_reload(tmp_path: Path):
    lib1 = ToolLibrary(tmp_path)
    lib1.register(make_primitive("alpha", domain="legal"))
    lib1.register_composite({
        "id": "comp_v1",
        "kind": "composite",
        "domain": "legal",
        "subdomain": "contract_analysis",
        "capability_tag": "clause_extraction",
        "input_type": "Document",
        "output_type": "Clauses",
        "steps": [{"tool": "alpha", "params": {}}],
        "lineage_parent_ids": [],
        "lineage_depth": 0,
        "metrics_history": [],
        "embedding": [0.0]*8,
        "description": "",
        "born_at_session": "2026-05-06T00:00:00",
        "version": 1,
    })
    lib1.save()

    lib2 = ToolLibrary(tmp_path)
    lib2.load()
    assert "comp_v1" in lib2.composites
    assert lib2.composites["comp_v1"]["domain"] == "legal"


@patch("stem_agent.tools.registry._embed_text", return_value=[0.0]*8)
def test_embedding_search_used_when_typed_yields_few(mock_embed, tmp_path: Path):
    lib = ToolLibrary(tmp_path)
    for n in ("alpha","beta","gamma"):
        lib.register(make_primitive(n))
    hits = lib.search_by_description("alphanumeric thing", k=2)
    assert isinstance(hits, list)
    assert len(hits) <= 2
