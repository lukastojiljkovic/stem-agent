from pathlib import Path
import pytest

from stem_agent.agent.pipeline import Pipeline, PipelineStep
from stem_agent.agent.promote import consider_promotion
from stem_agent.tools.archive import MAPElitesArchive
from stem_agent.tools.base import Tool, ToolKind
from stem_agent.tools.registry import ToolLibrary
from stem_agent.types import TypeName


def _lib(tmp_path: Path):
    lib = ToolLibrary(tmp_path)
    lib._tools["A"] = Tool(name="A", description="", input_type=TypeName.TEXT, output_type=TypeName.CLAUSES, kind=ToolKind.PRIMITIVE)
    lib._tools["B"] = Tool(name="B", description="", input_type=TypeName.CLAUSES, output_type=TypeName.TEXT, kind=ToolKind.PRIMITIVE)
    return lib


def test_first_in_cell_is_promoted(tmp_path: Path):
    lib = _lib(tmp_path); arc = MAPElitesArchive()
    p = Pipeline([PipelineStep("A"), PipelineStep("B")])
    d, comp = consider_promotion(library=lib, archive=arc, pipeline=p, score=0.6, parent_score=0.0,
                                 domain="legal", subdomain="contract_analysis",
                                 capability_tag="clause_extraction", parent_ids=[], session_id="s_abc123",
                                 improvement_min=0.02)
    assert d.accepted is True
    assert comp is not None
    assert comp["id"] in lib.composites


def test_no_promotion_without_min_improvement(tmp_path: Path):
    lib = _lib(tmp_path); arc = MAPElitesArchive()
    p = Pipeline([PipelineStep("A"), PipelineStep("B")])
    d, comp = consider_promotion(library=lib, archive=arc, pipeline=p, score=0.6, parent_score=0.59,
                                 domain="legal", subdomain="contract_analysis",
                                 capability_tag="clause_extraction", parent_ids=["p1"], session_id="s_abc123",
                                 improvement_min=0.05)
    assert d.accepted is False
    assert comp is None
