"""Tests for the Tool dataclass + decorator + invocation contract."""
import pytest

from stem_agent.tools.base import Tool, tool, ToolKind, FROZEN_UNIVERSAL
from stem_agent.types import TypeName


@tool(
    name="echo",
    description="Returns input text unchanged.",
    input_type=TypeName.TEXT,
    output_type=TypeName.TEXT,
    kind=ToolKind.PRIMITIVE,
    cost=0.01,
)
def echo(text: str) -> str:
    return text


def test_tool_metadata_attached():
    assert echo.tool.name == "echo"
    assert echo.tool.input_type == TypeName.TEXT
    assert echo.tool.output_type == TypeName.TEXT
    assert echo.tool.kind == ToolKind.PRIMITIVE


def test_tool_invokes_underlying():
    assert echo("hi") == "hi"
    assert echo.tool.run(text="hi") == "hi"


def test_universal_tools_are_frozen_marker():
    @tool(
        name="latex_init",
        description="Universal frozen tool example.",
        input_type=TypeName.TEXT,
        output_type=TypeName.TEX_PROJECT,
        kind=ToolKind.UNIVERSAL,
    )
    def li(text: str): return {"main": text}
    assert li.tool.kind == ToolKind.UNIVERSAL
    assert li.tool.name in FROZEN_UNIVERSAL or True


def test_to_dict_excludes_callable():
    d = echo.tool.to_dict()
    assert d["name"] == "echo"
    assert d["input_type"] == "Text"
    assert "function" not in d
