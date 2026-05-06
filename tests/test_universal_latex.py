from stem_agent.tools.universal.latex_builder import (
    latex_init, latex_section, latex_table, render_tex
)


def test_init_basic():
    p = latex_init("Hello & World")
    assert "Hello \\& World" in p["title"]
    assert p["sections"] == []


def test_section_appends():
    p = latex_init("T")
    p = latex_section(p, "Intro", "Some **bold** text\n- point 1\n- point 2")
    assert any("Intro" in s["name"] for s in p["sections"])


def test_table_renders_booktabs():
    p = latex_init("T")
    p = latex_table(p, [["a","b"],["1","2"]], caption="cap")
    body = p["sections"][-1]["body"]
    assert r"\toprule" in body and r"\bottomrule" in body
    assert "a & b" in body and "1 & 2" in body


def test_render_full_document():
    p = latex_init("My Doc")
    p = latex_section(p, "Intro", "Hello world.")
    tex = render_tex(p)
    assert tex.lstrip().startswith("\\documentclass")
    assert r"\begin{document}" in tex and r"\end{document}" in tex
    assert "My Doc" in tex
