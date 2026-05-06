"""Render the lineage graph of composite tools to Graphviz DOT format.

Nodes are composite IDs labeled with the capability_tag and the latest score.
Edges are parent -> child. Cells are clustered by (domain, capability_tag).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def render_dot(composites: Iterable[dict[str, Any]]) -> str:
    nodes_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    edges: list[tuple[str, str]] = []
    for c in composites:
        cell = (c.get("domain") or "general", c.get("capability_tag") or "generic")
        nodes_by_cell.setdefault(cell, []).append(c)
        for p in (c.get("lineage_parent_ids") or []):
            edges.append((p, c["id"]))

    lines = ["digraph Lineage {",
             '  rankdir="LR";',
             '  node [shape=box, style="rounded,filled", fillcolor="#eef"];']

    for (domain, cap), comps in nodes_by_cell.items():
        cluster_id = f"cluster_{domain}_{cap}".replace("-", "_")
        lines.append(f'  subgraph {cluster_id} {{')
        lines.append(f'    label="{domain} / {cap}";')
        lines.append('    style=dashed;')
        for c in comps:
            score = (c.get("metrics_history") or [{}])[-1].get("f1") \
                or (c.get("metrics_history") or [{}])[-1].get("step_avg") \
                or (c.get("metrics_history") or [{}])[-1].get("score") or 0.0
            label = f'{c["id"]}\\nscore={score:.2f}'
            lines.append(f'    "{c["id"]}" [label="{label}"];')
        lines.append('  }')

    for parent, child in edges:
        lines.append(f'  "{parent}" -> "{child}";')
    lines.append("}")
    return "\n".join(lines)


def write_dot(composites: Iterable[dict[str, Any]], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_dot(composites), encoding="utf-8")
    return p
