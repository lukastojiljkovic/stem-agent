"""Domain-economics primitives. Pure-numerical where possible."""
from __future__ import annotations

import math
from typing import Any

from ..types import TypeName
from ..ui.console import log_tool
from .base import tool, ToolKind


@tool(
    name="compute_indicators",
    description="Compute summary indicators from a TimeSeries: mean, std, min, max, last, pct_change_yoy.",
    input_type=TypeName.TIME_SERIES,
    output_type=TypeName.INDICATORS,
    kind=ToolKind.PRIMITIVE,
    domain="economics",
    capability_tag="indicators",
    cost=0.02,
)
def compute_indicators(time_series: dict[str, Any]) -> dict[str, Any]:
    log_tool("compute_indicators")
    vals = [v for v in (time_series.get("values") or []) if v is not None]
    if not vals:
        return {"series_id": time_series.get("series_id",""), "n": 0,
                "mean": None, "std": None, "min": None, "max": None, "last": None, "yoy_pct_change": None}
    mean = sum(vals)/len(vals)
    var = sum((v-mean)**2 for v in vals)/len(vals)
    std = math.sqrt(var)
    last = vals[-1]
    yoy = None
    if len(vals) >= 13:
        prev = vals[-13]
        if prev not in (0, None):
            yoy = (last/prev - 1.0) * 100.0
    return {"series_id": time_series.get("series_id",""),
            "n": len(vals), "mean": mean, "std": std,
            "min": min(vals), "max": max(vals), "last": last,
            "yoy_pct_change": yoy}


@tool(
    name="trend_analysis",
    description="Determine trend direction (up/down/flat) and strength (R^2 of linear fit) of a TimeSeries.",
    input_type=TypeName.TIME_SERIES,
    output_type=TypeName.TRENDS,
    kind=ToolKind.PRIMITIVE,
    domain="economics",
    capability_tag="trend",
    cost=0.02,
)
def trend_analysis(time_series: dict[str, Any]) -> dict[str, Any]:
    log_tool("trend_analysis")
    vals = [v for v in (time_series.get("values") or []) if v is not None]
    n = len(vals)
    if n < 3:
        return {"direction": "flat", "slope": 0.0, "r2": 0.0, "n": n}
    xs = list(range(n))
    mx = sum(xs)/n; my = sum(vals)/n
    num = sum((xs[i]-mx)*(vals[i]-my) for i in range(n))
    den = sum((xs[i]-mx)**2 for i in range(n))
    slope = num/den if den else 0.0
    intercept = my - slope*mx
    ss_tot = sum((v-my)**2 for v in vals) or 1e-12
    ss_res = sum((vals[i]-(slope*xs[i]+intercept))**2 for i in range(n))
    r2 = 1.0 - ss_res/ss_tot
    direction = "up" if slope > 0 and r2 > 0.1 else "down" if slope < 0 and r2 > 0.1 else "flat"
    return {"direction": direction, "slope": slope, "r2": r2, "n": n}


@tool(
    name="correlation_analysis",
    description="Compute pairwise Pearson correlations between aligned series in a dict {name: values_list}.",
    input_type=TypeName.STRUCTURED_DATA,
    output_type=TypeName.CORR_MATRIX,
    kind=ToolKind.PRIMITIVE,
    domain="economics",
    capability_tag="correlation",
    cost=0.05,
)
def correlation_analysis(series_dict: dict[str, list[float]]) -> dict[str, Any]:
    log_tool("correlation_analysis")
    names = list(series_dict.keys())
    if len(names) < 2:
        return {"names": names, "matrix": [[1.0]] if names else []}
    n = min(len(series_dict[k]) for k in names)
    cleaned = {k: [v for v in series_dict[k][-n:] if v is not None] for k in names}
    n = min(len(cleaned[k]) for k in names) if names else 0
    matrix: list[list[float]] = []
    for a in names:
        row: list[float] = []
        for b in names:
            row.append(_pearson(cleaned[a][-n:], cleaned[b][-n:]))
        matrix.append(row)
    return {"names": names, "matrix": matrix}


def _pearson(x: list[float], y: list[float]) -> float:
    n = min(len(x), len(y))
    if n < 2: return 0.0
    mx = sum(x[:n])/n; my = sum(y[:n])/n
    num = sum((x[i]-mx)*(y[i]-my) for i in range(n))
    dx = math.sqrt(sum((x[i]-mx)**2 for i in range(n)))
    dy = math.sqrt(sum((y[i]-my)**2 for i in range(n)))
    return num/(dx*dy) if dx and dy else 0.0


@tool(
    name="financial_ratios",
    description="Compute standard ratios + Altman Z + Piotroski F from a Filing's xbrl_facts.",
    input_type=TypeName.FILING,
    output_type=TypeName.RATIO_REPORT,
    kind=ToolKind.PRIMITIVE,
    domain="economics",
    capability_tag="financial_ratios",
    cost=0.05,
)
def financial_ratios(filing: dict[str, Any]) -> dict[str, Any]:
    log_tool("financial_ratios")
    f = (filing.get("xbrl_facts") or {}).copy()
    def g(k: str) -> float | None:
        v = f.get(k); return float(v) if isinstance(v, (int, float)) else None
    rev = g("revenue"); ni = g("net_income"); ta = g("total_assets")
    ca = g("current_assets"); cl = g("current_liabilities")
    tl = g("total_liabilities"); se = g("stockholders_equity")
    op = g("operating_income"); cfo = g("cash_from_ops")
    out: dict[str, Any] = {"ticker": filing.get("ticker", ""), "ratios": {}}

    def safe(num, den):
        if num is None or den is None or den == 0: return None
        return num/den

    out["ratios"]["current_ratio"] = safe(ca, cl)
    out["ratios"]["debt_equity"] = safe(tl, se)
    out["ratios"]["roa"] = safe(ni, ta)
    out["ratios"]["roe"] = safe(ni, se)
    out["ratios"]["operating_margin"] = safe(op, rev)

    if ca is not None and cl is not None and ta and ta != 0 and tl is not None and se is not None:
        wc = ca - cl
        x1 = wc/ta
        re_ = ni or 0.0; x2 = re_/ta
        x3 = (op or ni or 0.0)/ta
        x4 = (se / tl) if tl else 0
        z = 3.25 + 6.56*x1 + 3.26*x2 + 6.72*x3 + 1.05*x4
        out["ratios"]["altman_z"] = z
    else:
        out["ratios"]["altman_z"] = None

    f_score = 0
    if ni is not None and ni > 0: f_score += 1
    if cfo is not None and cfo > 0: f_score += 1
    if ni is not None and cfo is not None and cfo > ni: f_score += 1
    if out["ratios"].get("operating_margin") is not None and out["ratios"]["operating_margin"] > 0: f_score += 1
    if out["ratios"].get("current_ratio") is not None and out["ratios"]["current_ratio"] > 1: f_score += 1
    out["ratios"]["piotroski_f_partial"] = f_score
    return out
