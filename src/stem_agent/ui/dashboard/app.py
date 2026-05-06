"""Read-only Streamlit dashboard. Run with:

    streamlit run src/stem_agent/ui/dashboard/app.py

It does NOT trigger runs; it only visualizes what the agent is writing into
runs/<session>/event_log.jsonl. Refreshes every 2 seconds.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import streamlit as st


REPO = Path(__file__).resolve().parents[4]
RUNS = REPO / "runs"
TOOL_LIB = REPO / "tool_library"


def list_sessions() -> list[Path]:
    if not RUNS.exists(): return []
    return sorted([p for p in RUNS.iterdir() if p.is_dir() and not p.name.startswith("_")],
                  key=lambda p: p.stat().st_mtime, reverse=True)


def load_events(session_dir: Path) -> list[dict]:
    p = session_dir / "event_log.jsonl"
    if not p.exists(): return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try: out.append(json.loads(line))
        except Exception: pass
    return out


def session_picker() -> Path | None:
    sessions = list_sessions()
    if not sessions:
        st.info("No sessions yet. Run `python -m stem_agent run --domain legal` first.")
        return None
    pick = st.sidebar.selectbox("Session", [s.name for s in sessions])
    return next(s for s in sessions if s.name == pick)


def render_metrics(events: list[dict]) -> None:
    levels = defaultdict(list)
    for e in events:
        if e.get("event") == "eval":
            levels[e.get("level","unk")].append(e.get("score", 0.0))
    if not levels:
        st.write("No evaluations recorded yet.")
        return
    import pandas as pd
    rows = []
    for lv, scores in levels.items():
        if not scores: continue
        rows.append({"level": lv, "n": len(scores),
                     "mean": sum(scores)/len(scores),
                     "min": min(scores), "max": max(scores)})
    st.subheader("Metrics summary")
    st.dataframe(pd.DataFrame(rows))
    st.subheader("Per-task scores")
    long_rows = []
    for lv, scores in levels.items():
        for i, s in enumerate(scores):
            long_rows.append({"level": lv, "task_idx": i, "score": s})
    if long_rows:
        df = pd.DataFrame(long_rows)
        st.line_chart(df, x="task_idx", y="score", color="level")


def render_recent_events(events: list[dict], n: int = 60) -> None:
    st.subheader(f"Most recent {n} events")
    for e in events[-n:]:
        st.text(f"{e.get('ts','')[11:19]}  {e.get('event','-'):<22} {json.dumps({k:v for k,v in e.items() if k not in ('ts','event')}, default=str)[:240]}")


def render_library() -> None:
    st.subheader("Persisted composites")
    cp = TOOL_LIB / "composites.json"
    if not cp.exists():
        st.info("No composites yet.")
        return
    try:
        composites = json.loads(cp.read_text(encoding="utf-8"))
    except Exception:
        composites = []
    if not composites:
        st.info("No composites yet.")
        return
    import pandas as pd
    rows = []
    for c in composites:
        latest = (c.get("metrics_history") or [{}])[-1]
        rows.append({
            "id": c.get("id",""), "domain": c.get("domain",""),
            "subdomain": c.get("subdomain",""), "capability": c.get("capability_tag",""),
            "n_steps": len(c.get("steps") or []),
            "score": latest.get("score", 0.0),
            "lineage_depth": c.get("lineage_depth", 0),
        })
    st.dataframe(pd.DataFrame(rows))
    dot = TOOL_LIB / "lineage.dot"
    if dot.exists():
        with st.expander("Lineage diagram (Graphviz DOT)"):
            st.graphviz_chart(dot.read_text(encoding="utf-8"))


def render_brief(session_dir: Path) -> None:
    p = session_dir / "domain_brief.md"
    if not p.exists(): return
    with st.expander("Domain brief"):
        st.markdown(p.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(page_title="Stem Agent", layout="wide")
    st.title("Stem Agent — dashboard")
    st.caption("Read-only. Console output remains the primary view. Refreshes every 2 seconds.")
    st.sidebar.header("Navigation")
    sd = session_picker()
    if sd is None: return
    st.sidebar.write(f"Session: `{sd.name}`")

    tab_metrics, tab_events, tab_library, tab_files = st.tabs(["Metrics","Events","Library","Files"])
    with tab_metrics:
        events = load_events(sd)
        render_metrics(events)
        render_brief(sd)
    with tab_events:
        events = load_events(sd)
        render_recent_events(events, n=80)
    with tab_library:
        render_library()
    with tab_files:
        st.write([str(p.relative_to(sd)) for p in sd.rglob("*") if p.is_file()][:80])


if __name__ == "__main__":
    main()
