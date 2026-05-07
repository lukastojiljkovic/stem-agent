"""Walk runs/* and aggregate metrics across all sessions ever recorded.

Output:
    - runs/_aggregate/<timestamp>/trajectory.csv
    - runs/_aggregate/<timestamp>/trajectory.pdf  (matplotlib line plot)
    - prints a markdown table to stdout

Usage:
    python scripts/aggregate_runs.py [--domain legal|economics|all]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"


def _scan() -> list[dict]:
    if not RUNS.exists():
        return []
    rows = []
    for d in sorted(p for p in RUNS.iterdir()
                    if p.is_dir() and not p.name.startswith("_")):
        m_path = d / "metrics.json"
        if not m_path.exists():
            continue
        try:
            m = json.loads(m_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        ev = d / "event_log.jsonl"
        domain = "unknown"
        if ev.exists():
            for line in ev.read_text(encoding="utf-8").splitlines():
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("event") == "session.start":
                    domain = e.get("domain", "unknown")
                    break

        def mn(rs):
            if not rs: return 0.0
            return sum(r.get("score", 0.0) for r in rs) / len(rs)
        rows.append({
            "session_id": d.name,
            "domain": domain,
            "ts": d.name.split("_")[0],
            "n_baseline": len(m.get("baseline", [])),
            "baseline_mean": mn(m.get("baseline", [])),
            "L1_mean": mn(m.get("L1", [])),
            "L2_mean": mn(m.get("L2", [])),
        })
    return rows


def _print_markdown(rows: list[dict]) -> None:
    print(f"| session_id            | domain    | n  | baseline | L1     | L2     |")
    print(f"| --------------------- | --------- | -- | -------- | ------ | ------ |")
    for r in rows:
        print(f"| {r['session_id']:<21} | {r['domain']:<9} | {r['n_baseline']:<2} "
              f"| {r['baseline_mean']:.3f}    | {r['L1_mean']:.3f}  | {r['L2_mean']:.3f}  |")


def _write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)


def _plot(rows: list[dict], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    by_domain: dict[str, list[dict]] = {}
    for r in rows:
        by_domain.setdefault(r["domain"], []).append(r)
    fig, axes = plt.subplots(1, max(1, len(by_domain)), figsize=(5*max(1,len(by_domain)), 3.5),
                             squeeze=False)
    for ax, (domain, rs) in zip(axes[0], by_domain.items()):
        xs = list(range(1, len(rs) + 1))
        for series, color in [("baseline_mean","#888"), ("L1_mean","#4486bb"), ("L2_mean","#27a36b")]:
            ys = [r[series] for r in rs]
            label = series.replace("_mean", "")
            ax.plot(xs, ys, marker="o", label=label, color=color, linewidth=2)
        ax.set_xticks(xs)
        ax.set_xlabel(f"{domain} session #"); ax.set_ylabel("mean score")
        ax.set_title(f"{domain}  (n={len(rs)})"); ax.legend(loc="best")
        ax.grid(True, linestyle=":", alpha=0.4); ax.set_ylim(0, 1.0)
    fig.suptitle("Cross-session trajectory across all recorded runs")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="all", choices=["all", "legal", "economics"])
    args = ap.parse_args(argv)

    rows = _scan()
    if args.domain != "all":
        rows = [r for r in rows if r["domain"] == args.domain]
    if not rows:
        print("No sessions found under runs/. Run stem-agent first.")
        return 1

    out_root = REPO / "runs" / "_aggregate" / datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    csv_path = out_root / "trajectory.csv"
    plot_path = out_root / "trajectory.pdf"
    _write_csv(rows, csv_path)
    _plot(rows, plot_path)
    _print_markdown(rows)
    print(f"\nwrote {csv_path}\nwrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
