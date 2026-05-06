"""Run N legal sessions back to back; capture the F1 trajectory across runs.

The library is persisted between sessions so each run starts with the
graduated composites of the previous one.

Output: writes `runs/_xsession/<timestamp>/trajectory.json` and `trajectory.pdf`.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _latest_session() -> Path:
    runs = REPO / "runs"
    candidates = [p for p in runs.iterdir() if p.is_dir() and not p.name.startswith("_")]
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def _summary(metrics: list[dict]) -> float:
    if not metrics: return 0.0
    return sum(m.get("score",0.0) for m in metrics)/len(metrics)


def main(n: int = 3) -> int:
    out_root = REPO / "runs" / "_xsession" / datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    trajectory = []
    for i in range(1, n+1):
        print(f"=== cross-session run {i}/{n} ===")
        rc = subprocess.call([sys.executable, "-m", "stem_agent", "run",
                              "--domain", "legal", "--subdomain", "contract_analysis", "--track", "deep"])
        if rc != 0:
            print(f"run {i} failed; stopping.")
            break
        latest = _latest_session()
        m = json.loads((latest / "metrics.json").read_text(encoding="utf-8"))
        trajectory.append({
            "run": i, "session_id": latest.name,
            "baseline": _summary(m.get("baseline", [])),
            "L1": _summary(m.get("L1", [])),
            "L2": _summary(m.get("L2", [])),
        })

    (out_root / "trajectory.json").write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    print(f"trajectory: {out_root/'trajectory.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [t["run"] for t in trajectory]
        for series in ("baseline","L1","L2"):
            plt.plot(xs, [t[series] for t in trajectory], marker="o", label=series)
        plt.xlabel("session"); plt.ylabel("mean score"); plt.title("Cross-session trajectory")
        plt.legend(); plt.grid(True, linestyle=":"); plt.tight_layout()
        plt.savefig(out_root / "trajectory.pdf"); plt.close()
        print(f"plot: {out_root/'trajectory.pdf'}")
    except Exception as e:
        print(f"plot failed: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(3))
