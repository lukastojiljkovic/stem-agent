"""CLI entrypoint:

  stem-agent run --domain legal --subdomain contract_analysis [--track deep]
  stem-agent run --domain economics
  stem-agent baseline --domain legal
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="stem-agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run the full L0 -> L1 (-> L2) pipeline + eval.")
    p_run.add_argument("--domain", required=True, choices=["legal","economics"])
    p_run.add_argument("--subdomain", default=None)
    p_run.add_argument("--track", default="deep", choices=["deep","shallow"])
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--no-bootstrap", action="store_true",
                       help="Skip the domain bootstrapping pass (for debugging).")

    p_base = sub.add_parser("baseline", help="Run only the hand-coded baseline pipeline against eval.")
    p_base.add_argument("--domain", required=True, choices=["legal","economics"])

    p_show = sub.add_parser("library", help="Show the persisted tool library summary.")
    p_show.add_argument("--lineage", action="store_true", help="Also write/refresh lineage.dot.")

    args = p.parse_args(argv)

    from .eval.runner import run_full, run_baseline_only, show_library

    if args.cmd == "run":
        return run_full(domain=args.domain, subdomain=args.subdomain,
                        track=args.track, seed=args.seed,
                        do_bootstrap=not args.no_bootstrap)
    if args.cmd == "baseline":
        return run_baseline_only(domain=args.domain)
    if args.cmd == "library":
        return show_library(lineage=args.lineage)
    return 1


if __name__ == "__main__":
    sys.exit(main())
