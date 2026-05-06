"""One-shot fixture builder.

Run: `python scripts/fetch_fixtures.py`

Outputs:
  - data/fixtures/contracts/<id>.txt    (5 small CUAD contracts as plaintext)
  - data/fixtures/tasks_legal.jsonl
  - data/fixtures/tasks_economics.jsonl

If HF datasets / EDGAR are unavailable, falls back to small hand-coded
synthetic fixtures so the system can still demonstrate end-to-end behavior.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

FIX = REPO / "data" / "fixtures"
(FIX / "contracts").mkdir(parents=True, exist_ok=True)
(FIX / "filings").mkdir(parents=True, exist_ok=True)


LEGAL_TASKS: list[dict] = []


def _build_cuad_subset(n: int = 5) -> None:
    try:
        from datasets import load_dataset
    except Exception:
        return
    try:
        ds = load_dataset("theatticusproject/cuad-qa", split="train")
    except Exception:
        return
    seen_contracts: dict[str, dict] = {}
    for row in ds:
        ctx = row.get("context","")
        title = row.get("title") or row.get("id","contract").split("__")[0]
        if title not in seen_contracts:
            seen_contracts[title] = {"text": ctx, "qas": []}
        seen_contracts[title]["qas"].append({
            "question": row.get("question",""),
            "answers": row.get("answers", {}).get("text", []),
        })
        if len(seen_contracts) >= n + 5: break

    target_categories = ["Governing Law","Termination for Convenience","Cap on Liability","Anti-Assignment"]
    for tid, data in list(seen_contracts.items())[:n]:
        cid = _slug(tid)
        (FIX / "contracts" / f"{cid}.txt").write_text(data["text"][:80000], encoding="utf-8")
        gold: dict[str, list[str]] = {}
        for qa in data["qas"]:
            q = qa["question"]
            for cat in target_categories:
                if cat.lower() in q.lower() and qa["answers"]:
                    gold.setdefault(cat, []).extend(qa["answers"][:2])
        gold = {k: v for k, v in gold.items() if v}
        if not gold: continue
        LEGAL_TASKS.append({
            "kind": "cuad_clause", "id": f"cuad_{cid}",
            "contract": f"{cid}.txt",
            "question": "Extract verbatim Governing Law, Termination for Convenience, Cap on Liability, Anti-Assignment clauses.",
            "gold": gold,
        })


def _hand_legal_fixtures() -> None:
    c1 = ("Master Services Agreement between Acme Co. and Beta Ltd. "
          "This Agreement shall be governed by the laws of the State of Delaware. "
          "Either party may terminate for convenience on 30 days written notice. "
          "Total liability of either party shall not exceed the fees paid in the prior twelve months. "
          "This Agreement may not be assigned by either party without the prior written consent of the other.")
    c2 = ("Software License Agreement. Governed by the laws of England and Wales. "
          "Customer may terminate this agreement at any time without cause upon 60 days notice. "
          "In no event shall Vendor's aggregate liability exceed $1,000,000. "
          "Customer shall not assign or transfer this agreement without Vendor's prior written consent.")
    (FIX / "contracts" / "synth_msa_1.txt").write_text(c1, encoding="utf-8")
    (FIX / "contracts" / "synth_swl_2.txt").write_text(c2, encoding="utf-8")
    LEGAL_TASKS.extend([
        {"kind":"cuad_clause","id":"synth_msa_1","contract":"synth_msa_1.txt",
         "question":"Extract verbatim Governing Law, Termination for Convenience, Cap on Liability, Anti-Assignment clauses.",
         "gold":{
            "Governing Law":["governed by the laws of the State of Delaware"],
            "Termination for Convenience":["terminate for convenience on 30 days written notice"],
            "Cap on Liability":["total liability of either party shall not exceed the fees paid in the prior twelve months"],
            "Anti-Assignment":["may not be assigned by either party without the prior written consent of the other"],
         }},
        {"kind":"cuad_clause","id":"synth_swl_2","contract":"synth_swl_2.txt",
         "question":"Extract verbatim Governing Law, Termination for Convenience, Cap on Liability, Anti-Assignment clauses.",
         "gold":{
            "Governing Law":["laws of England and Wales"],
            "Termination for Convenience":["terminate this agreement at any time without cause upon 60 days notice"],
            "Cap on Liability":["aggregate liability exceed $1,000,000"],
            "Anti-Assignment":["shall not assign or transfer this agreement without Vendor's prior written consent"],
         }},
    ])


ECON_TASKS: list[dict] = []


def _build_econ_fixtures() -> None:
    ECON_TASKS.append({
        "kind":"ratios","id":"ratios_aapl_fy2024","ticker":"AAPL","form":"10-K","year":2024,
        "gold_ratios":{
            "current_ratio": 0.87,
            "debt_equity": 1.87,
            "roa": 0.27,
            "roe": 1.65,
            "operating_margin": 0.31,
        },
    })
    ECON_TASKS.append({
        "kind":"ratios","id":"ratios_msft_fy2024","ticker":"MSFT","form":"10-K","year":2024,
        "gold_ratios":{
            "current_ratio": 1.27,
            "debt_equity": 0.30,
            "roa": 0.17,
            "roe": 0.37,
            "operating_margin": 0.45,
        },
    })
    ECON_TASKS.append({
        "kind":"fin_qa","id":"finqa_aapl_revenue_fy2024","ticker":"AAPL","form":"10-K","year":2024,
        "question":"What was Apple's total net sales (revenue) for fiscal year 2024?",
        "gold_answer":"approximately $391 billion",
    })
    ECON_TASKS.append({
        "kind":"fin_qa","id":"finqa_msft_revenue_fy2024","ticker":"MSFT","form":"10-K","year":2024,
        "question":"What was Microsoft's total revenue for fiscal year 2024?",
        "gold_answer":"approximately $245 billion",
    })


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+","_", s)[:48].strip("_").lower() or "x"


def main() -> int:
    print(f"Repo: {REPO}")
    _build_cuad_subset(n=5)
    if not LEGAL_TASKS:
        print("CUAD HF dataset unavailable; using hand-coded synthetic fixtures.")
        _hand_legal_fixtures()
    _build_econ_fixtures()
    (FIX / "tasks_legal.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in LEGAL_TASKS) + "\n",
        encoding="utf-8",
    )
    (FIX / "tasks_economics.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in ECON_TASKS) + "\n",
        encoding="utf-8",
    )
    print(f"legal tasks: {len(LEGAL_TASKS)}; econ tasks: {len(ECON_TASKS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
