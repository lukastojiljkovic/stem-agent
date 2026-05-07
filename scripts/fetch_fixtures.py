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
    contracts: dict[str, dict] = {
        "synth_msa_1": {
            "text": ("Master Services Agreement between Acme Co. and Beta Ltd. "
                     "This Agreement shall be governed by the laws of the State of Delaware. "
                     "Either party may terminate for convenience on 30 days written notice. "
                     "Total liability of either party shall not exceed the fees paid in the prior twelve months. "
                     "This Agreement may not be assigned by either party without the prior written consent of the other."),
            "gold": {
                "Governing Law":["governed by the laws of the State of Delaware"],
                "Termination for Convenience":["terminate for convenience on 30 days written notice"],
                "Cap on Liability":["total liability of either party shall not exceed the fees paid in the prior twelve months"],
                "Anti-Assignment":["may not be assigned by either party without the prior written consent of the other"],
            },
        },
        "synth_swl_2": {
            "text": ("Software License Agreement. Governed by the laws of England and Wales. "
                     "Customer may terminate this agreement at any time without cause upon 60 days notice. "
                     "In no event shall Vendor's aggregate liability exceed $1,000,000. "
                     "Customer shall not assign or transfer this agreement without Vendor's prior written consent."),
            "gold": {
                "Governing Law":["laws of England and Wales"],
                "Termination for Convenience":["terminate this agreement at any time without cause upon 60 days notice"],
                "Cap on Liability":["aggregate liability exceed $1,000,000"],
                "Anti-Assignment":["shall not assign or transfer this agreement without Vendor's prior written consent"],
            },
        },
        "synth_nda_3": {
            "text": ("Non-Disclosure Agreement between Gamma Inc. and Delta LLC, effective March 1, 2026. "
                     "This Agreement shall be governed by the laws of New York. "
                     "Either party may terminate this Agreement upon ninety (90) days' written notice without cause. "
                     "In no event shall either party's total liability exceed the aggregate fees paid hereunder during the preceding six months. "
                     "Neither party may assign this Agreement to any third party without the prior written consent of the other."),
            "gold": {
                "Governing Law":["governed by the laws of New York"],
                "Termination for Convenience":["terminate this Agreement upon ninety (90) days' written notice without cause"],
                "Cap on Liability":["total liability exceed the aggregate fees paid hereunder during the preceding six months"],
                "Anti-Assignment":["may not assign this Agreement to any third party without the prior written consent of the other"],
            },
        },
        "synth_supply_4": {
            "text": ("Supply Agreement, effective January 15, 2026, between Epsilon Manufacturing and Zeta Distributors. "
                     "This Agreement shall be governed by and construed in accordance with the laws of the Commonwealth of Massachusetts. "
                     "Buyer may terminate this Agreement for convenience upon forty-five (45) days written notice to Seller. "
                     "Seller's total cumulative liability under this Agreement shall in no event exceed five million U.S. dollars (\\$5,000,000). "
                     "This Agreement may not be assigned, in whole or in part, by either party without the prior written consent of the other party."),
            "gold": {
                "Governing Law":["governed by and construed in accordance with the laws of the Commonwealth of Massachusetts"],
                "Termination for Convenience":["terminate this Agreement for convenience upon forty-five (45) days written notice"],
                "Cap on Liability":["total cumulative liability under this Agreement shall in no event exceed five million U.S. dollars"],
                "Anti-Assignment":["may not be assigned, in whole or in part, by either party without the prior written consent"],
            },
        },
        "synth_consulting_5": {
            "text": ("Consulting Agreement between Eta Advisors and Theta Holdings, effective February 10, 2026. "
                     "This Agreement is governed by the laws of the State of California. "
                     "Either party may terminate this Agreement at any time, with or without cause, by giving fifteen (15) days advance written notice. "
                     "Consultant's aggregate liability shall not in any event exceed the total fees actually paid to Consultant under this Agreement. "
                     "This Agreement and any rights or obligations hereunder may not be assigned by Consultant without Theta's prior written consent."),
            "gold": {
                "Governing Law":["governed by the laws of the State of California"],
                "Termination for Convenience":["terminate this Agreement at any time, with or without cause, by giving fifteen (15) days advance written notice"],
                "Cap on Liability":["aggregate liability shall not in any event exceed the total fees actually paid to Consultant under this Agreement"],
                "Anti-Assignment":["may not be assigned by Consultant without Theta's prior written consent"],
            },
        },
        "synth_distribution_6": {
            "text": ("Distribution Agreement between Iota Trading and Kappa Wholesale, effective April 5, 2026. "
                     "This Agreement is governed by the laws of the Republic of Singapore. "
                     "Distributor may terminate this Agreement at its convenience upon ninety (90) calendar days prior written notice to Supplier. "
                     "Supplier's total liability under this Agreement shall under no circumstances exceed two million euros (\\u20ac2,000,000) in the aggregate. "
                     "Neither party may assign this Agreement or any of its rights or obligations hereunder without the prior written consent of the other, such consent not to be unreasonably withheld."),
            "gold": {
                "Governing Law":["governed by the laws of the Republic of Singapore"],
                "Termination for Convenience":["terminate this Agreement at its convenience upon ninety (90) calendar days prior written notice"],
                "Cap on Liability":["total liability under this Agreement shall under no circumstances exceed two million euros"],
                "Anti-Assignment":["Neither party may assign this Agreement or any of its rights or obligations hereunder without the prior written consent of the other"],
            },
        },
        "synth_employment_7": {
            "text": ("Employment Agreement between Lambda Corp. and the Employee, effective May 12, 2026. "
                     "This Agreement shall be governed exclusively by the laws of the State of Texas. "
                     "Either Lambda Corp. or the Employee may terminate this Agreement for any reason or no reason upon thirty (30) days written notice to the other party. "
                     "In no event shall Lambda Corp.'s liability under this Agreement exceed one year of the Employee's then-current base salary. "
                     "The Employee may not assign this Agreement or any duties hereunder to any other person."),
            "gold": {
                "Governing Law":["governed exclusively by the laws of the State of Texas"],
                "Termination for Convenience":["terminate this Agreement for any reason or no reason upon thirty (30) days written notice"],
                "Cap on Liability":["liability under this Agreement exceed one year of the Employee's then-current base salary"],
                "Anti-Assignment":["may not assign this Agreement or any duties hereunder"],
            },
        },
        "synth_saas_8": {
            "text": ("Software-as-a-Service Subscription Agreement, dated June 1, 2026, between Mu Cloud Services and Customer. "
                     "This Agreement and any disputes arising hereunder shall be governed by and interpreted under the laws of the State of Washington. "
                     "Customer may cancel the subscription at any time without cause, effective at the end of the then-current billing period. "
                     "Provider's maximum aggregate liability arising out of or relating to this Agreement shall not exceed the fees paid by Customer in the trailing twelve (12) months. "
                     "Customer shall not assign, transfer, or sublicense this Agreement, in whole or in part, without the prior written consent of Provider."),
            "gold": {
                "Governing Law":["governed by and interpreted under the laws of the State of Washington"],
                "Termination for Convenience":["cancel the subscription at any time without cause"],
                "Cap on Liability":["maximum aggregate liability arising out of or relating to this Agreement shall not exceed the fees paid by Customer in the trailing twelve (12) months"],
                "Anti-Assignment":["shall not assign, transfer, or sublicense this Agreement, in whole or in part, without the prior written consent of Provider"],
            },
        },
        "synth_partnership_9": {
            "text": ("Limited Partnership Agreement among Nu Capital Partners, Xi Holdings, and Omicron Investments, executed July 20, 2026. "
                     "This Agreement is to be construed in accordance with and governed by the laws of the State of New Jersey. "
                     "Any limited partner may withdraw from the partnership at its convenience upon one hundred eighty (180) days advance written notice to the General Partner. "
                     "The General Partner's aggregate cumulative liability to the limited partners shall not exceed three million U.S. dollars (\\$3,000,000). "
                     "The interests of the limited partners may not be assigned, pledged, or transferred without the General Partner's prior written approval."),
            "gold": {
                "Governing Law":["construed in accordance with and governed by the laws of the State of New Jersey"],
                "Termination for Convenience":["withdraw from the partnership at its convenience upon one hundred eighty (180) days advance written notice"],
                "Cap on Liability":["aggregate cumulative liability to the limited partners shall not exceed three million U.S. dollars"],
                "Anti-Assignment":["may not be assigned, pledged, or transferred without the General Partner's prior written approval"],
            },
        },
        "synth_reseller_10": {
            "text": ("Reseller Agreement between Pi Software Inc. and Rho Resellers Ltd., effective August 8, 2026. "
                     "This Agreement and the rights and obligations of the parties shall be governed by the laws of the Province of Ontario, Canada. "
                     "Reseller may terminate this Agreement for convenience upon sixty (60) days prior written notice to Pi Software Inc. "
                     "In no event shall Pi Software Inc.'s aggregate liability arising under this Agreement exceed five hundred thousand Canadian dollars (CAD \\$500,000). "
                     "This Agreement may not be assigned by Reseller, by operation of law or otherwise, without the express prior written consent of Pi Software Inc."),
            "gold": {
                "Governing Law":["governed by the laws of the Province of Ontario, Canada"],
                "Termination for Convenience":["terminate this Agreement for convenience upon sixty (60) days prior written notice"],
                "Cap on Liability":["aggregate liability arising under this Agreement exceed five hundred thousand Canadian dollars"],
                "Anti-Assignment":["may not be assigned by Reseller, by operation of law or otherwise, without the express prior written consent of Pi Software Inc."],
            },
        },
        "synth_loan_11": {
            "text": ("Term Loan Agreement, dated September 3, 2026, between Sigma Bank N.A. (Lender) and Tau Industries Inc. (Borrower). "
                     "This Agreement shall be governed by and construed under the laws of the State of Illinois without regard to conflict-of-law principles. "
                     "Borrower may prepay and terminate this Agreement at its convenience upon thirty (30) Business Days prior written notice to the Lender. "
                     "The Lender's aggregate liability under or in connection with this Agreement shall not exceed two times (2x) the unpaid principal amount then outstanding. "
                     "No party hereto may assign or delegate any of its rights or obligations under this Agreement without the prior written consent of the other party."),
            "gold": {
                "Governing Law":["governed by and construed under the laws of the State of Illinois"],
                "Termination for Convenience":["prepay and terminate this Agreement at its convenience upon thirty (30) Business Days prior written notice"],
                "Cap on Liability":["aggregate liability under or in connection with this Agreement shall not exceed two times (2x) the unpaid principal amount"],
                "Anti-Assignment":["may not assign or delegate any of its rights or obligations under this Agreement without the prior written consent"],
            },
        },
        "synth_lease_12": {
            "text": ("Commercial Lease Agreement between Upsilon Properties (Landlord) and Phi Retail Group (Tenant), commencing October 1, 2026. "
                     "This Lease shall be governed by the laws of the State of Florida. "
                     "Tenant may terminate this Lease for convenience upon one hundred twenty (120) days prior written notice to Landlord at any time after the first anniversary of the commencement date. "
                     "Landlord's aggregate liability to Tenant under this Lease shall not exceed twelve (12) months of the then-current base rent. "
                     "Tenant shall not assign this Lease or sublet the Premises in whole or in part without the prior written consent of Landlord."),
            "gold": {
                "Governing Law":["governed by the laws of the State of Florida"],
                "Termination for Convenience":["terminate this Lease for convenience upon one hundred twenty (120) days prior written notice"],
                "Cap on Liability":["aggregate liability to Tenant under this Lease shall not exceed twelve (12) months of the then-current base rent"],
                "Anti-Assignment":["shall not assign this Lease or sublet the Premises in whole or in part without the prior written consent"],
            },
        },
        "synth_franchise_13": {
            "text": ("Franchise Agreement between Chi Brands Worldwide LLC and Psi Franchisee Holdings, dated November 15, 2026. "
                     "This Agreement is governed by the laws of the State of Georgia. "
                     "Either party may terminate this Agreement at its convenience by giving the other party one hundred eighty (180) days advance written notice. "
                     "Franchisor's total aggregate liability under this Agreement shall be capped at the cumulative franchise fees paid by Franchisee in the trailing twenty-four (24) months. "
                     "Franchisee shall not assign or transfer any rights under this Agreement to any successor entity without Franchisor's prior written approval."),
            "gold": {
                "Governing Law":["governed by the laws of the State of Georgia"],
                "Termination for Convenience":["terminate this Agreement at its convenience by giving the other party one hundred eighty (180) days advance written notice"],
                "Cap on Liability":["total aggregate liability under this Agreement shall be capped at the cumulative franchise fees paid by Franchisee in the trailing twenty-four (24) months"],
                "Anti-Assignment":["shall not assign or transfer any rights under this Agreement to any successor entity without Franchisor's prior written approval"],
            },
        },
        "synth_research_14": {
            "text": ("Sponsored Research Agreement between Omega University and Alpha BioPharma Corp., effective December 5, 2026. "
                     "This Agreement and any disputes arising under it shall be governed by the laws of the Commonwealth of Pennsylvania. "
                     "Sponsor may terminate this Agreement for convenience upon ninety (90) days written notice to the University. "
                     "University's cumulative liability arising out of or related to this Agreement shall in no event exceed the total funding received from Sponsor under this Agreement. "
                     "This Agreement may not be assigned by either party without the prior written consent of the other, except to a successor entity in the case of merger or acquisition."),
            "gold": {
                "Governing Law":["governed by the laws of the Commonwealth of Pennsylvania"],
                "Termination for Convenience":["terminate this Agreement for convenience upon ninety (90) days written notice"],
                "Cap on Liability":["cumulative liability arising out of or related to this Agreement shall in no event exceed the total funding received from Sponsor"],
                "Anti-Assignment":["may not be assigned by either party without the prior written consent of the other"],
            },
        },
        "synth_advertising_15": {
            "text": ("Advertising Services Agreement between Beta Media Group and Gamma Brands LLC, effective January 20, 2027. "
                     "This Agreement shall be governed exclusively by the laws of the State of New York, USA. "
                     "Client may terminate this Agreement for convenience at any time upon forty-five (45) days prior written notice to Agency. "
                     "Agency's total liability under this Agreement shall not exceed three (3) months of the most recent monthly retainer fees paid by Client. "
                     "Neither party may assign or transfer this Agreement, in whole or in part, without the prior written consent of the other party, which consent shall not be unreasonably withheld."),
            "gold": {
                "Governing Law":["governed exclusively by the laws of the State of New York"],
                "Termination for Convenience":["terminate this Agreement for convenience at any time upon forty-five (45) days prior written notice"],
                "Cap on Liability":["total liability under this Agreement shall not exceed three (3) months of the most recent monthly retainer fees"],
                "Anti-Assignment":["may not assign or transfer this Agreement, in whole or in part, without the prior written consent of the other party"],
            },
        },
        "synth_construction_16": {
            "text": ("Construction Services Agreement between Delta Builders Inc. (Contractor) and Epsilon Developments Ltd. (Owner), dated February 14, 2027. "
                     "This Agreement is to be governed and interpreted under the laws of the Province of Quebec, Canada. "
                     "Owner may terminate this Agreement for convenience by providing Contractor with seventy-five (75) calendar days advance written notice. "
                     "Contractor's aggregate liability under this Agreement shall under no circumstances exceed the total contract value as set forth in Schedule A. "
                     "This Agreement may not be assigned, transferred, or pledged by Contractor without the express prior written consent of Owner."),
            "gold": {
                "Governing Law":["governed and interpreted under the laws of the Province of Quebec, Canada"],
                "Termination for Convenience":["terminate this Agreement for convenience by providing Contractor with seventy-five (75) calendar days advance written notice"],
                "Cap on Liability":["aggregate liability under this Agreement shall under no circumstances exceed the total contract value"],
                "Anti-Assignment":["may not be assigned, transferred, or pledged by Contractor without the express prior written consent"],
            },
        },
        "synth_outsourcing_17": {
            "text": ("Business Process Outsourcing Agreement between Zeta Global Services and Eta Corporate Holdings, executed March 1, 2027. "
                     "This Agreement and the parties' performance hereunder shall be governed by the laws of England and Wales. "
                     "Customer may terminate this Agreement for convenience by giving Provider one hundred (100) days prior written notice without cause. "
                     "Provider's maximum aggregate liability under or in relation to this Agreement shall not exceed one hundred fifty percent (150%) of the fees paid in the preceding twelve months. "
                     "Neither party may assign this Agreement nor any rights or obligations arising under it without the prior written consent of the other, save to an Affiliate."),
            "gold": {
                "Governing Law":["governed by the laws of England and Wales"],
                "Termination for Convenience":["terminate this Agreement for convenience by giving Provider one hundred (100) days prior written notice without cause"],
                "Cap on Liability":["maximum aggregate liability under or in relation to this Agreement shall not exceed one hundred fifty percent (150%) of the fees paid in the preceding twelve months"],
                "Anti-Assignment":["may not assign this Agreement nor any rights or obligations arising under it without the prior written consent"],
            },
        },
        "synth_maintenance_18": {
            "text": ("Equipment Maintenance Agreement between Theta Industrial Services and Iota Manufacturing Corp., effective April 10, 2027. "
                     "This Agreement is governed by the laws of the State of Michigan, USA. "
                     "Customer may terminate this Agreement for convenience upon thirty (30) days written notice to Service Provider, with no early termination fees. "
                     "Service Provider's liability hereunder shall in no event exceed the annual maintenance fees actually paid by Customer in the preceding contract year. "
                     "This Agreement and any rights or duties hereunder may not be assigned by either party without the other party's prior written consent."),
            "gold": {
                "Governing Law":["governed by the laws of the State of Michigan"],
                "Termination for Convenience":["terminate this Agreement for convenience upon thirty (30) days written notice"],
                "Cap on Liability":["liability hereunder shall in no event exceed the annual maintenance fees actually paid by Customer in the preceding contract year"],
                "Anti-Assignment":["may not be assigned by either party without the other party's prior written consent"],
            },
        },
        "synth_data_processing_19": {
            "text": ("Data Processing Agreement between Kappa Cloud Solutions (Processor) and Lambda Health Network (Controller), effective May 5, 2027. "
                     "This DPA shall be governed by the laws of the Republic of Ireland and the General Data Protection Regulation (GDPR) where applicable. "
                     "Either party may terminate this DPA for convenience upon sixty (60) days prior written notice provided the underlying main services agreement permits. "
                     "Processor's aggregate liability arising out of this DPA shall not exceed two times (2x) the annual fees paid by Controller under the main services agreement. "
                     "Processor may not assign this DPA, in whole or in part, including by operation of law, without Controller's prior written consent."),
            "gold": {
                "Governing Law":["governed by the laws of the Republic of Ireland"],
                "Termination for Convenience":["terminate this DPA for convenience upon sixty (60) days prior written notice"],
                "Cap on Liability":["aggregate liability arising out of this DPA shall not exceed two times (2x) the annual fees paid by Controller"],
                "Anti-Assignment":["may not assign this DPA, in whole or in part, including by operation of law, without Controller's prior written consent"],
            },
        },
        "synth_managed_services_20": {
            "text": ("Managed Services Agreement between Mu IT Services Ltd. (MSP) and Nu Banking Group (Client), commencing June 1, 2027. "
                     "This Agreement and the legal relationship hereunder shall be governed by the laws of the Federal Republic of Germany. "
                     "Client may terminate this Agreement for convenience upon ninety (90) calendar days advance written notice to MSP at any time after the initial twelve-month term. "
                     "MSP's total cumulative liability under this Agreement shall not exceed one hundred percent (100%) of the fees paid by Client during the most recent six-month rolling period. "
                     "MSP shall not assign this Agreement or any rights hereunder, including by way of merger, change of control, or operation of law, without Client's prior written consent."),
            "gold": {
                "Governing Law":["governed by the laws of the Federal Republic of Germany"],
                "Termination for Convenience":["terminate this Agreement for convenience upon ninety (90) calendar days advance written notice"],
                "Cap on Liability":["total cumulative liability under this Agreement shall not exceed one hundred percent (100%) of the fees paid by Client during the most recent six-month rolling period"],
                "Anti-Assignment":["shall not assign this Agreement or any rights hereunder, including by way of merger, change of control, or operation of law, without Client's prior written consent"],
            },
        },
    }
    for cid, data in contracts.items():
        (FIX / "contracts" / f"{cid}.txt").write_text(data["text"], encoding="utf-8")
        LEGAL_TASKS.append({
            "kind":"cuad_clause","id":cid,"contract":f"{cid}.txt",
            "question":"Extract verbatim Governing Law, Termination for Convenience, Cap on Liability, Anti-Assignment clauses.",
            "gold": data["gold"],
        })


ECON_TASKS: list[dict] = []
SARA_LIKE_TASKS: list[dict] = []


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


def _build_sara_style_fixtures() -> None:
    """SARA-style statutory-reasoning items with clean Yes/No labels.
    These exercise the legal_qa capability with a cleaner gold-label set than
    LegalBench (whose loader had empty-label items that polluted scoring)."""
    SARA_LIKE_TASKS.extend([
        {"kind":"sara_like","id":"sara_estate_residence_01",
         "question":"Statute: A taxpayer may deduct property taxes paid on a primary residence if and only if the taxpayer was the legal owner during the taxable year. Fact: Alex paid property taxes on a house owned by his sister Brenda; Alex never held title. Did Alex deduct lawfully?",
         "gold_answer":"No"},
        {"kind":"sara_like","id":"sara_capital_gains_02",
         "question":"Statute: Long-term capital gains apply only to assets held more than one year. Fact: Carla bought stock on January 5, 2024 and sold it on December 31, 2024. Are her gains long-term?",
         "gold_answer":"No"},
        {"kind":"sara_like","id":"sara_dependent_03",
         "question":"Statute: A child qualifies as a dependent only if the child has gross income below $5,050 in the taxable year. Fact: David's daughter earned $4,200 from a part-time job and had no other income. Is she a qualifying dependent under this provision?",
         "gold_answer":"Yes"},
        {"kind":"sara_like","id":"sara_charitable_04",
         "question":"Statute: Charitable contributions exceeding 50% of adjusted gross income are not deductible in the current year. Fact: Eva had AGI of $100,000 and donated $40,000 to a qualifying charity. Is the full $40,000 deductible this year?",
         "gold_answer":"Yes"},
        {"kind":"sara_like","id":"sara_business_meal_05",
         "question":"Statute: Business meal deductions are limited to 50% of the actual expense. Fact: Frank spent $200 on a client dinner. May he deduct $200?",
         "gold_answer":"No"},
        {"kind":"sara_like","id":"sara_rental_06",
         "question":"Statute: Rental income is taxable in the year received. Fact: Greta received a $3,600 prepayment in December 2026 covering January through December 2027. Must she report this in 2026?",
         "gold_answer":"Yes"},
        {"kind":"sara_like","id":"sara_marriage_07",
         "question":"Statute: A taxpayer may file as married filing jointly only if married on the last day of the taxable year. Fact: Hank married on January 2, 2027. May he file jointly for tax year 2026?",
         "gold_answer":"No"},
        {"kind":"sara_like","id":"sara_self_employ_08",
         "question":"Statute: Self-employment tax applies when net earnings from self-employment are $400 or more. Fact: Iris had net self-employment earnings of $390 for the year. Does she owe self-employment tax?",
         "gold_answer":"No"},
    ])


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+","_", s)[:48].strip("_").lower() or "x"


def main() -> int:
    print(f"Repo: {REPO}")
    _build_cuad_subset(n=5)
    if not LEGAL_TASKS:
        print("CUAD HF dataset unavailable; using hand-coded synthetic fixtures.")
        _hand_legal_fixtures()
    _build_econ_fixtures()
    _build_sara_style_fixtures()
    # Append SARA-like (statutory reasoning, Yes/No gold) to the legal task list
    # so they're loaded by the legal eval pipeline alongside CUAD.
    LEGAL_TASKS.extend(SARA_LIKE_TASKS)
    (FIX / "tasks_legal.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in LEGAL_TASKS) + "\n",
        encoding="utf-8",
    )
    (FIX / "tasks_economics.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in ECON_TASKS) + "\n",
        encoding="utf-8",
    )
    print(f"legal tasks: {len(LEGAL_TASKS)} ({len(LEGAL_TASKS) - len(SARA_LIKE_TASKS)} CUAD + {len(SARA_LIKE_TASKS)} SARA-like); "
          f"econ tasks: {len(ECON_TASKS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
