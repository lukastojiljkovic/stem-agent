"""Deterministic scorers."""
from __future__ import annotations

import re
from typing import Any


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").lower()).strip()


def score_clauses_f1(predicted: Any,
                     gold: dict[str, list[str]]) -> float:
    """gold = {category: [acceptable_text_substrings]}.
    Predicted = {clauses: [{category, text}]} or list[{category, text}] or
    free text / list of strings (in which case we score against per-category gold spans via token overlap).
    Macro-F1 across categories present in gold."""
    clauses: list[dict[str, Any]] = []
    free_text: str | None = None
    if isinstance(predicted, dict) and "clauses" in predicted:
        clauses = [c for c in (predicted.get("clauses") or []) if isinstance(c, dict)]
    elif isinstance(predicted, list):
        clauses = [c for c in predicted if isinstance(c, dict) and "category" in c]
        if not clauses:
            free_text = " ".join(str(x) for x in predicted)
    elif isinstance(predicted, str):
        free_text = predicted
    elif predicted is not None:
        free_text = str(predicted)

    if free_text is not None:
        out_tokens = set(_norm(free_text).split())
        f1s: list[float] = []
        for cat, accept_texts in gold.items():
            best = 0.0
            for at in accept_texts:
                gt = set(_norm(at).split())
                if not out_tokens or not gt:
                    continue
                tp = len(out_tokens & gt)
                if tp == 0: continue
                prec = tp / len(out_tokens); rec = tp / len(gt)
                f1 = 2 * prec * rec / (prec + rec)
                if f1 > best: best = f1
            f1s.append(best)
        return sum(f1s) / len(f1s) if f1s else 0.0
    pred_by_cat: dict[str, str] = {}
    for c in clauses:
        cat = c.get("category","")
        txt = c.get("text","")
        if cat and txt:
            pred_by_cat.setdefault(cat, txt)

    f1s: list[float] = []
    for cat, accept_texts in gold.items():
        if cat not in pred_by_cat:
            f1s.append(0.0); continue
        ptokens = set(_norm(pred_by_cat[cat]).split())
        best = 0.0
        for at in accept_texts:
            gt = set(_norm(at).split())
            if not ptokens or not gt:
                continue
            tp = len(ptokens & gt)
            if tp == 0: continue
            prec = tp / len(ptokens); rec = tp / len(gt)
            f1 = 2*prec*rec/(prec+rec)
            if f1 > best: best = f1
        f1s.append(best)
    return sum(f1s)/len(f1s) if f1s else 0.0


def score_obligations_overlap(predicted: list[dict[str, Any]],
                              gold: list[dict[str, Any]]) -> float:
    if not gold: return 1.0 if not predicted else 0.0
    if not predicted: return 0.0
    used = set()
    f1s: list[float] = []
    for g in gold:
        gtoks = set(_norm(f"{g.get('party','')} {g.get('obligation','')}").split())
        best = 0.0; best_idx = -1
        for i, p in enumerate(predicted):
            if i in used: continue
            ptoks = set(_norm(f"{p.get('party','')} {p.get('obligation','')}").split())
            if not ptoks or not gtoks: continue
            tp = len(ptoks & gtoks)
            if tp == 0: continue
            prec = tp/len(ptoks); rec = tp/len(gtoks)
            f1 = 2*prec*rec/(prec+rec)
            if f1 > best: best = f1; best_idx = i
        if best_idx >= 0: used.add(best_idx)
        f1s.append(best)
    return sum(f1s)/len(f1s)


def score_ratios_within_tolerance(predicted: dict[str, Any],
                                  gold: dict[str, float],
                                  tol: float = 0.10) -> float:
    if not gold: return 1.0
    p = predicted.get("ratios") if isinstance(predicted, dict) and "ratios" in predicted else predicted
    p = p or {}
    hits = 0; total = 0
    for k, gv in gold.items():
        total += 1
        pv = p.get(k)
        if pv is None or gv is None:
            continue
        try:
            if abs(float(pv) - float(gv)) <= tol * max(1e-9, abs(float(gv))):
                hits += 1
        except Exception:
            pass
    return hits/total if total else 0.0


def score_classification_accuracy(predicted: str, gold: str) -> float:
    return 1.0 if _norm(predicted) == _norm(gold) else 0.0


def score_qa_answer_match(predicted: str, gold: str | list[str]) -> float:
    refs = gold if isinstance(gold, list) else [gold]
    p = set(_norm(predicted).split()); best = 0.0
    for r in refs:
        rt = set(_norm(r).split())
        if not p or not rt: continue
        tp = len(p & rt)
        if tp == 0: continue
        prec = tp/len(p); rec = tp/len(rt)
        f1 = 2*prec*rec/(prec+rec)
        if f1 > best: best = f1
    return best
