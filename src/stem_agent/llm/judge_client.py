"""External LLM judge with provider fallback: Anthropic > OpenAI > local Gemma.

Used in PHASE 3 final-eval comparison and at promotion gates where a more
credible reading than self-judging is desirable. Position bias mitigated
by `score_pairwise()` which always evaluates both orderings and averages.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..config import JUDGE, has_anthropic_key, has_openai_key
from .lm_client import LMClient, ChatMessage


@dataclass
class JudgeScore:
    score: float
    rationale: str
    provider: str


def resolve_provider() -> str:
    if has_anthropic_key():
        return "anthropic"
    if has_openai_key():
        return "openai"
    return "local"


def format_rubric_prompt(question: str, answer: str, rubric: str, reference: str | None = None) -> str:
    parts = [
        "You are an evaluator. Score the candidate ANSWER on a 0..1 scale.",
        f"RUBRIC:\n{rubric}",
        f"\nQUESTION:\n{question}",
    ]
    if reference is not None:
        parts.append(f"\nREFERENCE_ANSWER (gold):\n{reference}")
    parts.append(f"\nCANDIDATE ANSWER:\n{answer}")
    parts.append(
        "\nReturn ONLY a JSON object: "
        '{"score": <float 0..1>, "rationale": "<one short paragraph>"}'
    )
    return "\n".join(parts)


def format_pairwise_prompt(question: str, a: str, b: str, rubric: str) -> str:
    return (
        "You are evaluating two candidate answers (ANSWER_A and ANSWER_B) "
        f"to the same QUESTION using this RUBRIC:\n{rubric}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER_A:\n{a}\n\n"
        f"ANSWER_B:\n{b}\n\n"
        "Return ONLY a JSON object: "
        '{"winner": "A" | "B" | "tie", "rationale": "<short>"}'
    )


class JudgeClient:
    def __init__(self, provider: str | None = None, local_lm: LMClient | None = None):
        self.provider = provider or resolve_provider()
        self._local = local_lm

    def score_with_rubric(
        self,
        question: str,
        answer: str,
        rubric: str,
        reference: str | None = None,
    ) -> JudgeScore:
        prompt = format_rubric_prompt(question, answer, rubric, reference)
        text = self._call(prompt)
        score, rationale = self._parse_score(text)
        return JudgeScore(score=score, rationale=rationale, provider=self.provider)

    def score_pairwise(
        self,
        question: str,
        candidate_a: str,
        candidate_b: str,
        rubric: str,
    ) -> dict[str, Any]:
        ab = self._call(format_pairwise_prompt(question, candidate_a, candidate_b, rubric))
        ba = self._call(format_pairwise_prompt(question, candidate_b, candidate_a, rubric))
        win_ab, _ = self._parse_winner(ab)
        win_ba, _ = self._parse_winner(ba)

        a_score = (1 if win_ab == "A" else 0 if win_ab == "B" else 0.5) \
                  + (1 if win_ba == "B" else 0 if win_ba == "A" else 0.5)
        a_score /= 2.0
        return {"a_win_rate": a_score, "raw_ab": ab, "raw_ba": ba, "provider": self.provider}

    def _call(self, prompt: str) -> str:
        if self.provider == "anthropic":
            return self._call_anthropic(prompt)
        if self.provider == "openai":
            return self._call_openai(prompt)
        return self._call_local(prompt)

    def _call_anthropic(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=JUDGE.anthropic_model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
            timeout=JUDGE.request_timeout_s,
        )
        return "".join(b.text for b in msg.content if getattr(b, "text", None))

    def _call_openai(self, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(timeout=JUDGE.request_timeout_s)
        resp = client.chat.completions.create(
            model=JUDGE.openai_model,
            max_tokens=512,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    def _call_local(self, prompt: str) -> str:
        if self._local is None:
            self._local = LMClient()
        out = self._local.chat(
            [ChatMessage(role="system", content="You are a careful evaluator. Output only valid JSON."),
             ChatMessage(role="user", content=prompt)],
            temperature=0.2, top_p=0.9, max_tokens=512,
        )
        return out.text

    @staticmethod
    def _parse_score(text: str) -> tuple[float, str]:
        """Accept either:
          (a) the new multi-criterion shape {factual, completeness, consistency,
              domain, readability, rationale} with each rated 0..3 — final
              scalar is the mean of the five divided by 3;
          (b) the legacy {score, rationale} shape with a precomputed 0..1 float.
        Falls back to 0.0 if neither parses."""
        import json, re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return 0.0, f"unparseable: {text[:120]}"
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return 0.0, f"unparseable: {text[:120]}"
        rationale = str(obj.get("rationale", ""))
        criteria = ("factual", "completeness", "consistency", "domain", "readability")
        if all(k in obj for k in criteria):
            try:
                vals = [int(obj[k]) for k in criteria]
                vals = [max(0, min(3, v)) for v in vals]
                return sum(vals) / (3.0 * len(vals)), rationale
            except Exception:
                pass
        if "score" in obj:
            try:
                return max(0.0, min(1.0, float(obj["score"]))), rationale
            except Exception:
                pass
        return 0.0, f"unparseable: {text[:120]}"

    @staticmethod
    def _parse_winner(text: str) -> tuple[str, str]:
        import json, re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return "tie", text[:120]
        try:
            obj = json.loads(m.group(0))
            w = str(obj.get("winner", "tie")).upper()
            if w not in ("A", "B"):
                w = "TIE"
            return ("A" if w == "A" else "B" if w == "B" else "tie"), str(obj.get("rationale", ""))
        except Exception:
            return "tie", text[:120]
