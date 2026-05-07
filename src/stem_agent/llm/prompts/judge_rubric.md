Score the candidate output by independently rating five criteria on a 0..3 integer scale, then deriving the final 0..1 score.

For each of the five criteria, output an integer in {0, 1, 2, 3}:

- **factual** — does the output match the reference / ground truth where available? Numeric answers within tolerance; structured outputs aligned to the gold schema. (0 = wrong/fabricated, 1 = partially right, 2 = mostly right, 3 = exactly right or N/A when no ground truth applies)
- **completeness** — does the output cover what was asked, with no major omissions? (0 = empty/off-topic, 1 = partial, 2 = covers most, 3 = covers all)
- **consistency** — are claims internally consistent? cited numbers and entities the same throughout? (0 = self-contradicting, 1 = mixed signals, 2 = mostly consistent, 3 = fully consistent)
- **domain** — does the output use domain-correct vocabulary and structure (legal clause names, financial ratio definitions, statute citations)? (0 = wrong domain, 1 = generic, 2 = mostly correct, 3 = expert-level)
- **readability** — clear and skimmable without re-reading? (0 = incomprehensible, 1 = wall of text, 2 = readable, 3 = crisp)

Penalize fabrications heavily (factual=0). Penalize verbose padding by lowering completeness or readability. Reward terse, evidence-anchored, structured outputs.

The final 0..1 score is the simple average of the five criteria divided by 3 (so 5×3=15 maps to 1.0).

Output JSON only:

```json
{
  "factual": 0|1|2|3,
  "completeness": 0|1|2|3,
  "consistency": 0|1|2|3,
  "domain": 0|1|2|3,
  "readability": 0|1|2|3,
  "rationale": "<one sentence>"
}
```

The framework will compute the final scalar from the five integers; do not output a `score` field yourself. Do not output any prose outside the JSON object.
