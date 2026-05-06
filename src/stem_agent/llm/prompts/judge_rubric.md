Score the candidate output on a 0..1 scale, considering all of:

- FACTUAL CORRECTNESS: Does the output match the reference / ground truth where available? Numeric answers must be within tolerance; structured outputs must align with the gold schema.
- COMPLETENESS: Does the output cover everything the question asks for, no major omissions?
- CONSISTENCY: Are claims internally consistent? Are cited numbers and entities the same throughout?
- DOMAIN APPROPRIATENESS: Does the output use domain-correct vocabulary and structure (e.g., legal clause names from the taxonomy, financial-ratio definitions)?
- READABILITY: Can a human reader understand it without re-reading?

Penalize fabrications heavily. Penalize verbose padding moderately. Reward terse, evidence-anchored, structured outputs.

Output JSON only:
{"score": <float 0..1>, "rationale": "<2 sentences>"}
