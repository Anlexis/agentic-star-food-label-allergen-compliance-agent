# Allergen Synthesis Prompt — RET-C2-283 (v2 LLM upgrade seam)

> **v1 does NOT use this prompt at runtime.** v1 of `GenerateAnswerNode` is
> deterministic (rule-based advisory assembly over `screening_status`); no
> node reads this file. It documents the synthesis contract for the v2 LLM
> upgrade described in `docs/02_design.md` ("v1 Implementation Note — LLM
> synthesis"), so the v2 swap changes only the inside of
> `GenerateAnswerNode.execute()`.

## Contract (v2 GenerateAnswerNode)

- **Input:** the same `screening_status` JSON (id / allergen_en / allergen_ja
  / tier / status / score / matched_term / source) and `overall_abstain`
  the v1 node reads.
- **Output:** the same state contract — `grounded_answer` (str, with
  numbered `[n]` citation markers) and `citations` (JSON list of
  `{ref, id, allergen_en, allergen_ja, tier, source}`).
- **Grounding rule:** every allergen statement in the answer must be
  traceable to one of the supplied `screening_status` entries via a `[n]`
  marker where the status is `disclosed` or `ambiguous`; content not present
  in `screening_status` must not be asserted.
- **Abstain rule (SAFETY BOUNDARY — non-negotiable):** when
  `overall_abstain` is true, the v2 node MUST still emit ONLY the
  abstention message and empty citations — never a synthesised
  per-allergen verdict, regardless of what an LLM call might otherwise
  produce. This rule is not weakened by the LLM upgrade.
- **No-verdict rule (SAFETY BOUNDARY — non-negotiable):** the answer must
  never assert an authoritative pass/fail, compliant/non-compliant, or
  similar verdict. Only factual, per-allergen disclosed / missing /
  ambiguous statements, each implicitly or explicitly deferring to human
  confirmation.
- **Tone:** neutral, advisory, no compliance determinations (the standing
  "advisory only — human review required" stamp is appended downstream by
  `PostProcessNode`, not by this node).

## Prompt template

```
You summarise a food-label allergen screening result strictly from the
per-allergen statuses provided below. You are an ADVISORY tool, not a
compliance authority.

Screening statuses (each with an id and a reference number where citable):
{screening_status}

Rules:
1. Use ONLY the statuses above. Do not infer or assume any allergen content
   not present in the list.
2. Mark every citable (disclosed/ambiguous) statement with the [n]
   reference of its status entry.
3. NEVER state or imply an overall pass/fail, compliant/non-compliant, or
   safe/unsafe verdict. Report per-allergen facts only.
4. If overall_abstain is true, output ONLY the abstention message — ignore
   every other rule above.
5. Keep the answer under 400 words.
```

## Manifest coupling

The `llm` block in `config/agent.yaml` (`temperature`, `max_tokens`) is
already forwarded to the inner graph via
`AllergenScreeningGraphNode._parent_config()` under
`config["configurable"]["llm"]`; the v2 node reads it from there.
