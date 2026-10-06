"""AgentCore Platform v1.0"""

# State must be a flat TypedDict - never Pydantic BaseModel. LangGraph
# checkpoints use msgpack serialization; Pydantic objects cause silent
# corruption.  Extend AgentState with agent-specific fields only.  Do NOT
# add credentials, secrets, or Pydantic models.
#
# Checkpoint (msgpack) safety: structured fields (dict / list[dict]) are
# stored as JSON STRINGS, not bare Python containers. Producers serialize
# with to_json() on write; consumers deserialize with from_json() on read.
#
# RET-C2-283 - Food Label Allergen Compliance Agent (Cat 2 RAG).
# Two-layer nested Cat 2 graph: outer backbone (AgentBaseGraph) + inner
# domain workflow (BaseGraph).  Fields below cover both layers.
#
# PII / confidentiality note: this template screens NORMALIZED label /
# ingredient TEXT only (no OCR, no image bytes ever touch State). The
# framework's default PII scan masks user_input / validated_input /
# llm_response at every node boundary; label text on the structured
# input_context channel gets the equivalent high-precision personal-data
# masking in PreProcessNode (src/nodes/validation.py); the output gate in
# PostProcessNode additionally recurse-scans the final result for
# credential- and PII-shaped content before it leaves the agent.
#
# Safety note: this agent is an ADVISORY screening assistant, not the
# compliance authority. It never issues an authoritative pass/fail verdict
# and abstains (overall_abstain) on low-confidence input rather than
# guessing. See docs/02_design.md "Safety Boundary".

import json
from typing import Any, NotRequired, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a dict/list State field to a JSON string (checkpoint safety).

    None passes through unchanged so an 'unset' field stays distinguishable
    from an empty container.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON-string State field back to its dict/list.

    None / empty / malformed input -> the supplied ``default`` so a missing or
    corrupt field is non-fatal for the consuming node.
    """
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for RET-C2-283.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.
    Domain fields are NotRequired so the TypedDict is valid at graph
    initialisation, before any node has written a value.
    """

    # ------------------------------------------------------------------
    # Outer layer - set by PreProcessNode / AllergenScreeningGraphNode.merge_output
    # ------------------------------------------------------------------

    # Screened caller text produced by PreProcessNode. Raw input is NOT
    # persisted beyond it.
    validated_input: NotRequired[str]

    # JSON STRING (json.dumps) of the screened caller label request from
    # input_context["label_request"]: {"label_text": str,
    # "include_recommended": bool | None, "top_k": int | None}. Written by
    # PreProcessNode after field-by-field screening; carried into the inner
    # graph via the ContextVar bridge (src/graph/context_bridge.py) and
    # consumed by InputValidateNode. Absent on the legacy plain-text path.
    screening_request: NotRequired[Optional[str]]

    # JSON STRING (to_json) of channel metadata written by PreProcessNode:
    # {"source": str, "channel": str}.
    enriched_context: NotRequired[Optional[str]]

    # Final allergen-screening advisory result, mapped from the inner
    # graph's formatted_answer output via merge_output.
    allergen_screening_result: NotRequired[str]

    # ------------------------------------------------------------------
    # Inner layer - domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # InputValidateNode output
    # Normalised label/ingredient text (whitespace-collapsed, length-capped).
    # NOT an image, NOT OCR output - v1 accepts normalized text only.
    label_text: NotRequired[str]

    # JSON STRING (to_json) of parsed structured request params. Deserialised
    # dict shape: {"include_recommended": bool, "top_k": int | None}.
    # Consumers (RerankFilterNode) read it back via from_json(). Mandatory-tier
    # allergens are ALWAYS screened regardless of this filter - only the
    # recommended-tier surfacing can be narrowed by the caller.
    query_filters: NotRequired[Optional[str]]

    # Runtime `retrieval` block (config/config.yaml) forwarded by
    # AllergenScreeningGraphNode._parent_config() ->
    # DomainWorkflowGraph._extra_initial_state().
    # JSON STRING (to_json) of {"score_threshold": float, "top_k": int,
    # "kb_path": str, "abstain_min_chars": int}. Consumers (RetrieveNode,
    # RerankFilterNode) read it back via from_json().
    retrieval_config: NotRequired[Optional[str]]

    # RetrieveNode output
    # JSON STRING (to_json) of per-allergen match candidates. Deserialised
    # shape: list[dict], each entry {"id": str, "allergen_en": str,
    # "allergen_ja": str, "tier": "mandatory"|"recommended", "source": str,
    # "score": float, "matched_term": str | None, "excerpt": str}.
    # Mandatory-tier entries are ALWAYS present (even at score 0.0);
    # recommended-tier entries are present only when score > 0.
    # Consumers (RerankFilterNode) read it back via from_json().
    matched_provisions: NotRequired[Optional[str]]

    # RerankFilterNode output
    # JSON STRING (to_json) of the final per-allergen screening statuses.
    # Deserialised shape: list[dict], each entry {"id": str,
    # "allergen_en": str, "allergen_ja": str, "tier": str,
    # "status": "disclosed"|"missing"|"ambiguous", "score": float,
    # "matched_term": str | None, "source": str}.
    # Consumers (GenerateAnswerNode) read it back via from_json(). Empty
    # (to_json([])) when overall_abstain is True - abstention emits NO
    # synthesised per-allergen verdict.
    screening_status: NotRequired[Optional[str]]

    # RerankFilterNode output (SAFETY BOUNDARY field)
    # True when the input text is too short / sparse to support a reliable
    # screening (see docs/02_design.md "Safety Boundary"). When True,
    # GenerateAnswerNode emits an abstention message instead of a verdict -
    # this agent must NEVER assert a compliance determination on low-
    # confidence input.
    overall_abstain: NotRequired[bool]

    # GenerateAnswerNode outputs
    # Rule-assembled advisory body grouped by tier, with numbered citation
    # markers. Never an authoritative pass/fail verdict.
    grounded_answer: NotRequired[str]

    # JSON STRING (to_json) of citations. Deserialised shape: list[dict],
    # each entry {"ref": int, "id": str, "allergen_en": str,
    # "allergen_ja": str, "tier": str, "source": str}.
    # Consumers (OutputFormatNode) read it back via from_json().
    citations: NotRequired[Optional[str]]

    # OutputFormatNode output
    # Final formatted advisory body (grouped statuses + sources). The
    # standing "advisory only - human review required" stamp is appended
    # downstream by PostProcessNode (the output gate), NOT here - see
    # docs/02_design.md.
    formatted_answer: NotRequired[str]

    # Validation / parse notes accumulated during intake (no PII).
    # JSON STRING (to_json) of list[str].
    intake_notes: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Tracing / audit - framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: Optional[str]
    correlation_id: Optional[str]
    error_code: Optional[str]
    # node_history inherited from AgentState; listed here for clarity
    # node_history: Optional[List[str]]
