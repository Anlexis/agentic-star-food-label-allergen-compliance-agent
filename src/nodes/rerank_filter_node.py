"""AgentCore Platform v1.0"""

# RET-C2-283 - RerankFilterNode
# Domain node 3: turn the raw match candidates into final per-allergen
# screening statuses, and compute the SAFETY-CRITICAL overall_abstain flag.
#
# Status classification (deterministic, threshold-based):
#   score >= score_threshold  -> "disclosed"  (an explicit term matched)
#   0 < score < score_threshold -> "ambiguous" (only an implicit/processed-food
#                                   hint matched - needs human confirmation)
#   score == 0                -> "missing"    (mandatory tier only; recommended
#                                   tier entries at score 0 were already
#                                   dropped by RetrieveNode)
#
# Tier scope: the MANDATORY tier can never be narrowed by the caller - every
# mandatory allergen always gets a status. `include_recommended=False`
# (query_filters) suppresses recommended-tier entries from the output; the
# recommended tier is otherwise capped at `top_k` (mandatory entries are
# never capped/dropped).
#
# SAFETY BOUNDARY (docs/02_design.md): overall_abstain is True when the
# label text is too short/sparse to support a reliable screening
# (len < abstain_min_chars). On abstain, screening_status is emitted EMPTY
# (to_json([])) - this node never lets a low-confidence input reach
# GenerateAnswerNode with a synthesisable per-allergen verdict list. This is
# the mechanical enforcement of "abstain on low-confidence input, never a
# synthesised verdict".
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.schemas.agent_status import AgentStatus
from framework.nodes.function_node import FunctionNode
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json, to_json

# Defaults mirror the `retrieval` block in config/config.yaml.
_DEFAULT_RETRIEVAL: Dict[str, Any] = {
    "score_threshold": 0.75,
    "top_k": 10,
    "abstain_min_chars": 8,
}

_STATUS_DISCLOSED = "disclosed"
_STATUS_AMBIGUOUS = "ambiguous"
_STATUS_MISSING = "missing"


def _resolve_retrieval_config(state: Dict[str, Any]) -> Dict[str, Any]:
    """Effective retrieval config: state retrieval_config > module defaults."""
    effective = dict(_DEFAULT_RETRIEVAL)  # local copy - never mutate the module default
    from_state = from_json(state.get("retrieval_config"), None)
    if isinstance(from_state, dict):
        effective.update(from_state)
    return effective


def _classify(score: float, score_threshold: float) -> str:
    if score >= score_threshold:
        return _STATUS_DISCLOSED
    if score > 0:
        return _STATUS_AMBIGUOUS
    return _STATUS_MISSING


class RerankFilterNode(FunctionNode):
    """Classify match candidates into per-allergen statuses; compute abstain.

    Input state keys:
        matched_provisions: JSON list of match candidates (from RetrieveNode)
        query_filters:      JSON dict with optional include_recommended / top_k
        retrieval_config:   forwarded runtime retrieval block (JSON)
        label_text:         normalised label/ingredient text (abstain check)

    Output state keys (partial dict):
        screening_status: JSON list of final per-allergen statuses
                           (empty when overall_abstain is True)
        overall_abstain:  bool - True when input confidence is too low for a
                           reliable screening (SAFETY BOUNDARY)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        candidates: List[Dict[str, Any]] = from_json(state.get("matched_provisions"), []) or []
        filters = from_json(state.get("query_filters"), {}) or {}
        retrieval_cfg = _resolve_retrieval_config(state)
        text = state.get("label_text") or ""
        text = text if isinstance(text, str) else ""

        try:
            abstain_min_chars = int(retrieval_cfg.get("abstain_min_chars", _DEFAULT_RETRIEVAL["abstain_min_chars"]))
        except (TypeError, ValueError):
            abstain_min_chars = int(_DEFAULT_RETRIEVAL["abstain_min_chars"])
        overall_abstain = len(text.strip()) < max(0, abstain_min_chars)

        if overall_abstain:
            # SAFETY BOUNDARY: never synthesise a per-allergen verdict on
            # low-confidence input - emit no screening_status at all.
            emit_trace_event(
                "rerank_filter_complete",
                {"overall_abstain": True, "candidates_seen": len(candidates)},
                state,
            )
            return {"screening_status": to_json([]), "overall_abstain": True}

        try:
            score_threshold = float(retrieval_cfg.get("score_threshold", _DEFAULT_RETRIEVAL["score_threshold"]))
        except (TypeError, ValueError):
            score_threshold = float(_DEFAULT_RETRIEVAL["score_threshold"])
        score_threshold = max(0.0, min(1.0, score_threshold))

        try:
            top_k = int(retrieval_cfg.get("top_k", _DEFAULT_RETRIEVAL["top_k"]))
        except (TypeError, ValueError):
            top_k = int(_DEFAULT_RETRIEVAL["top_k"])
        top_k = max(1, min(20, top_k))
        caller_top_k = filters.get("top_k")
        if isinstance(caller_top_k, int) and 1 <= caller_top_k < top_k:
            top_k = caller_top_k

        include_recommended = filters.get("include_recommended")
        if include_recommended is None:
            include_recommended = True  # default: surface matched recommended-tier hits

        mandatory_out: List[Dict[str, Any]] = []
        recommended_out: List[Dict[str, Any]] = []
        for entry in candidates:
            if not isinstance(entry, dict):
                continue
            tier = str(entry.get("tier", ""))
            if tier != "mandatory" and not include_recommended:
                continue  # caller narrowed scope - mandatory tier is unaffected
            try:
                score = float(entry.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            status_entry = {
                "id": entry.get("id", ""),
                "allergen_en": entry.get("allergen_en", ""),
                "allergen_ja": entry.get("allergen_ja", ""),
                "tier": tier,
                "status": _classify(score, score_threshold),
                "score": round(score, 4),
                "matched_term": entry.get("matched_term"),
                "source": entry.get("source", ""),
            }
            if tier == "mandatory":
                mandatory_out.append(status_entry)
            else:
                recommended_out.append(status_entry)

        # Mandatory tier is NEVER capped/dropped; recommended tier is capped
        # at top_k (already score-desc ordered by RetrieveNode).
        screening_status = mandatory_out + recommended_out[:top_k]

        # Domain audit: statuses classified, abstain gate evaluated clear.
        emit_trace_event(
            "rerank_filter_complete",
            {
                "overall_abstain": False,
                "mandatory_count": len(mandatory_out),
                "recommended_count": len(recommended_out[:top_k]),
                "score_threshold": score_threshold,
            },
            state,
        )

        return {"screening_status": to_json(screening_status), "overall_abstain": False}
