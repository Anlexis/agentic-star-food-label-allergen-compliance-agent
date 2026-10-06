"""AgentCore Platform v1.0"""

# RET-C2-283 - GenerateAnswerNode
# Domain node 4: assemble the cited, tier-tagged advisory screening body from
# the final per-allergen statuses.
#
# v1 is DETERMINISTIC (no live LLM call): the body is rule-assembled from
# screening_status only - a grouped, per-allergen listing with numbered
# citation markers. Nothing outside screening_status reaches the answer
# body, so the output is grounded by construction. The LLM synthesis
# upgrade seam is documented in docs/02_design.md ("v1 Implementation Note -
# LLM synthesis") and config/prompts/allergen_synthesis_prompt.md.
#
# SAFETY BOUNDARY: this node NEVER emits an authoritative pass/fail /
# compliant / non-compliant verdict - only factual, per-allergen disclosed /
# missing / ambiguous statements, each pointing back to a human-review
# requirement. On overall_abstain, it emits ONLY an abstention message
# (no per-allergen synthesis at all, no citations) and routes the caller to
# human review - see RerankFilterNode for the abstain gate.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List, Optional

from framework.schemas.agent_status import AgentStatus
from framework.nodes.function_node import FunctionNode
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json, to_json

_ABSTAIN_ANSWER = (
    "The provided label/ingredient text is too short or too sparse to support "
    "a reliable allergen screening. No per-allergen determination has been "
    "made. Please provide the complete ingredient list, or route this item "
    "directly to a qualified human reviewer."
)

_NO_TEXT_TIER_HEADER = {
    "mandatory": "## Mandatory Allergens (特定原材料)",
    "recommended": "## Recommended-Tier Allergens Mentioned (特定原材料に準ずるもの)",
}

_STATUS_LABEL = {
    "disclosed": "DISCLOSED",
    "ambiguous": "AMBIGUOUS",
    "missing": "MISSING",
}


def _format_line(ref: Optional[int], entry: Dict[str, Any]) -> str:
    allergen = f"{entry.get('allergen_ja', '')} ({entry.get('allergen_en', '')})"
    status = _STATUS_LABEL.get(str(entry.get("status", "")), str(entry.get("status", "")).upper())
    matched_term = entry.get("matched_term")
    if entry.get("status") == "disclosed":
        detail = f'matched explicit term "{matched_term}"' if matched_term else "explicit match"
        cite = f" [{ref}]" if ref is not None else ""
        return f"- {allergen} — {status}{cite}: {detail}."
    if entry.get("status") == "ambiguous":
        detail = (
            f'only a processed-food/implicit hint "{matched_term}" was found - '
            "confirm explicit disclosure with the manufacturer"
            if matched_term
            else "a weak signal was found - confirm explicit disclosure"
        )
        cite = f" [{ref}]" if ref is not None else ""
        return f"- {allergen} — {status}{cite}: {detail}."
    return f"- {allergen} — {status}: no disclosure of this allergen was found in the provided text."


class GenerateAnswerNode(FunctionNode):
    """Rule-based, cited, tier-tagged advisory body assembly.

    Input state keys:
        screening_status: JSON list of final per-allergen statuses (from RerankFilterNode)
        overall_abstain:  bool (from RerankFilterNode)

    Output state keys (partial dict):
        grounded_answer: advisory body with [n] citation markers, or the
                          abstention message
        citations:        JSON list [{ref, id, allergen_en, allergen_ja, tier, source}]
                          (empty on abstain)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        overall_abstain = bool(state.get("overall_abstain", False))
        statuses: List[Dict[str, Any]] = from_json(state.get("screening_status"), []) or []

        if overall_abstain:
            emit_trace_event(
                "generate_answer_complete",
                {"overall_abstain": True, "citation_count": 0},
                state,
            )
            return {"grounded_answer": _ABSTAIN_ANSWER, "citations": to_json([])}

        citations: List[Dict[str, Any]] = []
        lines: List[str] = []

        mandatory = [s for s in statuses if isinstance(s, dict) and s.get("tier") == "mandatory"]
        recommended = [s for s in statuses if isinstance(s, dict) and s.get("tier") != "mandatory"]

        def _emit_group(header: str, group: List[Dict[str, Any]]) -> None:
            if not group:
                return
            lines.append(header)
            lines.append("")
            for entry in group:
                ref: Optional[int] = None
                if entry.get("status") in ("disclosed", "ambiguous"):
                    ref = len(citations) + 1
                    citations.append(
                        {
                            "ref": ref,
                            "id": entry.get("id", ""),
                            "allergen_en": entry.get("allergen_en", ""),
                            "allergen_ja": entry.get("allergen_ja", ""),
                            "tier": entry.get("tier", ""),
                            "source": entry.get("source", ""),
                        }
                    )
                lines.append(_format_line(ref, entry))
            lines.append("")

        if not mandatory and not recommended:
            lines.append(
                "No allergen provisions were screened for this request (no seeded "
                "knowledge-base entries were available)."
            )
        else:
            _emit_group(_NO_TEXT_TIER_HEADER["mandatory"], mandatory)
            _emit_group(_NO_TEXT_TIER_HEADER["recommended"], recommended)

        grounded_answer = "\n".join(lines).rstrip()

        # Domain audit: advisory body assembled.
        emit_trace_event(
            "generate_answer_complete",
            {
                "overall_abstain": False,
                "citation_count": len(citations),
                "mandatory_count": len(mandatory),
                "recommended_count": len(recommended),
                "answer_chars": len(grounded_answer),
            },
            state,
        )

        return {
            "grounded_answer": grounded_answer,
            "citations": to_json(citations),
        }
