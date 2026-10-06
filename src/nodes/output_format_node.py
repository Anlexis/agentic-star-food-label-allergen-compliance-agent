"""AgentCore Platform v1.0"""

# RET-C2-283 - OutputFormatNode
# Domain node 5 (terminal): compose the final advisory body - the grouped
# per-allergen advisory text plus the Sources list.
#
# The standing advisory disclaimer ("advisory only - human review
# required") is DELIBERATELY NOT added here. It is appended downstream by
# the OUTER PostProcessNode (output content gate) instead, so the stamp is
# genuinely non-suppressible - a domain node forgetting to add it can never
# happen because the stamp lives in the one gate every response passes
# through unconditionally. See docs/02_design.md "Safety Boundary".
#
# Wired by the inner graph (DomainWorkflowGraph). get_output() of the inner
# graph surfaces formatted_answer + status (+ structured fields) to the
# outer merge_output(). Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json


class OutputFormatNode(FunctionNode):
    """Compose the final advisory body: grouped statuses + sources.

    Input state keys:
        grounded_answer: advisory body with [n] citation markers (or the
                          abstention message)
        citations:        JSON list [{ref, id, allergen_en, allergen_ja, tier, source}]

    Output state keys (partial dict):
        formatted_answer: final rendered advisory body string
        status:           AgentStatus.SUCCESS.value (plain string — never
                          write the bare enum to State)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        grounded_answer = state.get("grounded_answer") or (
            "No advisory screening result is available for this request."
        )
        citations: List[Dict[str, Any]] = from_json(state.get("citations"), []) or []

        lines: List[str] = []
        lines.append("# Food Label Allergen Compliance — Advisory Screening")
        lines.append("")
        lines.append(grounded_answer)
        lines.append("")
        lines.append("## Sources")
        if citations:
            for citation in citations:
                if not isinstance(citation, dict):
                    continue
                ref = citation.get("ref", "?")
                allergen = f"{citation.get('allergen_ja', '')} ({citation.get('allergen_en', '')})"
                source = str(citation.get("source", "")).strip()
                suffix = f" — {source}" if source else ""
                lines.append(f"- [{ref}] {allergen}{suffix}")
        else:
            lines.append("- none (no allergen provision reached a citable disclosed/ambiguous match)")

        formatted_answer = "\n".join(lines)

        # Domain audit: final advisory body composed.
        emit_trace_event(
            "output_format_complete",
            {
                "answer_chars": len(formatted_answer),
                "citation_count": len(citations),
            },
            state,
        )

        return {
            "formatted_answer": formatted_answer,
            "status": AgentStatus.SUCCESS.value,
        }
