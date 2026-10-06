# RET-C2-283 — Unit Tests: OutputFormatNode (inner domain node 5, terminal; ANONYMOUS)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# Safety-boundary-by-construction check: this node must NEVER add the
# "advisory only" stamp itself -- that lives exclusively in the outer
# PostProcessNode (docs/02_design.md Safety Boundary #3), so the stamp is
# genuinely non-suppressible (a domain node forgetting it can never happen).
#
# Mirrors docs/03_test_spec.md S2.6 (FMT-01..FMT-06).
# Deterministic -- no LLM, no network. framework.* / src.* imports only.

from unittest.mock import MagicMock

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

import src.nodes.output_format_node
from src.nodes.output_format_node import OutputFormatNode
from src.schemas.state import to_json


def _make_state(grounded_answer="the assembled advisory body", citations=None, **extra) -> dict:
    state = {
        "grounded_answer": grounded_answer,
        "citations": to_json(citations if citations is not None else []),
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestFullCompose:
    def test_fmt_01_header_body_and_sources_present(self):
        citations = [{"ref": 1, "allergen_en": "Egg", "allergen_ja": "JA-Egg", "source": "law-src"}]
        result = OutputFormatNode()(_make_state(citations=citations))
        answer = result["formatted_answer"]
        assert answer.startswith("# Food Label Allergen Compliance — Advisory Screening")
        assert "the assembled advisory body" in answer
        assert "## Sources" in answer
        assert "- [1] JA-Egg (Egg) — law-src" in answer
        assert result["status"] == AgentStatus.SUCCESS.value
        # Plain string, never the enum.
        assert isinstance(result["status"], str)
        assert not isinstance(result["status"], AgentStatus)

    def test_fmt_02_blank_source_has_no_suffix(self):
        citations = [{"ref": 1, "allergen_en": "Egg", "allergen_ja": "JA-Egg", "source": ""}]
        result = OutputFormatNode()(_make_state(citations=citations))
        source_line = next(ln for ln in result["formatted_answer"].splitlines() if ln.startswith("- [1]"))
        assert (
            source_line == "- [1] JA-Egg (Egg)"
        ), f"expected no dangling suffix on a blank source, got: {source_line!r}"


class TestNoDisclaimerHere:
    """Safety-boundary-by-construction: OutputFormatNode must NOT stamp the
    advisory notice — PostProcessNode is the one gate every response passes
    through unconditionally (docs/02_design.md Safety Boundary #3)."""

    def test_fmt_03_no_advisory_stamp_added_by_this_node(self):
        result = OutputFormatNode()(_make_state())
        assert "advisory only" not in result["formatted_answer"]
        assert "human review required" not in result["formatted_answer"]


class TestNoCitations:
    def test_fmt_04_no_citations_yields_explicit_none_line(self):
        result = OutputFormatNode()(_make_state(citations=[]))
        assert (
            "- none (no allergen provision reached a citable disclosed/ambiguous match)" in result["formatted_answer"]
        )


class TestMissingBody:
    def test_fmt_05_missing_grounded_answer_uses_fallback_text(self):
        state = _make_state()
        del state["grounded_answer"]
        result = OutputFormatNode()(state)
        assert "No advisory screening result is available for this request." in result["formatted_answer"]
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_fmt_05_empty_grounded_answer_uses_fallback_text(self):
        result = OutputFormatNode()(_make_state(grounded_answer=""))
        assert "No advisory screening result is available for this request." in result["formatted_answer"]


class TestOutputFormatAudit:
    def test_fmt_06_domain_audit_payload(self, monkeypatch):
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.output_format_node, "emit_trace_event", spy)
        citations = [{"ref": 1, "allergen_en": "Egg", "allergen_ja": "JA-Egg", "source": "s"}]
        result = OutputFormatNode()(_make_state(citations=citations))
        events = [call.args[0] for call in spy.call_args_list]
        assert "output_format_complete" in events
        payload = spy.call_args_list[events.index("output_format_complete")].args[1]
        assert payload["citation_count"] == 1
        assert payload["answer_chars"] == len(result["formatted_answer"])
