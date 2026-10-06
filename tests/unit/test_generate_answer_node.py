# RET-C2-283 — Unit Tests: GenerateAnswerNode (inner domain node 4; ANONYMOUS)
#
# This is the SAFETY-CRITICAL answer-assembly node (docs/02_design.md "Safety
# Boundary" #1 and #2): it must never emit an authoritative verdict, and on
# overall_abstain it must emit ONLY the abstention message.
#
# Tier headers are asserted against the node's OWN module constant
# (_NO_TEXT_TIER_HEADER) rather than a hand-typed Japanese literal here --
# zero transcription risk for the 特定原材料 / 特定原材料に準ずるもの text.
# Allergen id/en/ja values fed into synthetic statuses are plain ASCII
# placeholders (this node only formats given structured input; real KB
# content is RetrieveNode's / the safety-boundary suite's concern).
#
# Mirrors docs/03_test_spec.md S2.5 (GEN-01..GEN-07).
# Deterministic -- no LLM, no network. framework.* / src.* imports only.

import re
from unittest.mock import MagicMock

from framework.schemas.trust_level import TrustLevel

import src.nodes.generate_answer_node
from src.nodes.generate_answer_node import _ABSTAIN_ANSWER, _NO_TEXT_TIER_HEADER, GenerateAnswerNode
from src.schemas.state import from_json, to_json

# A definitive pass/fail-style verdict token, word-boundary matched against a
# lowercased answer -- deliberately excludes "compliance"/"safety" (legitimate
# advisory-framing words the node's neighbours use) and targets only the
# banned verdict vocabulary itself.
_BANNED_VERDICT_RE = re.compile(r"\b(pass|fail|compliant|non-compliant|unsafe)\b")


def _status(id_, status, tier="mandatory", matched_term="term"):
    return {
        "id": id_,
        "allergen_en": f"Allergen-{id_}",
        "allergen_ja": f"JA-{id_}",
        "tier": tier,
        "status": status,
        "score": 1.0 if status == "disclosed" else 0.5,
        "matched_term": matched_term,
        "source": "src",
    }


def _make_state(statuses, overall_abstain=False, **extra) -> dict:
    state = {
        "screening_status": to_json(statuses),
        "overall_abstain": overall_abstain,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestAbstainPath:
    """SAFETY-CRITICAL: on abstain, NEVER a synthesised per-allergen verdict —
    only the fixed abstention message, and empty citations."""

    def test_gen_01_abstain_emits_only_the_fixed_message(self):
        result = GenerateAnswerNode()(_make_state([_status("m1", "disclosed")], overall_abstain=True))
        assert result["grounded_answer"] == _ABSTAIN_ANSWER
        assert result["citations"] == "[]"

    def test_gen_01b_abstain_ignores_any_populated_screening_status(self):
        """Even if screening_status somehow carries entries, overall_abstain
        wins -- the abstain branch returns unconditionally before any
        per-allergen formatting runs."""
        statuses = [_status(f"m{i}", "disclosed") for i in range(9)]
        result = GenerateAnswerNode()(_make_state(statuses, overall_abstain=True))
        assert result["grounded_answer"] == _ABSTAIN_ANSWER
        for entry in statuses:
            assert entry["allergen_en"] not in result["grounded_answer"]


class TestCitationMarkers:
    def test_gen_02_disclosed_and_ambiguous_get_citation_refs(self):
        statuses = [_status("m1", "disclosed"), _status("m2", "ambiguous")]
        result = GenerateAnswerNode()(_make_state(statuses))
        assert "[1]" in result["grounded_answer"]
        assert "[2]" in result["grounded_answer"]
        citations = from_json(result["citations"])
        assert [c["ref"] for c in citations] == [1, 2]

    def test_gen_03_missing_gets_no_citation_ref(self):
        statuses = [_status("m1", "missing")]
        result = GenerateAnswerNode()(_make_state(statuses))
        assert "[1]" not in result["grounded_answer"]
        assert from_json(result["citations"]) == []

    def test_gen_03_citations_carry_the_source_fields(self):
        statuses = [_status("m1", "disclosed")]
        result = GenerateAnswerNode()(_make_state(statuses))
        citation = from_json(result["citations"])[0]
        assert citation == {
            "ref": 1,
            "id": "m1",
            "allergen_en": "Allergen-m1",
            "allergen_ja": "JA-m1",
            "tier": "mandatory",
            "source": "src",
        }


class TestTierGrouping:
    """Both legal tiers must be distinguished and never conflated
    (docs/02_design.md Safety Boundary)."""

    def test_gen_04_mandatory_header_present_when_mandatory_entries_exist(self):
        statuses = [_status("m1", "disclosed", tier="mandatory")]
        result = GenerateAnswerNode()(_make_state(statuses))
        assert _NO_TEXT_TIER_HEADER["mandatory"] in result["grounded_answer"]
        assert _NO_TEXT_TIER_HEADER["recommended"] not in result["grounded_answer"]

    def test_gen_04_recommended_header_present_when_recommended_entries_exist(self):
        statuses = [_status("r1", "disclosed", tier="recommended")]
        result = GenerateAnswerNode()(_make_state(statuses))
        assert _NO_TEXT_TIER_HEADER["recommended"] in result["grounded_answer"]
        assert _NO_TEXT_TIER_HEADER["mandatory"] not in result["grounded_answer"]

    def test_gen_04_both_headers_present_when_both_tiers_populated(self):
        statuses = [
            _status("m1", "disclosed", tier="mandatory"),
            _status("r1", "disclosed", tier="recommended"),
        ]
        result = GenerateAnswerNode()(_make_state(statuses))
        assert _NO_TEXT_TIER_HEADER["mandatory"] in result["grounded_answer"]
        assert _NO_TEXT_TIER_HEADER["recommended"] in result["grounded_answer"]

    def test_gen_04_a_recommended_entry_never_appears_under_the_mandatory_header(self):
        statuses = [
            _status("m1", "disclosed", tier="mandatory"),
            _status("r1", "disclosed", tier="recommended"),
        ]
        answer = GenerateAnswerNode()(_make_state(statuses))["grounded_answer"]
        mandatory_section = answer.split(_NO_TEXT_TIER_HEADER["recommended"])[0]
        assert "Allergen-r1" not in mandatory_section


class TestNoEntries:
    def test_gen_05_no_entries_yields_the_no_coverage_message(self):
        result = GenerateAnswerNode()(_make_state([]))
        assert "No allergen provisions were screened" in result["grounded_answer"]
        assert from_json(result["citations"]) == []


class TestNoVerdictRule:
    """SAFETY-CRITICAL: never an authoritative pass/fail / compliant /
    non-compliant / safe-unsafe determination anywhere in the answer body
    (docs/02_design.md Safety Boundary #1)."""

    def test_gen_06_full_run_carries_no_banned_verdict_vocabulary(self):
        statuses = [
            _status("m1", "disclosed", tier="mandatory"),
            _status("m2", "ambiguous", tier="mandatory"),
            _status("m3", "missing", tier="mandatory"),
            _status("r1", "disclosed", tier="recommended"),
        ]
        answer = GenerateAnswerNode()(_make_state(statuses))["grounded_answer"]
        assert not _BANNED_VERDICT_RE.search(answer.lower()), answer

    def test_gen_06_status_vocabulary_is_limited_to_the_three_enum_labels(self):
        statuses = [
            _status("m1", "disclosed"),
            _status("m2", "ambiguous"),
            _status("m3", "missing"),
        ]
        answer = GenerateAnswerNode()(_make_state(statuses))["grounded_answer"]
        found = set(re.findall(r"\b(DISCLOSED|AMBIGUOUS|MISSING)\b", answer))
        assert found == {"DISCLOSED", "AMBIGUOUS", "MISSING"}


class TestGenerateAnswerAudit:
    def test_gen_07_domain_audit_payload(self, monkeypatch):
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.generate_answer_node, "emit_trace_event", spy)
        GenerateAnswerNode()(_make_state([_status("m1", "disclosed")]))
        events = [call.args[0] for call in spy.call_args_list]
        assert "generate_answer_complete" in events
        payload = spy.call_args_list[events.index("generate_answer_complete")].args[1]
        assert payload["overall_abstain"] is False
        assert payload["citation_count"] == 1
