# RET-C2-283 — Unit Tests: RerankFilterNode (inner domain node 3; ANONYMOUS)
#
# This is the SAFETY-CRITICAL classification + abstain gate (docs/02_design.md
# "Safety Boundary" #2 and #5). Invocation canon: node(state) via
# BaseNode.__call__ with an ANONYMOUS caller. Config reaches this node ONLY
# via state["retrieval_config"] (no execute() config parameter).
#
# All allergen names used here are plain ASCII placeholders ("Item-A" style)
# — this file tests CLASSIFICATION LOGIC on synthetic candidates, not real KB
# content (that is RetrieveNode's / the safety-boundary suite's job), so no
# Japanese transcription risk applies.
#
# Mirrors docs/03_test_spec.md S2.4 (RRF-01..RRF-10).
# Deterministic -- no LLM, no network. framework.* / src.* imports only.

from unittest.mock import MagicMock

from framework.schemas.trust_level import TrustLevel

import src.nodes.rerank_filter_node
from src.nodes.rerank_filter_node import RerankFilterNode
from src.schemas.state import from_json, to_json

_LONG_TEXT = "ingredients: item a, item b, item c, item d"  # >= 8 chars


def _mandatory(id_, score, matched_term="term"):
    return {
        "id": id_,
        "allergen_en": f"Mandatory-{id_}",
        "allergen_ja": "M",
        "tier": "mandatory",
        "source": "s",
        "score": score,
        "matched_term": matched_term if score > 0 else None,
        "excerpt": "",
    }


def _recommended(id_, score, matched_term="term"):
    return {
        "id": id_,
        "allergen_en": f"Recommended-{id_}",
        "allergen_ja": "R",
        "tier": "recommended",
        "source": "s",
        "score": score,
        "matched_term": matched_term if score > 0 else None,
        "excerpt": "",
    }


def _make_state(candidates, text=_LONG_TEXT, **extra) -> dict:
    state = {
        "matched_provisions": to_json(candidates),
        "label_text": text,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestAbstainGate:
    """SAFETY-CRITICAL: overall_abstain is a mechanically-enforced data gate,
    not a wording choice (docs/02_design.md Safety Boundary #2)."""

    def test_rrf_01_short_text_triggers_abstain(self):
        result = RerankFilterNode()(_make_state([_mandatory("m1", 1.0)], text="short"))
        assert result["overall_abstain"] is True
        assert from_json(result["screening_status"]) == []

    def test_rrf_01b_abstain_emits_no_partial_verdict(self):
        """Even with high-confidence candidates present, abstention still
        emits a completely EMPTY screening_status -- never a hedged/partial
        list."""
        result = RerankFilterNode()(_make_state([_mandatory("m1", 1.0), _recommended("r1", 1.0)], text="x"))
        assert result["screening_status"] == "[]"

    def test_rrf_02_sufficiently_long_text_does_not_abstain(self):
        result = RerankFilterNode()(_make_state([_mandatory("m1", 1.0)]))
        assert result["overall_abstain"] is False

    def test_boundary_at_exactly_abstain_min_chars_does_not_abstain(self):
        # abstain_min_chars default is 8; len("12345678") == 8 -> NOT < 8.
        result = RerankFilterNode()(_make_state([_mandatory("m1", 1.0)], text="12345678"))
        assert result["overall_abstain"] is False

    def test_boundary_one_under_abstain_min_chars_does_abstain(self):
        result = RerankFilterNode()(_make_state([_mandatory("m1", 1.0)], text="1234567"))
        assert result["overall_abstain"] is True


class TestClassification:
    def test_rrf_03_score_at_or_above_threshold_is_disclosed(self):
        result = RerankFilterNode()(_make_state([_mandatory("m1", 0.75)]))
        status = from_json(result["screening_status"])
        assert status[0]["status"] == "disclosed"

    def test_rrf_03_score_between_zero_and_threshold_is_ambiguous(self):
        result = RerankFilterNode()(_make_state([_mandatory("m1", 0.5)]))
        status = from_json(result["screening_status"])
        assert status[0]["status"] == "ambiguous"

    def test_rrf_03_zero_score_is_missing(self):
        result = RerankFilterNode()(_make_state([_mandatory("m1", 0.0)]))
        status = from_json(result["screening_status"])
        assert status[0]["status"] == "missing"

    def test_rrf_04_custom_threshold_from_state_retrieval_config(self):
        state = _make_state(
            [_mandatory("m1", 0.5)],
            retrieval_config=to_json({"score_threshold": 0.4}),
        )
        result = RerankFilterNode()(state)
        status = from_json(result["screening_status"])
        assert status[0]["status"] == "disclosed"


class TestMandatoryNeverCapped:
    def test_rrf_05_mandatory_tier_survives_a_tight_top_k(self):
        candidates = [_mandatory(f"m{i}", 1.0) for i in range(9)]
        state = _make_state(candidates, retrieval_config=to_json({"top_k": 1}))
        result = RerankFilterNode()(state)
        status = from_json(result["screening_status"])
        assert len(status) == 9
        assert {s["tier"] for s in status} == {"mandatory"}


class TestRecommendedCapping:
    def test_rrf_06_recommended_tier_is_capped_at_top_k(self):
        # Pre-sorted score-desc, as RetrieveNode would deliver them.
        candidates = [_recommended(f"r{i}", 1.0 - i * 0.05) for i in range(5)]
        state = _make_state(candidates, retrieval_config=to_json({"top_k": 2}))
        result = RerankFilterNode()(state)
        status = from_json(result["screening_status"])
        assert len(status) == 2
        assert [s["id"] for s in status] == ["r0", "r1"]

    def test_rrf_07_include_recommended_false_drops_recommended_only(self):
        candidates = [_mandatory("m1", 1.0), _recommended("r1", 1.0)]
        state = _make_state(candidates, query_filters=to_json({"include_recommended": False}))
        result = RerankFilterNode()(state)
        status = from_json(result["screening_status"])
        tiers = {s["tier"] for s in status}
        assert tiers == {"mandatory"}

    def test_default_include_recommended_is_true_when_absent(self):
        candidates = [_mandatory("m1", 1.0), _recommended("r1", 1.0)]
        result = RerankFilterNode()(_make_state(candidates))
        status = from_json(result["screening_status"])
        assert {s["tier"] for s in status} == {"mandatory", "recommended"}


class TestCallerTopK:
    def test_rrf_08_caller_top_k_narrower_than_config_wins(self):
        candidates = [_recommended(f"r{i}", 1.0 - i * 0.05) for i in range(5)]
        state = _make_state(
            candidates,
            retrieval_config=to_json({"top_k": 10}),
            query_filters=to_json({"top_k": 2}),
        )
        result = RerankFilterNode()(state)
        assert len(from_json(result["screening_status"])) == 2

    def test_rrf_08_caller_top_k_wider_than_config_does_not_widen(self):
        candidates = [_recommended(f"r{i}", 1.0 - i * 0.05) for i in range(5)]
        state = _make_state(
            candidates,
            retrieval_config=to_json({"top_k": 3}),
            query_filters=to_json({"top_k": 10}),
        )
        result = RerankFilterNode()(state)
        assert len(from_json(result["screening_status"])) == 3


class TestGarbageEntries:
    def test_rrf_09_non_dict_entries_are_skipped(self):
        candidates = [_mandatory("m1", 1.0), "not-a-dict", 42, None]
        result = RerankFilterNode()(_make_state(candidates))
        status = from_json(result["screening_status"])
        assert [s["id"] for s in status] == ["m1"]

    def test_rrf_09_uncoercible_score_is_treated_as_zero(self):
        entry = _mandatory("m1", 1.0)
        entry["score"] = "not-a-number"
        result = RerankFilterNode()(_make_state([entry]))
        status = from_json(result["screening_status"])
        assert status[0]["status"] == "missing"
        assert status[0]["score"] == 0.0


class TestRerankAudit:
    def test_rrf_10_domain_audit_payload(self, monkeypatch):
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.rerank_filter_node, "emit_trace_event", spy)
        RerankFilterNode()(_make_state([_mandatory("m1", 1.0)]))
        events = [call.args[0] for call in spy.call_args_list]
        assert "rerank_filter_complete" in events
        payload = spy.call_args_list[events.index("rerank_filter_complete")].args[1]
        assert payload["overall_abstain"] is False
        assert payload["mandatory_count"] == 1
