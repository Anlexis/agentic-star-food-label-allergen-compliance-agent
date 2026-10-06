# RET-C2-283 — Unit Tests: RetrieveNode (inner domain node 2; ANONYMOUS)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# There is NO config parameter on execute()
# -- every config-precedence case here seeds State's retrieval_config, never
# a 2nd execute() argument.
#
# Japanese KB vocabulary safety: this file NEVER hand-types a Japanese/CJK
# literal. Every allergen term used to build a test payload is read
# PROGRAMMATICALLY from the real config/kb/allergen_kb.json at test-collection
# time and concatenated into a plain-ASCII carrier phrase -- this proves
# matching against the actual shipped KB content and removes any hand-
# transcription risk for the safety-critical allergen strings (see
# docs/03_test_spec.md "Japanese-text safety" note).
#
# Mirrors docs/03_test_spec.md S2.3 (RET-01..RET-10).
# Deterministic -- no LLM, no network. framework.* / src.* imports only.

import json
import pathlib
from unittest.mock import MagicMock

from framework.schemas.trust_level import TrustLevel

import src.nodes.retrieve_node
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json, to_json

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_KB_PATH = _REPO_ROOT / "config" / "kb" / "allergen_kb.json"
_KB = json.loads(_KB_PATH.read_text(encoding="utf-8"))
_KB_BY_ID = {e["id"]: e for e in _KB}

_MANDATORY_IDS = {e["id"] for e in _KB if e["tier"] == "mandatory"}
_RECOMMENDED_IDS = {e["id"] for e in _KB if e["tier"] == "recommended"}

# A no-match carrier: pure ASCII prose containing none of the KB's English
# alternative expressions (egg/milk/wheat/buckwheat/peanut/shrimp/prawn/crab/
# walnut/cashew/soy*/sesame/chicken/pork/beef/apple/gelatin*) and, being pure
# ASCII, none of the Japanese explicit/implicit terms either.
_NO_MATCH_TEXT = "no allergen relevant content here at all"


def _make_state(text, **extra) -> dict:
    state = {
        "label_text": text,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


def _candidate(result, kb_id):
    docs = from_json(result["matched_provisions"])
    matches = [d for d in docs if d["id"] == kb_id]
    assert matches, f"{kb_id} not present in matched_provisions: {docs!r}"
    return matches[0]


class TestRetrieveExplicitMatch:
    def test_ret_01_explicit_term_scores_full_confidence(self):
        entry = _KB_BY_ID["kb-mand-003"]  # Wheat
        term = entry["explicit_terms"][0]
        result = RetrieveNode()(_make_state(f"made with {term} as a base"))
        candidate = _candidate(result, "kb-mand-003")
        assert candidate["score"] == 1.0
        assert candidate["matched_term"] == term
        assert candidate["tier"] == "mandatory"
        assert candidate["allergen_en"] == "Wheat"


class TestRetrieveImplicitMatch:
    def test_ret_02_implicit_hint_scores_half_confidence(self):
        entry = _KB_BY_ID["kb-mand-002"]  # Milk
        assert entry["implicit_hint_terms"], "fixture assumption: Milk has an implicit hint term"
        hint = entry["implicit_hint_terms"][0]
        result = RetrieveNode()(_make_state(f"processed extract containing {hint}"))
        candidate = _candidate(result, "kb-mand-002")
        assert candidate["score"] == 0.5
        assert candidate["matched_term"] == hint


class TestRetrieveNoMatch:
    def test_ret_03_no_match_text_still_returns_every_mandatory_entry(self):
        """Mandatory-tier entries are NEVER dropped, even at score 0 -- a
        compliance checklist needs a status for every mandatory allergen
        every time (docs/02_design.md Safety Boundary #5)."""
        result = RetrieveNode()(_make_state(_NO_MATCH_TEXT))
        docs = from_json(result["matched_provisions"])
        mandatory_ids = {d["id"] for d in docs if d["tier"] == "mandatory"}
        assert mandatory_ids == _MANDATORY_IDS
        for d in docs:
            if d["tier"] == "mandatory":
                assert d["score"] == 0.0
                assert d["matched_term"] is None

    def test_ret_04_no_match_text_drops_every_recommended_entry(self):
        result = RetrieveNode()(_make_state(_NO_MATCH_TEXT))
        docs = from_json(result["matched_provisions"])
        recommended_ids = {d["id"] for d in docs if d["tier"] == "recommended"}
        assert recommended_ids == set()

    def test_ret_05_empty_text_behaves_like_no_match(self):
        result = RetrieveNode()(_make_state(""))
        docs = from_json(result["matched_provisions"])
        assert {d["id"] for d in docs} == _MANDATORY_IDS


class TestRetrieveEntryShape:
    def test_ret_06_entry_shape_and_excerpt_cap(self):
        result = RetrieveNode()(_make_state(_NO_MATCH_TEXT))
        docs = from_json(result["matched_provisions"])
        assert docs, "mandatory entries always populate matched_provisions"
        for d in docs:
            assert set(d.keys()) == {
                "id",
                "allergen_en",
                "allergen_ja",
                "tier",
                "source",
                "score",
                "matched_term",
                "excerpt",
            }
            assert len(d["excerpt"]) <= 280

    def test_matched_provisions_is_a_json_string(self):
        # Checkpoint safety: list-shaped State fields travel as JSON strings.
        result = RetrieveNode()(_make_state(_NO_MATCH_TEXT))
        assert isinstance(result["matched_provisions"], str)


class TestRetrieveOrdering:
    def test_ret_07_mandatory_entries_precede_recommended_and_sort_by_id(self):
        entry = _KB_BY_ID["kb-rec-001"]  # Soybean (recommended)
        term = entry["explicit_terms"][0]
        result = RetrieveNode()(_make_state(f"contains {term} lecithin"))
        docs = from_json(result["matched_provisions"])
        tiers = [d["tier"] for d in docs]
        assert "recommended" in tiers, "fixture assumption: the soy term must match kb-rec-001"
        # every "mandatory" entry precedes every "recommended" entry
        tier_rank = {"mandatory": 0, "recommended": 1}
        ranks = [tier_rank[t] for t in tiers]
        assert ranks == sorted(ranks), f"mandatory entries must all precede recommended: {tiers!r}"
        mandatory_run = [d["id"] for d in docs if d["tier"] == "mandatory"]
        assert mandatory_run == sorted(mandatory_run)


class TestRetrieveConfigPrecedence:
    """Config reaches this ANONYMOUS node ONLY via State (no execute() config
    parameter). Every case below seeds
    state["retrieval_config"], never a 2nd execute() argument."""

    def test_ret_08_bogus_kb_path_in_state_degrades_gracefully(self):
        state = _make_state(
            _NO_MATCH_TEXT,
            retrieval_config=to_json({"kb_path": "config/kb/does_not_exist.json"}),
        )
        result = RetrieveNode()(state)
        assert from_json(result["matched_provisions"]) == []
        notes = from_json(result.get("intake_notes"), [])
        assert any("not readable" in n for n in notes)

    def test_ret_09_absent_retrieval_config_falls_back_to_the_real_kb(self):
        result = RetrieveNode()(_make_state(_NO_MATCH_TEXT))
        docs = from_json(result["matched_provisions"])
        assert {d["id"] for d in docs} == _MANDATORY_IDS


class TestRetrieveNotesAccumulation:
    def test_ret_10_notes_append_never_clobber(self):
        state = _make_state(
            _NO_MATCH_TEXT,
            intake_notes=to_json(["earlier note from input validation"]),
            retrieval_config=to_json({"kb_path": "config/kb/bogus.json"}),
        )
        result = RetrieveNode()(state)
        notes = from_json(result["intake_notes"])
        assert notes[0] == "earlier note from input validation"
        assert len(notes) == 2


class TestRetrieveAudit:
    def test_ret_11_domain_audit_payload(self, monkeypatch):
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.retrieve_node, "emit_trace_event", spy)
        RetrieveNode()(_make_state(_NO_MATCH_TEXT))
        events = [call.args[0] for call in spy.call_args_list]
        assert "retrieve_complete" in events
        payload = spy.call_args_list[events.index("retrieve_complete")].args[1]
        assert payload["kb_entries"] == len(_KB)
        assert payload["candidates"] == len(_MANDATORY_IDS)
        assert payload["text_chars"] == len(_NO_MATCH_TEXT)


class TestSeededKnowledgeBase:
    """KB integrity — mirrors docs/03_test_spec.md CFG-08."""

    def test_kb_is_a_well_formed_entry_list(self):
        assert isinstance(_KB, list)
        assert len(_KB) == len(_MANDATORY_IDS) + len(_RECOMMENDED_IDS)
        for entry in _KB:
            assert set(entry.keys()) == {
                "id",
                "allergen_en",
                "allergen_ja",
                "tier",
                "source",
                "explicit_terms",
                "implicit_hint_terms",
                "content",
            }
            assert entry["tier"] in ("mandatory", "recommended")
            assert entry["id"] and entry["allergen_en"] and entry["allergen_ja"]

    def test_kb_ids_are_unique(self):
        ids = [e["id"] for e in _KB]
        assert len(ids) == len(set(ids))

    def test_mandatory_tier_has_nine_specified_raw_materials(self):
        # 食品表示基準 特定原材料 — 9 legally mandatory items (docs/02_design.md).
        assert len(_MANDATORY_IDS) == 9
