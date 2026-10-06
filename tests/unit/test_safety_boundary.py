# RET-C2-283 — Safety-Boundary Proof Suite
#
# THIS IS THE SAFETY CASE for a food-allergen advisory template — a false
# "disclosed" could seriously harm someone. Each class below proves ONE of
# the hard invariants named in docs/02_design.md "Safety Boundary", driven
# end-to-end through the REAL outer agent (Graph().invoke()), not just at the
# per-node unit level (that coverage lives in the individual node test files
# — this file is the integration-level proof that the properties survive
# composition, not just isolated node logic).
#
# Japanese-text safety: every Japanese/allergen string used to build a test
# payload or assertion is either (a) read PROGRAMMATICALLY from the real
# config/kb/allergen_kb.json / deploy/invoke_payload.json at test-collection
# time, or (b) imported directly from the node/graph module's own constant
# (_ABSTAIN_ANSWER, _ADVISORY_STAMP, _ADVISORY_NOTICE) — this file contains
# ZERO hand-typed CJK literals, eliminating transcription risk for the
# safety-critical allergen vocabulary end to end.
#
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import json
import pathlib
import re

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import _ADVISORY_NOTICE, Graph
from src.nodes.generate_answer_node import _ABSTAIN_ANSWER
from src.nodes.post_process_node import _ADVISORY_STAMP

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_KB = json.loads((_REPO_ROOT / "config" / "kb" / "allergen_kb.json").read_text(encoding="utf-8"))
_KB_BY_ID = {e["id"]: e for e in _KB}
_MANDATORY_IDS = {e["id"] for e in _KB if e["tier"] == "mandatory"}
_RECOMMENDED_IDS = {e["id"] for e in _KB if e["tier"] == "recommended"}

_DEPLOY_PAYLOAD = json.loads((_REPO_ROOT / "deploy" / "invoke_payload.json").read_text(encoding="utf-8"))
_DEPLOY_LABEL = _DEPLOY_PAYLOAD["input"]

_WHEAT = _KB_BY_ID["kb-mand-003"]
_WHEAT_PAYLOAD = f"ingredient declaration includes {_WHEAT['explicit_terms'][0]} as the base"

_BANNED_VERDICT_RE = re.compile(r"\b(pass|fail|compliant|non-compliant|unsafe)\b")


def _invoke(text: str, trust: TrustLevel = TrustLevel.VERIFIED_EXTERNAL) -> dict:
    ctx = InvocationContext(caller_trust_level=trust, caller_id="safety-suite")
    return Graph().invoke(text, ctx=ctx)


class TestAbstentionIsReal:
    """Safety Boundary #2 — a low-confidence retrieval abstains, it never
    synthesises a verdict from insufficient signal."""

    def test_short_input_abstains_end_to_end(self):
        result = _invoke("hi")
        assert (
            result["status"] == AgentStatus.SUCCESS.value
        ), "abstention is itself a valid, successful outcome — not an error"
        assert result["overall_abstain"] is True
        assert result["screening_status"] == []
        assert result["citations"] == []

    def test_abstention_output_routes_to_a_human_reviewer(self):
        result = _invoke("hi")
        assert _ABSTAIN_ANSWER in result["output"]
        assert "route" in _ABSTAIN_ANSWER.lower() and "human reviewer" in _ABSTAIN_ANSWER.lower()

    def test_abstention_still_carries_the_advisory_stamp(self):
        # The non-suppressible stamp survives even the no-verdict abstain path.
        result = _invoke("hi")
        assert _ADVISORY_STAMP in result["output"]


class TestNeverAnAuthoritativeVerdict:
    """Safety Boundary #1 — no node, at any layer, ever emits a pass/fail /
    compliant / non-compliant / safe / unsafe determination."""

    def test_full_run_output_carries_no_banned_verdict_vocabulary(self):
        result = _invoke(_DEPLOY_LABEL)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert not _BANNED_VERDICT_RE.search(result["output"].lower()), result["output"]

    def test_structured_screening_status_uses_only_the_three_status_labels(self):
        result = _invoke(_DEPLOY_LABEL)
        for entry in result["screening_status"]:
            assert entry["status"] in ("disclosed", "missing", "ambiguous")


class TestTiersAreNeverConflated:
    """Safety Boundary #5 — mandatory (9 items) vs recommended must be
    tier-tagged correctly and never conflated; mandatory can never be
    narrowed away."""

    def test_every_returned_tier_matches_the_kb_ground_truth(self):
        result = _invoke(_DEPLOY_LABEL)
        for entry in result["screening_status"]:
            assert entry["tier"] == _KB_BY_ID[entry["id"]]["tier"], (
                f"tier mismatch for {entry['id']}: got {entry['tier']!r}, "
                f"KB says {_KB_BY_ID[entry['id']]['tier']!r}"
            )

    def test_all_nine_mandatory_allergens_always_get_a_status(self):
        result = _invoke(_DEPLOY_LABEL)
        returned_mandatory_ids = {e["id"] for e in result["screening_status"] if e["tier"] == "mandatory"}
        assert returned_mandatory_ids == _MANDATORY_IDS

    def test_recommended_tier_never_appears_as_mandatory_or_vice_versa(self):
        result = _invoke(_DEPLOY_LABEL)
        for entry in result["screening_status"]:
            if entry["id"] in _MANDATORY_IDS:
                assert entry["tier"] == "mandatory"
            if entry["id"] in _RECOMMENDED_IDS:
                assert entry["tier"] == "recommended"

    def test_include_recommended_false_never_narrows_the_mandatory_tier(self):
        payload = json.dumps({"text": _DEPLOY_LABEL, "include_recommended": False})
        result = _invoke(payload)
        returned_mandatory_ids = {e["id"] for e in result["screening_status"] if e["tier"] == "mandatory"}
        assert returned_mandatory_ids == _MANDATORY_IDS
        assert all(e["tier"] != "recommended" for e in result["screening_status"])


class TestAdvisoryStampNonSuppressible:
    """Safety Boundary #3 — the stamp is appended by the ONE gate every
    response passes through unconditionally; the structured payload carries
    the equivalent advisory_notice as a fixed constant on every SUCCESS."""

    def test_text_output_carries_the_stamp_verbatim(self):
        result = _invoke(_DEPLOY_LABEL)
        assert _ADVISORY_STAMP in result["output"]

    def test_structured_payload_carries_the_fixed_advisory_notice(self):
        result = _invoke(_DEPLOY_LABEL)
        assert result["advisory_notice"] == _ADVISORY_NOTICE

    def test_stamp_present_regardless_of_which_allergens_were_found(self):
        for payload in (_DEPLOY_LABEL, _WHEAT_PAYLOAD, "hi"):
            result = _invoke(payload)
            assert _ADVISORY_STAMP in result["output"], f"stamp missing for payload {payload!r}"


class TestAllergenNamesRoundTripByteIdentical:
    """A corrupted allergen string is a safety defect, not a cosmetic one:
    prove the KB's allergen_ja value survives the full outer invoke() byte-
    for-byte -- both by Python string equality AND explicit UTF-8 byte
    comparison."""

    def test_wheat_allergen_ja_round_trips_through_full_invoke(self):
        result = _invoke(_WHEAT_PAYLOAD)
        matches = [e for e in result["screening_status"] if e["id"] == "kb-mand-003"]
        assert matches, "expected kb-mand-003 (Wheat) in the structured screening_status"
        returned_ja = matches[0]["allergen_ja"]
        expected_ja = _WHEAT["allergen_ja"]
        assert returned_ja == expected_ja
        assert returned_ja.encode("utf-8") == expected_ja.encode("utf-8")
        # Also present verbatim in the free-text advisory body.
        assert expected_ja in result["output"]

    def test_wheat_allergen_ja_round_trips_through_citations(self):
        result = _invoke(_WHEAT_PAYLOAD)
        citation_matches = [c for c in result["citations"] if c["id"] == "kb-mand-003"]
        assert citation_matches
        assert citation_matches[0]["allergen_ja"] == _WHEAT["allergen_ja"]

    def test_every_kb_entry_id_present_in_a_full_screening_maps_to_its_own_ja_name(self):
        """No cross-contamination: every id in the structured output carries
        exactly the allergen_ja that THAT id owns in the KB -- never a
        neighbouring entry's name."""
        result = _invoke(_DEPLOY_LABEL)
        for entry in result["screening_status"]:
            assert entry["allergen_ja"] == _KB_BY_ID[entry["id"]]["allergen_ja"]
            assert entry["allergen_en"] == _KB_BY_ID[entry["id"]]["allergen_en"]


class TestNoRawInputLeakage:
    """Defence in depth: the domain nodes never echo raw caller-supplied text
    into the answer -- only vetted, KB-sourced allergen fields. A uniquely
    tagged raw input therefore never appears in the final output (this is
    also WHY an output-gate credential/PII leak cannot be forced through the normal
    domain data path -- see test_post_process_node.py for the direct gate
    proof on a crafted `result` string)."""

    def test_a_unique_raw_input_marker_never_appears_in_the_final_output(self):
        marker = "UNIQUE-RAW-INPUT-MARKER-2026-07-29-not-in-kb"
        result = _invoke(f"{_WHEAT_PAYLOAD} {marker}")
        assert marker not in result["output"]
        assert marker not in json.dumps(result.get("citations", []))
