# RET-C2-283 — Unit Tests: PostProcessNode (outer post_process slot; output gate)
#
# Invocation canon: node(state) via BaseNode.__call__. PostProcessNode is the
# second outer S-gate slot and requires VERIFIED_EXTERNAL (like PreProcessNode
# — the ANONYMOUS rejection lives in test_trust_gate.py).
#
# SAFETY-CRITICAL (docs/02_design.md "Safety Boundary" #3): the standing
# "advisory only — human review required" stamp is appended HERE,
# unconditionally, on the one clean-result path every response passes
# through — that is what makes it non-suppressible. On the blocked path
# the caller never receives an unqualified screening verdict either: the
# result is fully replaced by the ERROR-status closed-set envelope
# ({"reason": "output_withheld"}) with `result` cleared (a distinct, but
# equally load-bearing, safety property — see TestBlockedPathNeverLeaks).
#
# Mirrors docs/03_test_spec.md S2.7 (POST-01..POST-10).
# Deterministic -- no LLM, no network. framework.* / src.* imports only.

import json
from unittest.mock import MagicMock

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

import src.nodes.post_process_node
from src.nodes.post_process_node import ERROR_REASONS, PostProcessNode, _ADVISORY_STAMP, error_envelope

_CLEAN_RESULT = (
    "# Food Label Allergen Compliance — Advisory Screening\n\n"
    "## Mandatory Allergens (specified raw materials)\n\n"
    "- Egg (JA-Egg) — DISCLOSED [1]: matched explicit term.\n"
)

# JWT-shaped token built at runtime so no credential-shaped literal ever sits
# in the repository (CI credential-scan hygiene).
_FAKE_JWT = "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12


def _make_state(result_text, **extra) -> dict:
    state = {
        "result": result_text,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPostProcessClean:
    def test_post_01_clean_output_passes_through_with_stamp(self):
        result = PostProcessNode()(_make_state(_CLEAN_RESULT))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Regression guard: State carries the plain string, never the enum.
        assert isinstance(result["status"], str)
        assert not isinstance(result["status"], AgentStatus)
        assert result["formatted_output"].startswith(_CLEAN_RESULT)
        assert _ADVISORY_STAMP in result["formatted_output"]

    def test_post_02_empty_result_is_non_fatal_and_unstamped(self):
        result = PostProcessNode()(_make_state(""))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == ""


class TestAdvisoryStampNonSuppressible:
    """SAFETY-CRITICAL: every non-empty SUCCESS result carries the stamp,
    appended unconditionally by this ONE gate — never by a domain node
    (docs/02_design.md Safety Boundary #3)."""

    def test_post_08_stamp_present_on_every_clean_result(self):
        for body in (
            _CLEAN_RESULT,
            "a minimal one-line result",
            "## Mandatory Allergens\n\n- Wheat — MISSING: no disclosure found.",
        ):
            result = PostProcessNode()(_make_state(body))
            assert _ADVISORY_STAMP in result["formatted_output"], f"stamp missing for body: {body!r}"

    def test_post_08_stamp_text_matches_the_docs02_safety_boundary_wording(self):
        assert _ADVISORY_STAMP.startswith("This screening result is advisory only — human review required.")
        assert "does not issue a compliance verdict" in _ADVISORY_STAMP


class TestPostProcessS3Gate:
    def _assert_blocked(self, result, secret):
        assert result["status"] == AgentStatus.ERROR.value
        assert any("output blocked" in str(e) for e in result["error_log"])
        # The raw secret must not survive into either surfaced field.
        assert secret not in str(result.get("formatted_output", ""))
        assert secret not in str(result.get("result", ""))
        # The caller-visible value is the closed-set envelope; `result` is
        # cleared so the base envelope's `formatted_output or result` fallback
        # has nothing pre-gate to fall back to.
        assert result["formatted_output"] == error_envelope("output_withheld")
        assert result["result"] is None

    def test_post_03_api_key_is_blocked(self):
        secret = "sk-ABCDEF0123456789abcdef"
        result = PostProcessNode()(_make_state(f"{_CLEAN_RESULT}\n<!-- debug api_key={secret} -->"))
        self._assert_blocked(result, secret)

    def test_post_04_credential_assignment_is_blocked(self):
        secret = "password=super_secret_value_123"
        result = PostProcessNode()(_make_state(f"{_CLEAN_RESULT}\ninternal note: {secret}"))
        self._assert_blocked(result, "super_secret_value_123")

    def test_post_05_jwt_is_blocked(self):
        result = PostProcessNode()(_make_state(f"{_CLEAN_RESULT}\nsession token {_FAKE_JWT}"))
        self._assert_blocked(result, _FAKE_JWT)

    def test_post_06_bearer_token_is_blocked(self):
        secret = "Bearer abcdefghijklmnopqrstuvwxyz0123456789"
        result = PostProcessNode()(_make_state(f"{_CLEAN_RESULT}\nauthorization: {secret}"))
        self._assert_blocked(result, secret)

    def test_post_07_email_pii_is_blocked(self):
        secret = "reviewer.contact@example.com"
        result = PostProcessNode()(_make_state(f"{_CLEAN_RESULT}\ncontact {secret} for questions"))
        self._assert_blocked(result, secret)

    def test_post_07_long_digit_run_pii_is_blocked(self):
        secret = "1234567890123"
        result = PostProcessNode()(_make_state(f"{_CLEAN_RESULT}\nref number {secret}"))
        self._assert_blocked(result, secret)


class TestBlockedPathNeverLeaks:
    """The blocked path does not carry the literal advisory-stamp substring
    (it is replaced entirely, by design — the caller gets an explicit ERROR +
    the closed-set reason instead of a stamped result). What matters for
    safety is proven here: no path EVER returns an unqualified/unstamped
    screening verdict — status=ERROR and the constant `output_withheld`
    reason together are an equally unambiguous "do not trust this output"
    signal, and nothing of the refused result rides out with them."""

    def test_blocked_response_is_exactly_the_closed_set_envelope_never_a_partial_result(self):
        secret = "sk-ABCDEF0123456789abcdef"
        result = PostProcessNode()(_make_state(f"{_CLEAN_RESULT} api_key={secret}"))
        assert result["formatted_output"] == {"reason": "output_withheld"}
        rendered = json.dumps(result, ensure_ascii=False, default=str)
        assert "DISCLOSED" not in rendered
        assert "MISSING" not in rendered
        assert _ADVISORY_STAMP not in rendered
        assert secret not in rendered
        assert result["status"] == AgentStatus.ERROR.value

    def test_blocked_envelope_is_truthy_and_drawn_from_the_closed_set(self):
        # AgentBaseGraph.get_output() selects `formatted_output or result` with
        # no status check: a falsy envelope would re-open the fallback.
        result = PostProcessNode()(_make_state(f"{_CLEAN_RESULT}\nsession token {_FAKE_JWT}"))
        assert result["formatted_output"]
        assert result["formatted_output"]["reason"] in ERROR_REASONS
        assert result["result"] is None


class TestPostProcessAudit:
    def test_post_10_domain_audit_payload(self, monkeypatch):
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.post_process_node, "emit_trace_event", spy)
        result = PostProcessNode()(_make_state(_CLEAN_RESULT))
        events = [call.args[0] for call in spy.call_args_list]
        assert "post_process_complete" in events
        payload = spy.call_args_list[events.index("post_process_complete")].args[1]
        assert payload["output_chars"] == len(result["formatted_output"])

    def test_blocked_path_audit_carries_the_reason_and_the_rule_never_the_value(self, monkeypatch):
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.post_process_node, "emit_trace_event", spy)
        secret = "reviewer.contact@example.com"
        PostProcessNode()(_make_state(f"{_CLEAN_RESULT}\ncontact {secret}"))
        events = {call.args[0]: call.args[1] for call in spy.call_args_list}
        assert events["post_process_output_blocked"] == {"reason": "output_withheld", "violation": "pii_email"}
        assert "post_process_complete" not in events
        assert secret not in json.dumps(events, default=str)
