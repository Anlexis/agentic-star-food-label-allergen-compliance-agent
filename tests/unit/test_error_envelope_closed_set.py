# RET-C2-283 — the caller-visible ERROR envelope carries closed-set labels only.
#
# On any non-success invoke the caller must receive values this template chose
# from a closed set — a constant reason code — and nothing read from error_log
# or from any other node- or framework-authored string. In this repository the
# live channel is FoodLabelAllergenComplianceAgent.get_output(): the backbone
# routes every non-success status straight to finalize (post_process never
# runs on an error), and the override used to copy state["error_log"] —
# single-line entries, de-duplicated — into the invoke body. An entry there can
# be a caller-data rejection, the trust gate's denial line, an inner rejection
# carried over by on_subgraph_error(), or — through the framework's exception
# wrapping in BaseNode.__call__ — an exception's message with its traceback.
# Filtering such text down to one line is not a closed set; not publishing it
# is.
#
# Two surfaces are held here:
#   - get_output(), parameterised over every non-success shape the state can
#     take at finalize, including the ones a compiled run cannot be coaxed
#     into (a foreign value in the reason slot, a surviving pre-gate result);
#   - PostProcessNode's formatted_output on every path that returns ERROR
#     (each output-gate rule, through both execute() and the framework
#     pipeline; the already-errored entry via execute()): the closed-set
#     envelope, truthy, with result cleared and the gate message on error_log
#     only.
# The full path through the real ASGI /invoke is held in
# tests/proof_of_boundary/test_invoke_e2e.py.
#
# The sentinel is deliberately NOT credential-shaped and carries no trace
# fragment: a filtering envelope passes it straight through, which is the
# defect these tests must fail on — not one a filter happened to catch.

import json
from unittest.mock import MagicMock

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

import src.nodes.post_process_node
from src.graph.graph import FoodLabelAllergenComplianceAgent
from src.nodes.post_process_node import (
    ERROR_REASONS,
    PostProcessNode,
    _REASON_OUTPUT_WITHHELD,
    _REASON_WORKFLOW_FAILED,
    error_envelope,
)
from src.schemas.state import to_json

# Assembled at runtime (never a committed literal): a name, an e-mail and a
# token-shaped fragment — what an echoed upstream body or a wrapped exception
# message can carry.
_FRAGMENTS = ("A. Tanaka", "a.tanaka@example.com", "sk-" + "live-xxx")
_SENTINEL = (
    "boom: upstream said {'customer':'"
    + _FRAGMENTS[0]
    + "','email':'"
    + _FRAGMENTS[1]
    + "','token':'"
    + _FRAGMENTS[2]
    + "'}"
)
_MARKERS = (_SENTINEL, "upstream said", *_FRAGMENTS)

# Entries of the shapes the real pipeline writes to error_log — internal too,
# never projected: this template's own single-line rejection, the framework's
# trust-gate denial, and the framework's exception wrapper (message + traceback).
_REJECTION = "PreProcessNode: " + _SENTINEL
_TRUST_DENIAL = "[PreProcessNode] S-1 trust gate denied: required=verified_external, caller=anonymous"
_WRAPPED_EXCEPTION = "[RetrieveNode] " + _SENTINEL + '\nTraceback (most recent call last):\n  File "/abs/x.py", line 1'
_INTERNAL_ENTRIES = [_REJECTION, _TRUST_DENIAL, _WRAPPED_EXCEPTION]
_INTERNAL_MARKERS = (*_MARKERS, "trust gate denied", "Traceback", 'File "', "PreProcessNode:")

_BASE_ENVELOPE_KEYS = {"output", "status", "trace_id", "correlation_id", "node_history"}
_STRUCTURED_KEYS = ("screening_status", "citations", "overall_abstain", "advisory_notice")
_CLEAN_ANSWER = "# Food Label Allergen Compliance — Advisory Screening\n\n- Egg — DISCLOSED [1]: matched explicit term."


def _strings(value):
    """Every string reachable in value: dict keys and values, list/tuple items,
    and the repr of anything else that is not a plain scalar."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _strings(key)
            yield from _strings(child)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for child in value:
            yield from _strings(child)
    elif value is not None and not isinstance(value, (bool, int, float)):
        yield repr(value)


def _found(mapping, *markers) -> list:
    """The markers reachable anywhere inside mapping, walking nested values."""
    texts = list(_strings(mapping))
    return [marker for marker in markers if any(marker in text for text in texts)]


def _assert_internal_text_absent(mapping: dict) -> None:
    assert _found(mapping, *_INTERNAL_MARKERS) == [], mapping
    rendered = json.dumps(mapping, ensure_ascii=False, default=str)
    for marker in _INTERNAL_MARKERS:
        assert marker not in rendered


# ── get_output(): the invoke envelope on every non-success state ──────────────


def _finalize_state(**overrides) -> dict:
    """The outer state as finalize sees it after a non-success run: the
    internal channel carries every shape of entry the pipeline can put there,
    and the pre-gate fields survive in state."""
    state = {
        "status": AgentStatus.ERROR.value,
        "user_input": "screen this label for allergen disclosures",
        "validated_input": "screen this label for allergen disclosures",
        "error_log": list(_INTERNAL_ENTRIES),
        "result": _SENTINEL,
        "allergen_screening_result": _SENTINEL,
        "screening_status": to_json([{"id": "kb-mand-001", "status": "disclosed", "source": _SENTINEL}]),
        "citations": to_json([{"ref": 1, "id": "kb-mand-001", "source": _SENTINEL}]),
        "overall_abstain": False,
        "intake_notes": to_json([_SENTINEL]),
        "trace_id": "tr",
        "correlation_id": "co",
        "node_history": ["InitializeNode", "PreProcessNode", "FinalizeNode"],
    }
    state.update(overrides)
    return state


_NON_SUCCESS_STATES = [
    pytest.param({}, _REASON_WORKFLOW_FAILED, id="caller-data-rejected-at-pre-process"),
    pytest.param({"status": AgentStatus.TIMEOUT.value}, _REASON_WORKFLOW_FAILED, id="timeout-status"),
    pytest.param({"status": AgentStatus.PENDING.value}, _REASON_WORKFLOW_FAILED, id="pending-status"),
    pytest.param(
        {"formatted_output": error_envelope(_REASON_OUTPUT_WITHHELD), "result": None},
        _REASON_OUTPUT_WITHHELD,
        id="output-gate-refused",
    ),
    pytest.param({"formatted_output": _SENTINEL}, _REASON_WORKFLOW_FAILED, id="non-dict-formatted-output-survives"),
    pytest.param(
        {"formatted_output": {"reason": _SENTINEL}}, _REASON_WORKFLOW_FAILED, id="reason-outside-the-closed-set"
    ),
    pytest.param(
        {"formatted_output": {"reason": [_REASON_OUTPUT_WITHHELD]}}, _REASON_WORKFLOW_FAILED, id="reason-not-a-string"
    ),
    pytest.param({"formatted_output": {}}, _REASON_WORKFLOW_FAILED, id="empty-dict-formatted-output"),
]


class TestInvokeErrorEnvelope:
    @pytest.mark.parametrize("overrides, reason", _NON_SUCCESS_STATES)
    def test_error_values_are_drawn_from_the_declared_constants(self, overrides, reason):
        out = FoodLabelAllergenComplianceAgent().get_output(_finalize_state(**overrides))

        assert out["status"] != AgentStatus.SUCCESS.value
        assert set(out["error"]) == {"reason"}
        assert out["error"]["reason"] in ERROR_REASONS
        assert out["error"] == error_envelope(reason)
        assert out["error"], "the error envelope must stay truthy"
        assert out["output"] is None
        assert "error_log" not in out
        for key in _STRUCTURED_KEYS:
            assert key not in out, key

    @pytest.mark.parametrize("overrides, reason", _NON_SUCCESS_STATES)
    def test_internal_text_appears_nowhere_in_the_invoke_envelope(self, overrides, reason):
        out = FoodLabelAllergenComplianceAgent().get_output(_finalize_state(**overrides))
        _assert_internal_text_absent(out)

    @pytest.mark.parametrize("overrides, reason", _NON_SUCCESS_STATES)
    def test_the_envelope_keys_are_a_fixed_set(self, overrides, reason):
        out = FoodLabelAllergenComplianceAgent().get_output(_finalize_state(**overrides))
        assert set(out) == _BASE_ENVELOPE_KEYS | {"error"}

    def test_success_envelope_is_unchanged_and_never_carries_error_log(self):
        # The internal channel is NOT empty on this success run — it is still
        # not projected, on any status.
        state = _finalize_state(
            status=AgentStatus.SUCCESS.value,
            formatted_output=_CLEAN_ANSWER,
            result=_CLEAN_ANSWER,
            allergen_screening_result=_CLEAN_ANSWER,
            screening_status=to_json([{"id": "kb-mand-001", "status": "disclosed"}]),
            citations=to_json([{"ref": 1, "id": "kb-mand-001"}]),
        )
        out = FoodLabelAllergenComplianceAgent().get_output(state)

        assert out["status"] == AgentStatus.SUCCESS.value
        assert out["output"] == _CLEAN_ANSWER
        assert out["screening_status"] == [{"id": "kb-mand-001", "status": "disclosed"}]
        assert out["citations"] == [{"ref": 1, "id": "kb-mand-001"}]
        assert out["overall_abstain"] is False
        assert "error" not in out
        assert "error_log" not in out
        assert set(out) == _BASE_ENVELOPE_KEYS | set(_STRUCTURED_KEYS)
        _assert_internal_text_absent(out)

    def test_the_probe_finds_the_sentinel_where_it_lives(self):
        # Verify the verifier: the same walk DOES find every marker in a mapping
        # that carries it, so the "nowhere" assertions above are not vacuous.
        carrier = {"error_log": list(_INTERNAL_ENTRIES), "nested": {"deep": [{"k": _SENTINEL}]}}
        assert set(_found(carrier, *_INTERNAL_MARKERS)) == set(_INTERNAL_MARKERS)


# ── PostProcessNode: formatted_output on every path that returns ERROR ────────


def _post_state(**overrides) -> dict:
    state = {
        "status": AgentStatus.SUCCESS.value,
        "result": _CLEAN_ANSWER,
        "allergen_screening_result": _CLEAN_ANSWER,
        # The internal channel already carries the sentinel when the gate runs.
        "error_log": list(_INTERNAL_ENTRIES),
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "session_id": "closed-set-session",
        "trace_id": "closed-set-trace",
        "correlation_id": "closed-set-test",
        "node_history": [],
        "execution_time": {},
    }
    state.update(overrides)
    return state


def _drive(state: dict, entry: str) -> dict:
    node = PostProcessNode()
    return node.execute(state) if entry == "execute" else node(state)


# One offending value per output-gate rule, each built at runtime so no
# credential-shaped literal sits in the repository.
_GATE_RULES = [
    pytest.param("api_key", "sk-" + "A1b2C3d4E5f6G7h8I9j0", id="api_key"),
    pytest.param("jwt", "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12, id="jwt"),
    pytest.param("bearer_token", "Bearer " + "a" * 24, id="bearer_token"),
    pytest.param("credential_assignment", "password=" + "x" * 12, id="credential_assignment"),
    pytest.param("pii_email", _FRAGMENTS[1], id="pii_email"),
    pytest.param("pii_long_digit_run", "1234567890123", id="pii_long_digit_run"),
]
_ENTRY_POINTS = ["execute", "call"]


class TestPostProcessErrorEnvelope:
    @pytest.mark.parametrize("entry", _ENTRY_POINTS)
    @pytest.mark.parametrize("rule, offending", _GATE_RULES)
    def test_gate_refusal_carries_the_declared_reason_only(self, entry, rule, offending):
        result = _drive(_post_state(result=_CLEAN_ANSWER + "\ninternal note: " + offending), entry)

        assert result["status"] == AgentStatus.ERROR.value
        envelope = result["formatted_output"]
        assert set(envelope) == {"reason"}
        assert envelope["reason"] in ERROR_REASONS
        assert envelope == error_envelope(_REASON_OUTPUT_WITHHELD)
        assert envelope, "the envelope must stay truthy — the base envelope selects formatted_output or result"
        assert result["result"] is None

    @pytest.mark.parametrize("entry", _ENTRY_POINTS)
    @pytest.mark.parametrize("rule, offending", _GATE_RULES)
    def test_gate_message_stays_on_the_internal_channel_and_names_the_rule_only(self, entry, rule, offending):
        result = _drive(_post_state(result=_CLEAN_ANSWER + "\ninternal note: " + offending), entry)

        # One line, this node's own, naming the rule — never the matched value.
        assert result["error_log"] == [f"PostProcessNode: output blocked - disallowed content detected ({rule})"]
        assert offending not in json.dumps(result, ensure_ascii=False, default=str)
        assert not any("output blocked" in text for text in _strings(result["formatted_output"]))

    @pytest.mark.parametrize("entry", _ENTRY_POINTS)
    @pytest.mark.parametrize("rule, offending", _GATE_RULES)
    def test_seeded_internal_text_is_not_re_emitted(self, entry, rule, offending):
        result = _drive(_post_state(result=_CLEAN_ANSWER + "\ninternal note: " + offending), entry)
        # The entries already in error_log are not re-emitted (the state
        # reducer appends, so they would be duplicated) and appear nowhere in
        # the delta.
        assert _found(result, *_INTERNAL_MARKERS) == [], result

    def test_already_errored_state_is_contained_not_gated_into_a_success(self):
        """Direct execute() on an already-errored state (the backbone routes an
        errored run straight to finalize and the framework pipeline skips
        execute() on it, so this is the only way in): the node must not
        fabricate a success from a surviving pre-gate result, and it carries
        the constant reason code only."""
        result = PostProcessNode().execute(_post_state(status=AgentStatus.ERROR.value, result=_SENTINEL))

        assert result["status"] == AgentStatus.ERROR.value
        assert result["formatted_output"] == error_envelope(_REASON_WORKFLOW_FAILED)
        assert result["formatted_output"]
        assert result["result"] is None
        assert "error_log" not in result
        _assert_internal_text_absent(result)

    def test_the_framework_pipeline_never_runs_execute_on_an_errored_state(self, monkeypatch):
        """node(state) with an errored incoming state: BaseNode.__call__
        short-circuits before execute(), so no gate audit event fires and no
        formatted_output is authored — the state it hands back is a
        graph-internal partial update, never the caller's envelope."""
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.post_process_node, "emit_trace_event", spy)
        result = PostProcessNode()(_post_state(status=AgentStatus.ERROR.value, result=_SENTINEL))

        assert result["status"] == AgentStatus.ERROR.value
        assert "formatted_output" not in result
        assert spy.call_args_list == []

    @pytest.mark.parametrize("rule, offending", _GATE_RULES)
    def test_the_block_audit_event_carries_the_reason_and_the_rule_only(self, monkeypatch, rule, offending):
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.post_process_node, "emit_trace_event", spy)
        PostProcessNode().execute(_post_state(result=_CLEAN_ANSWER + "\ninternal note: " + offending))

        events = {call.args[0]: call.args[1] for call in spy.call_args_list}
        assert events["post_process_output_blocked"] == {"reason": _REASON_OUTPUT_WITHHELD, "violation": rule}
        assert "post_process_complete" not in events
        for payload in events.values():
            assert offending not in json.dumps(payload, ensure_ascii=False, default=str)
            assert _found(payload, *_MARKERS) == []

    def test_a_sentinel_that_reaches_the_answer_is_refused_without_echo(self):
        # The sentinel's e-mail fragment trips the gate's PII rule: an upstream
        # body that made it into the rendered answer is withheld, and the
        # refusal itself carries none of it.
        result = PostProcessNode()(_post_state(result=_CLEAN_ANSWER + "\ncontact " + _SENTINEL))
        assert result["formatted_output"] == error_envelope(_REASON_OUTPUT_WITHHELD)
        assert result["error_log"] == ["PostProcessNode: output blocked - disallowed content detected (pii_email)"]
        _assert_internal_text_absent(result)

    def test_clean_result_is_stamped_and_records_no_reason(self):
        result = PostProcessNode()(_post_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        assert isinstance(result["formatted_output"], str)
        assert result["formatted_output"].startswith(_CLEAN_ANSWER)
        assert "error_log" not in result


# ── The envelope builder itself ───────────────────────────────────────────────


class TestErrorEnvelopeBuilder:
    def test_the_reason_vocabulary_is_exactly_two_constants(self):
        assert ERROR_REASONS == frozenset({"workflow_failed", "output_withheld"})

    def test_every_declared_reason_builds_a_truthy_single_key_envelope(self):
        for reason in ERROR_REASONS:
            envelope = error_envelope(reason)
            assert envelope
            assert envelope == {"reason": reason}

    def test_a_reason_outside_the_closed_set_is_refused_and_not_echoed(self):
        with pytest.raises(ValueError) as info:
            error_envelope(_SENTINEL)
        assert _SENTINEL not in str(info.value)
