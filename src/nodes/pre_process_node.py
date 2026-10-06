"""AgentCore Platform v1.0"""

# RET-C2-283 — PreProcessNode
# Outer backbone pre_process slot: caller trust gate + label-request screening.
#
# Caller-data contract:
#   The structured label request travels on the input_context channel
#   (`input_context["label_request"]`), NOT inside the input text. The
#   platform input gate masks Title-Case name-like runs in user_input before
#   execute() ever sees it, so English label text embedded there arrives
#   with ingredient names such as "Whole Milk Powder" or "Cashew Nut"
#   already rewritten — the screening would then falsely report a disclosed
#   allergen as missing. input_context is not rewritten by the platform, so
#   this node owns ALL screening for it:
#
#   - structural checks (object shape, known fields only, length caps)
#   - chat-template control-token screen over every string, keys included,
#     at any nesting depth (post-parse, so JSON escape tricks do not bypass it)
#   - finite+bounded parsing for every caller number (fail closed)
#   - high-precision personal-data masking on the label text (a
#     manufacturer's contact line is legitimate on a label, so those spans
#     are masked rather than the request refused; ingredient/product names
#     are never masked here — that is the point of this channel)
#
#   Rejections name the offending field and never echo the rejected value.
#
#   Plain-text callers remain supported: a request without a label_request
#   record degrades to the legacy path where the input text itself is the
#   label text (Japanese labels are unaffected by the Title-Case heuristic).
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import json
import logging
from typing import Any, ClassVar, Dict

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.failure_message import INPUT_REJECTED
from src.services.progress import emit_progress
from src.nodes.validation import (
    bounded_label_text,
    find_injection_token,
    finite_in_range,
    mask_personal_data,
)
from src.schemas.state import to_json

logger = logging.getLogger(__name__)

# Hard cap on the label text length (defence-in-depth on input size; matches
# the inner parse cap).
MAX_LABEL_TEXT_CHARS = 4000

# Bounds for the caller-supplied top_k override on the recommended-tier cap.
TOP_K_MIN = 1
TOP_K_MAX = 20

# The only fields a label_request may carry — anything else is rejected by
# name (the unknown field's own name is never echoed back).
_KNOWN_REQUEST_FIELDS = frozenset({"label_text", "include_recommended", "top_k"})


def _error(
    reason: str, detail: Dict[str, Any], state: AgentState, message: str, code: str = "INVALID_REQUEST"
) -> Dict[str, Any]:
    """Audit + stop the request here, naming the field and never echoing its value.

    Two ways to stop, and the caller can act on only one of them. A value the
    caller can correct completes the run carrying `code`, so the reason reaches
    the caller and a corrected request can be sent on the same conversation.
    Content the agent refuses outright passes ``code=""`` and terminates, so a
    refusal is never presented as something a reworded request would get past.
    """
    logger.warning("PreProcessNode: %s", message)
    emit_trace_event("pre_process_validation_failed", {"reason": reason, **detail}, state)
    if code:
        # A value the caller can correct: the run COMPLETES carrying the
        # reason so the request can be sent again on the same conversation.
        emit_progress(INPUT_REJECTED)
        return {
            "status": AgentStatus.SUCCESS.value,
            "error_code": code,
            "error_log": [f"PreProcessNode: {message}"],
        }
    return {
        "status": AgentStatus.ERROR.value,
        "error_log": [f"PreProcessNode: {message}"],
    }


class PreProcessNode(FunctionNode):
    """Caller-contract screening for RET-C2-283.

    This is the outer backbone's pre_process slot — the only entry node with
    VERIFIED_EXTERNAL trust, so unauthenticated callers are rejected here
    (fail-fast; inner domain nodes never see unscreened caller data).

    Input state keys:
        user_input:    str  — caller text (label text on the legacy path,
                              instruction text on the structured path)
        input_context: dict — {"label_request": {...}} structured record
                              (optional; validated here when present)

    Output state keys (partial dict):
        screening_request: str       — screened, normalised JSON string
                                       (only when a label_request was sent)
        validated_input:   str       — screened caller text
        enriched_context:  str       — JSON-serialised channel metadata
        status:            str       — success or error
        error_code:        str       — set when the run COMPLETES without
                                       screening, because the caller can
                                       correct the value and send the request
                                       again
        error_log:         list[str] — the internal reason, on either stop path
    """

    # Outer caller gate — matches the manifest's declared required_trust_level.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        input_context = state.get("input_context") or {}

        # ── Caller text: required, bounded ────────────────────────────────────
        if not isinstance(user_input, str) or not user_input.strip():
            return _error("empty_input", {}, state, "user_input is empty or missing", code="EMPTY_INPUT")
        if len(user_input) > MAX_LABEL_TEXT_CHARS:
            return _error(
                "input_too_long",
                {"limit": MAX_LABEL_TEXT_CHARS},
                state,
                f"user_input must be at most {MAX_LABEL_TEXT_CHARS} characters",
                code="QUESTION_TOO_LONG",
            )
        # The platform input gate already masks personal data and rejects
        # high-confidence injection content on user_input; the template still
        # owns its own refusal for control-token forms (never framework-only).
        #
        # This one terminates. The checks above complete carrying a reason
        # because the caller can correct the value; a refusal is not a value to
        # correct, and reporting it the same way would read as an invitation to
        # reword the request until it is accepted.
        if find_injection_token(user_input) is not None:
            emit_trace_event("pre_process_injection_blocked", {"reason": "control_token_in_text"}, state)
            logger.warning("PreProcessNode: control token found in user_input")
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["PreProcessNode: user_input contains a disallowed control sequence"],
            }
        validated_input = user_input.strip()

        # ── Structured record: optional, screened when present ────────────────
        if not isinstance(input_context, dict):
            return _error("input_context_not_object", {}, state, "input_context must be a JSON object")
        payload = input_context.get("label_request")
        out: Dict[str, Any] = {}
        if payload is not None:
            if not isinstance(payload, dict):
                return _error("label_request_not_object", {}, state, "label_request must be a JSON object")

            # Control-token screen: every string, keys included, any depth.
            # Terminal, for the same reason as the text channel above: a refusal
            # is not a value the caller can correct and send again.
            token = find_injection_token(payload)
            if token is not None:
                # The matched token goes to the audit trail only — never the caller.
                emit_trace_event(
                    "pre_process_injection_blocked",
                    {"reason": "control_token", "token": token},
                    state,
                )
                logger.warning("PreProcessNode: control token found in label_request")
                return {
                    "status": AgentStatus.ERROR.value,
                    "error_log": ["PreProcessNode: label_request contains a disallowed control sequence"],
                }

            # Known fields only — a stray/hostile field name is rejected without echo.
            unknown = set(payload.keys()) - _KNOWN_REQUEST_FIELDS
            if unknown:
                return _error(
                    "unknown_request_fields",
                    {"count": len(unknown)},
                    state,
                    "label_request contains unsupported fields " "(allowed: label_text, include_recommended, top_k)",
                )

            # label_text: required on this channel, bounded printable text.
            label_text, err = bounded_label_text(
                payload.get("label_text"),
                "label_request.label_text",
                MAX_LABEL_TEXT_CHARS,
                required=True,
            )
            if err:
                return _error("invalid_label_text", {}, state, err)

            # Personal-data hygiene, on par with the platform's text channel:
            # mask unambiguous personal-data spans, keep the rest of the label.
            label_text, masked_types = mask_personal_data(label_text)
            if masked_types:
                emit_trace_event("pre_process_personal_data_masked", {"types": masked_types}, state)

            # include_recommended: optional strict boolean (fail closed).
            include_recommended = payload.get("include_recommended")
            if include_recommended is not None and not isinstance(include_recommended, bool):
                return _error(
                    "invalid_include_recommended",
                    {},
                    state,
                    "label_request.include_recommended must be a boolean",
                )

            # top_k: optional finite+bounded integer (fail closed — NaN and
            # Infinity parse as floats and must never reach a comparison).
            top_k, err = finite_in_range(
                payload.get("top_k"),
                "label_request.top_k",
                TOP_K_MIN,
                TOP_K_MAX,
                integral=True,
                optional=True,
            )
            if err:
                return _error("invalid_top_k", {}, state, err)

            out["screening_request"] = json.dumps(
                {
                    "label_text": label_text,
                    "include_recommended": include_recommended,
                    "top_k": top_k,
                },
                ensure_ascii=False,
            )

        # ── Success ───────────────────────────────────────────────────────────
        emit_trace_event(
            "pre_process_complete",
            {
                "input_chars": len(validated_input),
                "has_label_request": "screening_request" in out,
            },
            state,
        )

        out.update(
            {
                "validated_input": validated_input,
                "enriched_context": to_json(
                    {
                        "source": "FoodLabelAllergenComplianceAgent",
                        "channel": str(input_context.get("channel", "unknown"))[:64],
                    }
                ),
                "status": AgentStatus.SUCCESS.value,
            }
        )
        return out
