"""AgentCore Platform v1.0"""

# RET-C2-283 - InputValidateNode
# Domain node 1: resolve and normalise the label/ingredient text for screening.
#
# Two supported input paths, in priority order:
#
#   1. screening_request (bridged structured record — preferred):
#      PreProcessNode screened input_context["label_request"] field-by-field
#      and the graph seeds the result into inner state (see
#      src/graph/context_bridge.py). This is the channel for English label
#      text: it is never rewritten by the platform's Title-Case masking, so
#      ingredient names like "Whole Milk Powder" survive to the matcher.
#
#   2. plain text (legacy path): the caller's input text is the label text.
#      A JSON envelope ({"text": ..., "include_recommended": ..., "top_k":
#      ...}) inside the text is still parsed for backward compatibility, with
#      the same fail-closed field validation as the structured channel.
#
# v1 accepts NORMALIZED TEXT ONLY - no image bytes, no OCR call. If the
# caller's label text was extracted from an image, that extraction is
# assumed to have already happened upstream of this agent.
#
# Caller-controlled values fail CLOSED: a malformed top_k or
# include_recommended rejects the request with a field-naming error (the
# value is never echoed); absent fields fall back to configured defaults.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import json
import re
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
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
from src.schemas.state import from_json, to_json

# Hard cap on the normalised label text length (defence-in-depth on input size).
_MAX_TEXT_CHARS = 4000

# Bounds for the caller-supplied top_k override on the recommended-tier cap.
_TOP_K_MIN = 1
_TOP_K_MAX = 20

_WHITESPACE_RE = re.compile(r"\s+")


def _reject(state: Dict[str, Any], message: str, code: str = "INVALID_REQUEST") -> Dict[str, Any]:
    """Stop the request here: no label text is published, and the field is named, never valued.

    Two ways to stop, and the caller can act on only one of them. A value the
    caller can correct completes the run carrying `code`, so the reason reaches
    the caller and a corrected request can be sent on the same conversation.
    Content the agent refuses outright passes ``code=""`` and terminates, so a
    refusal is never presented as something a reworded request would get past.
    """
    emit_trace_event("input_validate_rejected", {"message": message}, state)
    if code:
        # A value the caller can correct: the run COMPLETES carrying the
        # reason so the request can be sent again on the same conversation.
        emit_progress(INPUT_REJECTED)
        return {
            "status": AgentStatus.SUCCESS.value,
            "error_code": code,
            "error_log": [f"InputValidateNode: {message}"],
        }
    return {
        "status": AgentStatus.ERROR.value,
        "error_log": [f"InputValidateNode: {message}"],
    }


class InputValidateNode(FunctionNode):
    """Resolve the screened request into normalised label text + options.

    Input state keys:
        screening_request:            bridged, pre-screened JSON record (preferred)
        validated_input | user_input: caller text (legacy path)

    Output state keys (partial dict):
        label_text:    normalised label/ingredient text
        query_filters: JSON dict {"include_recommended": bool|None, "top_k": int|None}
        intake_notes:  (when anomalies were seen) JSON list[str]
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        notes: List[str] = []

        text = ""
        include_recommended: Optional[bool] = None
        top_k: Optional[int] = None

        bridged = from_json(state.get("screening_request"), None)
        if isinstance(bridged, dict):
            # Structured path: PreProcessNode already screened every field —
            # reads here stay defensive but re-screening is not repeated.
            raw_text = bridged.get("label_text")
            text = raw_text if isinstance(raw_text, str) else ""
            flag = bridged.get("include_recommended")
            include_recommended = flag if isinstance(flag, bool) else None
            top_k_val = bridged.get("top_k")
            if top_k_val is not None:
                parsed, err = finite_in_range(top_k_val, "top_k", _TOP_K_MIN, _TOP_K_MAX, integral=True, optional=True)
                if err:
                    return _reject(state, err)
                top_k = int(parsed) if parsed is not None else None
        else:
            raw = state.get("validated_input") or state.get("user_input", "")
            if isinstance(raw, str) and raw.strip():
                payload: Any = None
                stripped = raw.strip()
                if stripped.startswith("{"):
                    try:
                        payload = json.loads(stripped)
                    except (json.JSONDecodeError, ValueError):
                        notes.append(
                            "InputValidateNode: JSON-looking input did not parse - " "treated as plain label text."
                        )
                if isinstance(payload, dict):
                    # Legacy JSON envelope: same fail-closed contract as the
                    # structured channel, screened HERE because this envelope
                    # never passed through PreProcessNode's record screening.
                    if find_injection_token(payload) is not None:
                        # Terminal: a refusal, not a correctable value. Rewording the
                        # request must not be presented as a route past it.
                        return _reject(state, "input contains a disallowed control sequence", code="")
                    candidate = payload.get("text") or payload.get("label_text") or payload.get("ingredients") or ""
                    text, err = bounded_label_text(candidate, "text", _MAX_TEXT_CHARS)
                    if err:
                        return _reject(state, err)
                    flag = payload.get("include_recommended")
                    if flag is not None and not isinstance(flag, bool):
                        return _reject(state, "include_recommended must be a boolean")
                    include_recommended = flag
                    parsed, err = finite_in_range(
                        payload.get("top_k"),
                        "top_k",
                        _TOP_K_MIN,
                        _TOP_K_MAX,
                        integral=True,
                        optional=True,
                    )
                    if err:
                        return _reject(state, err)
                    top_k = int(parsed) if parsed is not None else None
                else:
                    if find_injection_token(stripped) is not None:
                        # Terminal: a refusal, not a correctable value. Rewording the
                        # request must not be presented as a route past it.
                        return _reject(state, "input contains a disallowed control sequence", code="")
                    text = stripped
                # Personal-data hygiene on the legacy path mirrors the
                # structured channel (the platform masks the raw input text,
                # but an envelope's inner text field is screened here).
                text, masked_types = mask_personal_data(text)
                if masked_types:
                    notes.append("InputValidateNode: personal-data patterns were masked " "from the label text.")
            else:
                notes.append("InputValidateNode: empty request - no label text to screen.")

        # Normalise whitespace and cap length.
        text = _WHITESPACE_RE.sub(" ", text).strip()
        if len(text) > _MAX_TEXT_CHARS:
            text = text[:_MAX_TEXT_CHARS]
            notes.append(f"InputValidateNode: label text truncated to {_MAX_TEXT_CHARS} chars.")

        filters = {"include_recommended": include_recommended, "top_k": top_k}

        # Domain audit: label text parsed and normalised.
        emit_trace_event(
            "input_validate_complete",
            {
                "text_chars": len(text),
                "used_structured_channel": isinstance(bridged, dict),
                "has_include_recommended_override": include_recommended is not None,
                "has_top_k_override": top_k is not None,
            },
            state,
        )

        out: Dict[str, Any] = {
            "label_text": text,
            "query_filters": to_json(filters),
        }
        if notes:
            out["intake_notes"] = to_json(notes)
        return out
