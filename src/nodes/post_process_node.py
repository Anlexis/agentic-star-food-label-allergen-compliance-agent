"""AgentCore Platform v1.0"""

# RET-C2-283 - PostProcessNode (outer post_process slot; output content gate)
#
# Reads the final advisory screening result from state["result"] (populated
# by AllergenScreeningGraphNode.merge_output(), mapped from the inner
# graph's formatted_answer), and surfaces it as the finalized output AFTER
# running the output content-safety gate.
#
# This node calls the MODULE-LEVEL `_security_gate_output()` scan from
# execute() itself. Unlike a generic top-level-string-only scan, this gate
# RECURSES into nested dict/list/tuple content (signature `content: Any`) -
# a nested/structured payload cannot smuggle a credential- or PII-shaped
# string past a scan that only ever looked at the top-level string. The
# same function is reused by FoodLabelAllergenComplianceAgent.get_output()
# (src/graph/graph.py) to defensively re-scan the structured payload
# (screening_status / citations) before it is surfaced to a programmatic
# caller - fail-closed on either leg. No _extra_security_gate_input/_output
# instance methods are defined on this node (the framework auto-wraps such
# hooks).
#
# SAFETY BOUNDARY (docs/02_design.md): every clean result this node returns
# carries the standing "advisory only - human review required" stamp,
# appended HERE (not by any inner domain node) so it is genuinely
# non-suppressible - it cannot be omitted by a domain node forgetting to add
# it, because every response passes through this one gate unconditionally.
# This node ALSO blocks PII-shaped content in the gated output (RET-C2-283
# domain requirement, in addition to the generic credential scan).
#
# ERROR CONTRACT: on every non-success path this node returns the closed-set
# error envelope (see `_contain` / `error_envelope` below) - a constant reason
# code the caller can read, never an error_log line, a gate message or an
# exception's text. `error_log` stays the internal audit channel.

import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, INVALID_VALUE, TOO_LONG

logger = logging.getLogger(__name__)

# Disallowed output content patterns.
# Each tuple: (name, compiled regex) - order matters (most specific first).
_DISALLOWED_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    # API key patterns: sk-..., pk-..., ak-...
    ("api_key", re.compile(r"\b(?:sk|pk|ak)-[A-Za-z0-9]{16,}", re.IGNORECASE)),
    # JWT: three base64url segments separated by dots
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    # Bearer token in Authorization-like context
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._~+/]{20,}", re.IGNORECASE)),
    # Credential assignment patterns
    (
        "credential_assignment",
        re.compile(
            r"\b(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
    # PII (RET-C2-283 domain requirement): e-mail address
    ("pii_email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # PII: a long digit run (phone / card / ID number), optionally grouped
    ("pii_long_digit_run", re.compile(r"\b\d[\d\- ]{9,18}\d\b")),
]

# The standing advisory stamp - genuinely non-suppressible because it is
# appended unconditionally on the one clean-result path through this gate,
# never by a domain node. The wording is part of the template's public contract.
_ADVISORY_STAMP = (
    "This screening result is advisory only — human review required. Confirm "
    "every flagged allergen status against the physical product label and "
    "manufacturer documentation before making any compliance determination. "
    "This tool does not issue a compliance verdict."
)

# ── Caller-visible ERROR envelope - closed-set labels only ────────────────────
#
# On every non-success path the caller receives values this module chose: one
# constant reason code, nothing else. Nothing read from `error_log`, from the
# gate's own violation message, or from any other node-authored string is
# projected - an entry there can carry a framework-wrapped exception message
# and traceback, or whatever the failing node was handed - and truncating or
# filtering such text is not a closed set; not publishing it is. `error_log`
# itself is untouched: it stays the internal channel the state reducer appends
# to and the audit trail reads.
#
# The envelope always carries its constant key, so it is always truthy:
# AgentBaseGraph.get_output() selects `formatted_output or result` with no
# status check, and a falsy envelope would re-open that fallback.
_REASON_WORKFLOW_FAILED = "workflow_failed"  # the run reported status=error before this gate ran
_REASON_OUTPUT_WITHHELD = "output_withheld"  # this node's output gate refused the response
ERROR_REASONS = frozenset({_REASON_WORKFLOW_FAILED, _REASON_OUTPUT_WITHHELD})


def error_envelope(reason: str) -> Dict[str, str]:
    """The caller-visible ERROR envelope: a constant reason code and nothing else."""
    if reason not in ERROR_REASONS:
        raise ValueError("error envelope reason must be one of ERROR_REASONS")
    return {"reason": reason}


def _contain(reason: str, new_errors: Optional[List[str]] = None) -> Dict[str, Any]:
    """Fail-closed ERROR result: the closed-set envelope, `result` cleared.

    `formatted_output` becomes the (truthy) envelope and `result` is cleared -
    the base envelope falls back to it for `output` - so nothing pre-gate can
    surface. `new_errors` (this node's own gate message, which names a
    violation TYPE and never a value) goes to `error_log`, the internal
    channel, never into the envelope. Entries already in `error_log` are not
    re-emitted: the state reducer appends, so they would be duplicated.
    """
    contained: Dict[str, Any] = {
        "formatted_output": error_envelope(reason),
        "result": None,
        "status": AgentStatus.ERROR.value,
    }
    if new_errors:
        contained["error_log"] = list(new_errors)
    return contained


def _security_gate_output(content: Any) -> Optional[str]:
    """Run the output content gate. RECURSES into dict/list/tuple.

    Returns the name of the first matched violation, or None if clean.
    Fail-closed but never raises: any leaf that is not a str/dict/list/tuple
    (numbers, bools, None) is not scannable and is treated as clean.
    """
    if content is None:
        return None
    if isinstance(content, str):
        for name, pattern in _DISALLOWED_PATTERNS:
            if pattern.search(content):
                return name
        return None
    if isinstance(content, dict):
        for value in content.values():
            violation = _security_gate_output(value)
            if violation:
                return violation
        return None
    if isinstance(content, (list, tuple)):
        for item in content:
            violation = _security_gate_output(item)
            if violation:
                return violation
        return None
    return None


# Reason code -> the sentence the caller reads. A code with no entry falls
# back to the generic one rather than leaking the code itself.
_DEGRADED_MESSAGES = {
    "EMPTY_INPUT": EMPTY_INPUT,
    "QUESTION_TOO_LONG": TOO_LONG,
    "INVALID_REQUEST": INVALID_VALUE,
}


class PostProcessNode(FunctionNode):
    """Output gate: scan the final advisory result, stamp, and finalize.

    Outer backbone post_process slot. Reads state["result"] (the merged
    formatted_answer from AllergenScreeningGraphNode.merge_output()) and
    applies the content-safety gate before the response is returned to
    the caller. The gate is the module-level `_security_gate_output()`
    above, called from execute().

    Input state keys:
        result: final formatted advisory result (from merge_output)

    Output state keys (partial dict):
        formatted_output: the advisory-stamped screening result on SUCCESS;
                          the fixed reason sentence (a string, same shape as
                          SUCCESS) when the run was declined upstream and
                          completes carrying a reason code;
                          the closed-set envelope {"reason": <code>} on ERROR
        error_code:       carried onward on the declined-completion path, so
                          get_output() withholds the structured product
        result:           unchanged on the clean/empty paths; None on ERROR
        status:           AgentStatus.SUCCESS.value or AgentStatus.ERROR.value
                          (plain strings — never write the bare enum to State)
        error_log:        (on a gate block only) the violation TYPE - the
                          internal channel, never part of the envelope
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # A run declined upstream has nothing to format. Render the reason as
        # the caller-facing body and carry the marker onward.
        marker = state.get("error_code")
        if marker:
            message = _DEGRADED_MESSAGES.get(marker, INPUT_REJECTED)
            emit_trace_event("post_process_degraded", {"reason": marker}, state)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": marker,
                "formatted_output": message,
            }
        # ── Already-errored state: contain, never fabricate a success ─────────
        # The backbone routes an errored run straight to finalize and the
        # framework pipeline skips execute() on an errored incoming state, so
        # on the compiled graph this node never sees one; a direct call still
        # returns the same closed-set envelope as every other error path
        # instead of gating a stale answer into a SUCCESS. The entries already
        # in error_log stay internal and are not re-emitted.
        if state.get("status") == AgentStatus.ERROR.value:
            return _contain(_REASON_WORKFLOW_FAILED)

        result = state.get("result") or ""

        if not result or not str(result).strip():
            # No answer was generated - forward as-is (non-fatal). No stamp:
            # there is no screening result to qualify.
            return {
                "formatted_output": result,
                "status": AgentStatus.SUCCESS.value,
            }

        violation = _security_gate_output(str(result))
        if violation:
            logger.error(
                "PostProcessNode: OUTPUT BLOCKED - violation type: %s",
                violation,
            )
            # Audit the block - outcome signals only: the reason code and the
            # violation TYPE (a name from _DISALLOWED_PATTERNS, never the value).
            emit_trace_event(
                "post_process_output_blocked",
                {"reason": _REASON_OUTPUT_WITHHELD, "violation": violation},
                state,
            )
            # The message names the violation TYPE and never the matched value.
            # It goes to error_log only - the internal channel; the caller
            # receives the reason code.
            return _contain(
                _REASON_OUTPUT_WITHHELD,
                [f"PostProcessNode: output blocked - disallowed content detected ({violation})"],
            )

        # Clean - append the non-suppressible advisory stamp.
        formatted_output = f"{result}\n\n---\n\n*{_ADVISORY_STAMP}*"

        # Domain audit: record that a finalized advisory result was emitted.
        emit_trace_event(
            "post_process_complete",
            {"output_chars": len(formatted_output)},
            state,
        )

        return {
            "formatted_output": formatted_output,
            "status": AgentStatus.SUCCESS.value,
        }
