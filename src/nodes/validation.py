"""AgentCore Platform v1.0"""

# src/nodes/validation.py — shared caller-contract validation helpers.
#
# Every caller-controlled value that enters the screening pipeline goes
# through these helpers. The rules they enforce, in one place so every node
# applies the same contract:
#
#   * Numbers are FINITE and BOUNDED, and validation fails CLOSED.
#     float("nan") / float("inf") parse fine and arrive both via string
#     coercion and via raw JSON bodies (json.loads accepts bare NaN/Infinity),
#     but every IEEE-754 comparison against NaN is False — so an unchecked
#     non-finite value silently disables the exact bound it exists to drive.
#     Rejection names the field, never echoes the value.
#   * Label text is bounded printable text: no control characters other than
#     whitespace (ingredient panels are legitimately multi-line), explicit
#     length caps.
#   * Chat-template control tokens are refused as a CLASS, post-parse, in
#     values AND keys, at any nesting depth. JSON \u-escapes are already
#     decoded by the parser, so scanning the parsed object closes that
#     encoding bypass.
#   * High-precision personal-data patterns (card numbers, national IDs,
#     e-mail addresses, phone numbers) found in label text are MASKED, not
#     refused: a retail label legitimately carries a manufacturer's contact
#     line, and the screening needs the rest of the text. Masking mirrors
#     what the platform input gate does on the plain-text channel, so the
#     context channel gets the same personal-data hygiene. The broader
#     Title-Case name heuristic is deliberately NOT applied here: ingredient
#     and product names ("Whole Milk Powder", "Cashew Nut") are legitimately
#     Title-Case, and masking them destroys the allergen match itself — the
#     reason the structured channel exists (see docs/02_design.md).

import math
import re
from typing import Any, Iterator, List, Optional, Tuple, Union

from framework.security.pii_detector import detect_pii
from framework.security.pii_masking import mask_pii

# ── Chat-template / instruction control tokens (refused as a class) ───────────
# Token FORMS, not phrases: <|...|> chat-markup specials, [INST]-style
# instruction wrappers, and <<SYS>>-style system wrappers. Ordinary food-label
# prose ("system", "instructions for use", "保存方法") does not match.
_INJECTION_TOKEN_RE = re.compile(
    r"(<\|[a-zA-Z0-9_]{1,32}\|>)"  # <|im_start|>, <|endoftext|>, ...
    r"|(\[/?INST\])"  # [INST] ... [/INST]
    r"|(<</?SYS>>)",  # <<SYS>> ... <</SYS>>
)

# High-precision personal-data pattern types masked in label text. The
# heuristic "name" types are excluded on purpose: ingredient and product
# names are Title-Case by nature and are not personal data on a food label.
_MASKED_PII_TYPES = frozenset({"credit_card", "ssn_us", "my_number_jp", "email", "phone_jp", "phone_us"})

# Control characters except whitespace (tab/newline/CR are legitimate in a
# multi-line ingredient panel; they are whitespace-normalised downstream).
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

Number = Union[int, float]


def finite_in_range(
    value: Any,
    field: str,
    lo: Number,
    hi: Number,
    *,
    integral: bool = False,
    optional: bool = False,
) -> Tuple[Optional[Number], Optional[str]]:
    """Parse a caller-controlled number: finite, bounded, fail-closed.

    Returns (parsed, None) on success or (None, error) on rejection. The
    error names the field and the accepted range — never the supplied value.
    Rejects: bools, non-numerics, NaN, +/-Infinity, out-of-range magnitudes,
    and non-integral values where an integer is required.
    """
    if value is None:
        if optional:
            return None, None
        return None, f"{field} is required"
    if isinstance(value, bool):
        return None, f"{field} must be a number in [{lo}, {hi}]"
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value.strip())
        except (TypeError, ValueError):
            return None, f"{field} must be a number in [{lo}, {hi}]"
    else:
        return None, f"{field} must be a number in [{lo}, {hi}]"
    if not math.isfinite(parsed):
        return None, f"{field} must be a finite number in [{lo}, {hi}]"
    if not (float(lo) <= parsed <= float(hi)):
        return None, f"{field} must be within [{lo}, {hi}]"
    if integral:
        if parsed != int(parsed):
            return None, f"{field} must be an integer in [{lo}, {hi}]"
        return int(parsed), None
    return parsed, None


def bounded_label_text(
    value: Any,
    field: str,
    max_len: int,
    *,
    required: bool = False,
) -> Tuple[str, Optional[str]]:
    """Validate caller label text: printable (whitespace allowed), length-capped.

    Returns (text, None) or ("", error). The error names the field only.
    Empty text is accepted unless required=True. Newlines/tabs are allowed —
    ingredient panels are multi-line — and are whitespace-normalised by the
    parse step downstream; every other control character is refused.
    """
    if value is None:
        value = ""
    if not isinstance(value, str):
        return "", f"{field} must be a string of at most {max_len} characters"
    text = value.strip()
    if not text:
        if required:
            return "", f"{field} is required"
        return "", None
    if len(text) > max_len:
        return "", f"{field} must be at most {max_len} characters"
    if _CONTROL_CHARS_RE.search(text):
        return "", f"{field} must not contain control characters"
    return text, None


def _iter_strings(obj: Any) -> Iterator[str]:
    """Yield every string in a parsed JSON object — values AND keys, any depth."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for key, val in obj.items():
            if isinstance(key, str):
                yield key
            yield from _iter_strings(val)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_strings(item)


def find_injection_token(obj: Any) -> Optional[str]:
    """Scan a parsed payload for chat-template control tokens (keys included).

    Returns the matched token (for the audit trail — never echoed to the
    caller) or None when the payload is clean.
    """
    for text in _iter_strings(obj):
        match = _INJECTION_TOKEN_RE.search(text)
        if match:
            return match.group(0)
    return None


def mask_personal_data(text: str) -> Tuple[str, List[str]]:
    """Mask high-precision personal-data patterns in label text.

    Returns (masked_text, sorted list of masked pattern types). Only the
    unambiguous pattern types are masked (see _MASKED_PII_TYPES); the
    Title-Case name heuristic is deliberately not applied so ingredient and
    product names survive intact.
    """
    findings = [f for f in detect_pii(text) if f["type"] in _MASKED_PII_TYPES]
    if not findings:
        return text, []
    return mask_pii(text, findings), sorted({f["type"] for f in findings})
