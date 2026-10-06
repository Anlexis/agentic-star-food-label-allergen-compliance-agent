"""AgentCore Platform v1.0"""

# RET-C2-283 - RetrieveNode
# Domain node 2: deterministic keyword/synonym matching over the seeded
# allergen provisions knowledge base (config/kb/allergen_kb.json). v1 is
# fully deterministic - no embedding model, no vector store, no live OCR
# model call. The match contract (matched_provisions JSON) is store-agnostic
# so a later vector-store upgrade only swaps this node's internals.
#
# Matching rule (mirrors how 食品表示基準 alternative expressions / 代替表記
# work in practice):
#   - a KB entry's `explicit_terms` (canonical name + recognised alternative
#     expressions) found as a substring of the label text -> score 1.0
#     ("explicit" match - a confident disclosure signal)
#   - else a KB entry's `implicit_hint_terms` (processed/compound-food names
#     that conventionally imply the allergen, e.g. マヨネーズ implies egg)
#     found as a substring -> score 0.5 ("implicit" match - a soft signal;
#     RerankFilterNode buckets this as "ambiguous", never "disclosed")
#   - else score 0.0 ("none")
#
# MANDATORY-tier entries are always kept (even at score 0.0 - a compliance
# checklist needs a status for every mandatory allergen every time).
# RECOMMENDED-tier entries are kept only when score > 0 (disclosure of a
# recommended-tier item is not legally required, so an unmatched recommended
# item is simply omitted rather than reported "missing").
#
# Config: reads `score_threshold` / `top_k` / `kb_path` / `abstain_min_chars`
# from the state field retrieval_config (republished by
# DomainWorkflowGraph._extra_initial_state() from the declared runtime
# config, forwarded by AllergenScreeningGraphNode._parent_config()); falls
# back to module defaults that mirror config/config.yaml when the field is
# absent (e.g. this node invoked standalone in a boundary test).
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.schemas.agent_status import AgentStatus
from framework.nodes.function_node import FunctionNode
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json, to_json

# Defaults mirror the `retrieval` block in config/config.yaml.
_DEFAULT_RETRIEVAL: Dict[str, Any] = {
    "score_threshold": 0.75,
    "top_k": 10,
    "kb_path": "config/kb/allergen_kb.json",
    "abstain_min_chars": 8,
}

# Repo root: src/nodes/retrieve_node.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]

_EXPLICIT_SCORE = 1.0
_IMPLICIT_SCORE = 0.5
_NO_MATCH_SCORE = 0.0

# Excerpt length carried into matched_provisions (keeps State small).
_EXCERPT_CHARS = 280


def _resolve_retrieval_config(state: Dict[str, Any]) -> Dict[str, Any]:
    """Effective retrieval config: state retrieval_config > module defaults."""
    effective = dict(_DEFAULT_RETRIEVAL)  # local copy - never mutate the module default
    from_state = from_json(state.get("retrieval_config"), None)
    if isinstance(from_state, dict):
        effective.update(from_state)
    return effective


def _load_kb(kb_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load the seeded KB JSON. Missing / malformed file degrades gracefully."""
    notes: List[str] = []
    path = Path(kb_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        notes.append(f"RetrieveNode: allergen knowledge base not readable at {kb_path}.")
        return [], notes
    if not isinstance(entries, list):
        notes.append("RetrieveNode: allergen knowledge base root must be a JSON list.")
        return [], notes
    return [e for e in entries if isinstance(e, dict)], notes


def _match_entry(entry: Dict[str, Any], text: str) -> Tuple[float, Optional[str]]:
    """(score, matched_term) for one KB entry against the label text.

    Empty text never matches anything (a substring check against "" would
    otherwise spuriously match every entry).
    """
    if not text:
        return _NO_MATCH_SCORE, None
    for term in entry.get("explicit_terms", []) or []:
        if isinstance(term, str) and term and term in text:
            return _EXPLICIT_SCORE, term
    for term in entry.get("implicit_hint_terms", []) or []:
        if isinstance(term, str) and term and term in text:
            return _IMPLICIT_SCORE, term
    return _NO_MATCH_SCORE, None


class RetrieveNode(FunctionNode):
    """Match the label text against every seeded allergen provision.

    Input state keys:
        label_text:        normalised label/ingredient text (from InputValidateNode)
        retrieval_config:   forwarded runtime retrieval block (JSON)

    Output state keys (partial dict):
        matched_provisions: JSON list of per-allergen match candidates
        intake_notes:       (on KB anomalies) JSON list[str]
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        text = state.get("label_text") or state.get("validated_input") or state.get("user_input", "")
        text = text if isinstance(text, str) else ""

        retrieval_cfg = _resolve_retrieval_config(state)
        entries, notes = _load_kb(str(retrieval_cfg.get("kb_path", _DEFAULT_RETRIEVAL["kb_path"])))

        candidates: List[Dict[str, Any]] = []
        for entry in entries:
            tier = str(entry.get("tier", ""))
            score, matched_term = _match_entry(entry, text)
            if tier != "mandatory" and score <= _NO_MATCH_SCORE:
                continue  # recommended-tier: omit entirely when unmatched
            candidates.append(
                {
                    "id": str(entry.get("id", "")),
                    "allergen_en": str(entry.get("allergen_en", "")),
                    "allergen_ja": str(entry.get("allergen_ja", "")),
                    "tier": tier,
                    "source": str(entry.get("source", "")),
                    "score": score,
                    "matched_term": matched_term,
                    "excerpt": str(entry.get("content", ""))[:_EXCERPT_CHARS],
                }
            )

        # Deterministic ordering: mandatory tier first (id asc), then
        # recommended tier by score desc (id asc for stable ties).
        candidates.sort(key=lambda c: (0 if c["tier"] == "mandatory" else 1, -c["score"], c["id"]))

        # Domain audit: allergen-provision matching pass completed.
        emit_trace_event(
            "retrieve_complete",
            {
                "candidates": len(candidates),
                "kb_entries": len(entries),
                "text_chars": len(text),
            },
            state,
        )

        out: Dict[str, Any] = {"matched_provisions": to_json(candidates)}
        if notes:
            # Append to (never clobber) the notes accumulated upstream.
            prior = from_json(state.get("intake_notes"), []) or []
            out["intake_notes"] = to_json(list(prior) + notes)
        return out
