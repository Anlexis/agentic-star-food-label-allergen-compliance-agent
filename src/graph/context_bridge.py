"""AgentCore Platform v1.0"""

# src/graph/context_bridge.py — carries the screened label request across
# the outer-to-inner graph boundary.
#
# Why this exists: GraphNode.execute() (framework, SDK 1.0.1) invokes the
# inner graph as `subgraph.invoke(user_input, session_id=..., ctx=...)`
# WITHOUT forwarding the outer state's structured caller data, so inner-node
# reads of the screened label request would always see nothing through the
# full nested graph. The sanctioned subclass hooks bridge it:
#
#   AllergenScreeningGraphNode.extract_input(state)  [runs BEFORE subgraph.invoke]
#       -> set_label_request(state["screening_request"])
#   DomainWorkflowGraph._extra_initial_state()       [runs INSIDE subgraph.invoke]
#       -> returns {"screening_request": get_label_request()}
#
# A ContextVar keeps the hand-off correct per thread/task, so concurrent
# invocations in one process cannot see each other's request.

from contextvars import ContextVar
from typing import Optional

_LABEL_REQUEST: ContextVar[Optional[str]] = ContextVar("ret_c2_283_label_request", default=None)


def set_label_request(label_request_json: Optional[str]) -> None:
    """Stash the screened label request (JSON string) for the imminent inner-graph invoke."""
    _LABEL_REQUEST.set(label_request_json)


def get_label_request() -> Optional[str]:
    """Read (without consuming) the stashed label request; None when none was set."""
    return _LABEL_REQUEST.get()
