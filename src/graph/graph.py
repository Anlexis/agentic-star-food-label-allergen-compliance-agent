"""AgentCore Platform v1.0"""

# RET-C2-283 - Outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Food Label Allergen Compliance Agent (Cat 2 RAG advisory-screening workflow).
#
# Architecture (Cat 2):
#
#   Outer backbone (fixed - identical to Cat 1, do NOT override add_edges()):
#     START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
#                                             |  (RETRY, up to the configured max_retry)
#                                             -> pre_process
#
#   `main` slot is a GraphNode subclass (AllergenScreeningGraphNode) that
#   delegates the full allergen-screening domain workflow to
#   DomainWorkflowGraph (inner BaseGraph: input_validate -> retrieve ->
#   rerank_filter -> generate_answer -> output_format).
#
#   Domain complexity is fully encapsulated inside the inner graph. The outer
#   backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (multi-step topology)
#   src/graph/context_bridge.py        <- outer-to-inner caller-data bridge
#
# Class-name contract:
#   graph.py class:           FoodLabelAllergenComplianceAgent (this file)
#   config/agent.yaml class:  "src.graph.graph.FoodLabelAllergenComplianceAgent"
#   src/api/server.py import: from src.graph.graph import Graph, _runtime_config
#
# Runtime configuration:
#   Declared runtime parameters (max_retry, timeout_s, retrieval.*, llm.*)
#   live in config/config.yaml and are passed as Graph(config=...) by the
#   platform registry; the standalone server mirrors that construction via
#   _runtime_config(). register_nodes() hands the same dict to the main-slot
#   node so the inner graph receives the declared retrieval/llm tuning on
#   every construction path — a declared value is live, never a dead entry.
#
# Structured product: get_output() is overridden below to surface the
# per-allergen tier-tagged screening_status + citations + overall_abstain to
# a programmatic caller, on SUCCESS only, fail-closed (re-scanned via the
# same recursive _security_gate_output() scan PostProcessNode uses).
#
# Error contract: on any non-success status get_output() publishes closed-set
# labels only - `output: None` and `error: {"reason": <constant>}` (the
# reason vocabulary PostProcessNode declares). `error_log` is node- and
# framework-authored text (a wrapped exception carries its message and a
# traceback there) and is NEVER projected into the invoke body; it stays the
# internal channel the state reducer appends to and the audit trail reads.

import os
from typing import Any, ClassVar, Dict, Optional

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import set_label_request
from src.nodes.post_process_node import (
    ERROR_REASONS,
    PostProcessNode,
    _REASON_WORKFLOW_FAILED,
    _security_gate_output,
    error_envelope,
)
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json

# Runtime parameters — config/config.yaml at the repo root (three levels up
# from src/graph/graph.py).
_RUNTIME_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "config.yaml",
)

# Fallbacks mirror the `retrieval` / `llm` blocks in config/config.yaml so
# _parent_config() never forwards an empty config even if the file is
# unreadable in an exotic deployment layout.
_FALLBACK_RETRIEVAL = {
    "score_threshold": 0.75,
    "top_k": 10,
    "kb_path": "config/kb/allergen_kb.json",
    "abstain_min_chars": 8,
}
_FALLBACK_LLM = {
    "temperature": 0.0,
    "max_tokens": 1500,
}

# Non-suppressible advisory notice surfaced on the STRUCTURED payload - a
# constant, never derived from gateable content, so it cannot be withheld by
# any input (mirrors the text-side stamp PostProcessNode appends).
_ADVISORY_NOTICE = "advisory only — human review required"


def _runtime_config() -> Dict[str, Any]:
    """Return the runtime parameters from config/config.yaml.

    This is the same file the platform registry loads and passes as
    Graph(config=...). Loading is best-effort: a missing or unparseable file
    yields ``{}`` so graph construction never breaks (the framework and the
    inner nodes then fall back to their declared defaults). PyYAML is loaded
    lazily — it is a framework runtime dependency, so importing it on demand
    avoids a hard module-load coupling.
    """
    try:
        import yaml

        with open(_RUNTIME_CONFIG_PATH, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


class AllergenScreeningGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of the outer agent.

    Wraps DomainWorkflowGraph (inner Cat 2 BaseGraph advisory-screening pipeline).
    Called by AgentBaseGraph backbone after pre_process and before post_process.

    Contracts:
      get_subgraph()    - instantiate DomainWorkflowGraph with the forwarded
                          runtime config (_parent_config())
      extract_input()   - bridge the screened label request to the inner graph
                          (see context_bridge.py) and hand over the screened
                          caller text as the inner input string
      merge_output()    - map sub_result fields into outer state delta (changed keys only)
      error_strategy    - "handle": an inner failure fails the invocation
                          CLOSED while surfacing the inner nodes' field-naming
                          error messages (see on_subgraph_error) instead of a
                          raw exception trace
    """

    # "handle": route inner failures to on_subgraph_error() — the invocation
    # fails closed with clean, caller-actionable messages; a raw traceback
    # (which carries runtime paths) never crosses the graph boundary.
    error_strategy: ClassVar[str] = "handle"

    # False: HITL interrupts are handled inside the inner graph only (this
    # template has no HITL path - config/config.yaml has no hitl block).
    propagate_hitl: ClassVar[bool] = False

    def __init__(self, parent_config: Optional[Dict[str, Any]] = None) -> None:
        """Store the outer graph's runtime config for forwarding to the inner graph.

        The outer graph's register_nodes() passes its own config (from
        Graph(config=...) or the config/config.yaml fallback) so the inner
        DomainWorkflowGraph receives the declared runtime parameters through
        one path regardless of how the agent was constructed.
        """
        super().__init__()
        self._parent_config_dict: Dict[str, Any] = parent_config or {}

    def _parent_config(self) -> Dict[str, Any]:
        """Forward the declared `retrieval` + `llm` blocks to the inner graph.

        Returns the two tuning blocks under config["configurable"] - never an
        empty dict. The inner graph republishes the `retrieval` block into
        inner state (DomainWorkflowGraph._extra_initial_state()) so
        RetrieveNode / RerankFilterNode read live score_threshold / top_k /
        abstain_min_chars values instead of dead declarations. The `llm`
        block is forwarded verbatim for the documented LLM-synthesis upgrade
        (unused by the current deterministic nodes).
        """
        retrieval = self._parent_config_dict.get("retrieval")
        if not isinstance(retrieval, dict) or not retrieval:
            retrieval = dict(_FALLBACK_RETRIEVAL)
        llm = self._parent_config_dict.get("llm")
        if not isinstance(llm, dict) or not llm:
            llm = dict(_FALLBACK_LLM)
        return {"configurable": {"retrieval": retrieval, "llm": llm}}

    def get_subgraph(self) -> Any:
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported lazily (inside the method) to avoid
        circular-import risk at module load time and to match the Cat 2
        sample pattern.

        The inner graph receives the forwarded config via its BaseGraph
        ctor; its domain NODES still take no constructor arguments and read
        config exclusively via State seeding (execute(self, state) -> dict —
        no config parameter).
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def execute(self, state: AgentState) -> dict[str, Any]:
        """Skip the inner graph when the request was already found unacceptable.

        A request declined by pre_process has no validated input to act on, so
        running the inner graph would only produce a second, vaguer reason for
        the same rejection - and overwrite the specific one already settled.
        """
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        result: dict[str, Any] = super().execute(state)
        return result

    def extract_input(self, state: AgentState) -> str:
        """Bridge the screened label request, then return the inner input string.

        The framework invokes the inner graph as subgraph.invoke(user_input,
        session_id=..., ctx=...) without forwarding structured caller data,
        so the screened record (written to state by PreProcessNode) is
        stashed in a ContextVar here and read back by
        DomainWorkflowGraph._extra_initial_state() inside the same
        invocation. The string handed to the inner graph is the screened
        caller text — on the legacy path it IS the label text; on the
        structured path the label text rides the bridge instead, where
        name-like runs are not subject to rewriting by the platform's input
        gate.
        """
        set_label_request(state.get("screening_request"))
        return str(state.get("validated_input") or state.get("user_input", ""))

    def on_subgraph_error(self, state: AgentState, error: Exception) -> Dict[str, Any]:
        """Fail closed on an inner-graph failure, carrying clean messages.

        A domain validation rejection inside DomainWorkflowGraph names the
        offending field in the inner error_log. Carry those messages onto
        the outer error_log — the internal channel the audit trail reads
        (get_output() never projects it to the caller); never carry a raw
        traceback (it carries runtime paths — multi-line entries are dropped
        for exactly that reason). The status stays ERROR — no partial
        screening result is ever produced.
        """
        from framework.errors import SubgraphError

        messages: list[str] = []
        if isinstance(error, SubgraphError):
            seen = set()
            for message in error.error_log or []:
                if isinstance(message, str) and "\n" not in message and message not in seen:
                    seen.add(message)
                    messages.append(message)
        if not messages:
            messages = ["AllergenScreeningGraphNode: allergen screening workflow failed"]
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": messages,
        }

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map inner graph sub_result back into the outer state delta.

        sub_result is the dict returned by DomainWorkflowGraph.get_output().
        Returns ONLY changed keys - never the full state.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          Inner get_output() emits  -> "formatted_answer", "citations",
                                       "screening_status", "overall_abstain",
                                       "status", ...
          This merge_output() reads -> sub_result.get("formatted_answer"),
                                       sub_result.get("citations"),
                                       sub_result.get("screening_status"),
                                       sub_result.get("overall_abstain"),
                                       sub_result.get("status")

        allergen_screening_result (str | None): final rendered advisory
          result; written by OutputFormatNode inside the inner graph.
        result: PostProcessNode (outer post_process slot) reads
          state.get("result") - the inner graph emits the rendered result
          under "formatted_answer", so map it to "result" as well; otherwise
          the final output surfaced by PostProcessNode (and the output gate)
          is always empty.
        status (str | None): terminal status value from the inner graph run.
        """
        return {
            # Outer reason wins: a reason settled before the inner run is the real
            # one, and a plain sub_result.get() would erase it.
            "error_code": state.get("error_code") or sub_result.get("error_code", ""),
            "allergen_screening_result": sub_result.get("formatted_answer"),
            "result": sub_result.get("formatted_answer"),
            "citations": sub_result.get("citations"),
            "screening_status": sub_result.get("screening_status"),
            "overall_abstain": sub_result.get("overall_abstain"),
            "status": sub_result.get("status"),
        }


class FoodLabelAllergenComplianceAgent(AgentBaseGraph):
    """Outer graph for RET-C2-283 (Cat 2 RAG advisory screening).

    Inherits AgentBaseGraph directly (L1 Base - framework base class). Domain
    logic is fully encapsulated in AllergenScreeningGraphNode (main slot),
    which delegates to DomainWorkflowGraph (inner BaseGraph).

    Backbone (fixed - identical to Cat 1):
        START -> initialize -> pre_process -> main -> post_process -> finalize -> END

    register_nodes() and get_output() are the only overrides:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process:  PreProcessNode (caller trust gate + label-request screening)
      - main:         AllergenScreeningGraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode (output content gate + advisory stamp)

    add_edges() is NOT overridden - backbone wiring belongs to the framework.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with AgentRegistry."""
        return "FoodLabelAllergenComplianceAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first - it injects the
        framework's default InitializeNode (sets schema_version, session_id,
        trust_level) and FinalizeNode (builds response_metadata, total_time_ms).

        The runtime config handed to the main-slot node is self.config when
        the graph was constructed with one (the registry / server path),
        otherwise the config/config.yaml contents — so the declared runtime
        parameters reach the inner graph on every construction path.
        """
        super().register_nodes()  # fills: initialize, finalize

        runtime_config = self.config or _runtime_config()
        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = AllergenScreeningGraphNode(parent_config=runtime_config)
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden - backbone wiring belongs to the framework.

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Extend the base envelope with the structured screening product.

        base = {output, status, trace_id, correlation_id, node_history} from
        super(); on SUCCESS only, ADD screening_status / citations /
        overall_abstain / advisory_notice - never on a non-SUCCESS status
        (fail-closed: withhold the structured fields entirely rather than
        surface a partial/gated result).

        On any non-success status the caller receives closed-set labels only:
        ``output`` is None (the base envelope's ``formatted_output or result``
        fallback is not consulted) and ``error`` is ``error_envelope()`` — one
        constant reason code. ``output_withheld`` is PostProcessNode's own
        reason when its gate ran and refused; every other non-success outcome
        — a caller-data rejection, a trust denial, an inner-workflow error, a
        timeout, all of which the backbone routes straight to finalize — is
        ``workflow_failed``. A value in that slot outside the closed set is
        replaced, never echoed. ``error_log`` is never projected, on any
        status: it is the internal channel (state reducer, audit trail) and
        carries node- and framework-authored text, including a wrapped
        exception's message and traceback with runtime paths.

        Each surfaced structured field is built from explicit, whitelisted,
        KB-derived SCALAR values (ids, tier/status labels, scores, source
        strings) - never a raw forwarded dict - and is defensively re-scanned
        through the SAME recursive `_security_gate_output()` scan
        PostProcessNode uses before being returned, so a nested
        credential/PII leak cannot bypass the gate by riding the structured
        path instead of the text path.
        """
        base: Dict[str, Any] = dict(super().get_output(state))
        # A run that completed WITHOUT carrying out the request holds the
        # sentence saying what to correct, not a product: none of the
        # structured fields below were produced, so none is released.
        if state.get("error_code"):
            return base
        if state.get("status") != AgentStatus.SUCCESS.value:
            formatted = state.get("formatted_output")
            recorded = formatted.get("reason") if isinstance(formatted, dict) else None
            reason = recorded if isinstance(recorded, str) and recorded in ERROR_REASONS else _REASON_WORKFLOW_FAILED
            base["output"] = None
            base["error"] = error_envelope(reason)
            return base

        screening_status = from_json(state.get("screening_status"), []) or []
        citations = from_json(state.get("citations"), []) or []

        if _security_gate_output(screening_status) or _security_gate_output(citations):
            # Fail-closed: withhold the structured fields entirely on any
            # gate violation. The text-side output already carries the gate
            # verdict via PostProcessNode; nothing further is added here.
            return base

        base["screening_status"] = screening_status
        base["citations"] = citations
        base["overall_abstain"] = bool(state.get("overall_abstain", False))
        base["advisory_notice"] = _ADVISORY_NOTICE
        return base


# Back-compat alias - server.py imports the agent as Graph. The class name
# matches the config/agent.yaml entry point.
Graph = FoodLabelAllergenComplianceAgent
