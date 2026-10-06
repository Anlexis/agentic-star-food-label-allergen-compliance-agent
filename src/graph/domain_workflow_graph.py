"""AgentCore Platform v1.0"""

# RET-C2-283 - DomainWorkflowGraph (inner BaseGraph)
#
# This is the INNER graph for the Cat 2 two-layer nested architecture.
# It encapsulates the full food-label allergen-compliance advisory workflow:
#
#   START -> input_validate -> retrieve -> rerank_filter
#         -> generate_answer -> output_format -> END
#
# Called by AllergenScreeningGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   - Inherits BaseGraph (fully custom topology - no forced backbone)
#   - Implements all 7 BaseGraph ABC methods
#   - register_nodes() does NOT call super() (abstract in BaseGraph)
#   - register_nodes() instantiates every domain node with NO ctor args
#   - Does NOT register initialize / finalize (outer backbone concerns)
#   - get_output() designed together with AllergenScreeningGraphNode.merge_output()

from typing import Any, Dict

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.graph.context_bridge import get_label_request
from src.schemas.state import State, to_json


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for RET-C2-283.

    Inherits BaseGraph directly for a fully custom node topology.
    Called by AllergenScreeningGraphNode.get_subgraph() in graph.py, which
    passes the declared runtime config (`_parent_config()`) into the ctor.

    Pipeline (linear):
        START
          -> input_validate  (InputValidateNode)  - parse + normalise the label text
          -> retrieve        (RetrieveNode)       - match seeded allergen provisions
          -> rerank_filter   (RerankFilterNode)   - classify statuses + abstain gate
          -> generate_answer (GenerateAnswerNode) - cited, tier-tagged advisory body
          -> output_format   (OutputFormatNode)   - final format (no disclaimer - see graph.py)
          -> END

    All nodes are FunctionNode subclasses returning partial-dict state updates.
    initialize / finalize are outer backbone concerns - not registered here.
    """

    # -- Identity --------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "ret_c2_283_allergen_compliance_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict subclass shared across inner and outer graph."""
        return State

    # -- Config validation -----------------------------------------------------

    def _validate_config(self) -> None:
        """Validate inner graph config before compilation.

        The forwarded `retrieval` block (score_threshold / top_k / kb_path /
        abstain_min_chars) is read per-call by the domain nodes with safe
        defaults, so absence is non-fatal. Validation is permissive here
        rather than raising ConfigError.
        """
        pass

    # -- Config + caller-data forwarding into state (config/bridge -> inner nodes)

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the declared `retrieval` block and the bridged label request.

        AllergenScreeningGraphNode._parent_config() forwards the declared
        `retrieval` + `llm` keys under config["configurable"]; this hook makes
        the `retrieval` block reachable by the domain nodes at runtime as the
        JSON-string state field `retrieval_config` (structured State fields
        travel as JSON strings for checkpoint safety). Every inner node reads
        a passed retrieval_config first and falls back to its own module
        defaults - there is no `config` parameter on execute() (config
        reaches nodes via ctor or State only).

        The screened caller record (input_context["label_request"], screened
        by the outer PreProcessNode) rides the ContextVar bridge because the
        framework's GraphNode does not forward structured caller data into
        subgraph.invoke() - see src/graph/context_bridge.py.
        """
        retrieval = (self.config or {}).get("configurable", {}).get("retrieval") or {}
        return {
            "retrieval_config": to_json(retrieval),
            "screening_request": get_label_request(),
        }

    # -- Node registration -----------------------------------------------------

    def register_nodes(self) -> None:
        """Register all 5 domain nodes.

        No super() call - BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize; those are outer backbone
        concerns handled by AgentBaseGraph in graph.py.

        Every node is instantiated with NO constructor arguments; config
        flows in via State seeding only (_extra_initial_state() above), per
        the execute(self, state) -> dict contract.
        """
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # -- Edge wiring -----------------------------------------------------------

    def add_edges(self) -> None:
        """Wire the linear allergen-screening domain topology.

        Each step passes its partial-dict output into the shared State.
        For this template the topology is intentionally linear - no
        conditional branching between domain nodes. route() is implemented
        as required by the ABC but add_conditional_edges() is not used.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # -- Routing ---------------------------------------------------------------

    def route(self, state: AgentState) -> str:
        """Conditional routing - required by BaseGraph ABC.

        For this linear topology add_conditional_edges() is not used, so this
        method is never called at runtime. It is implemented to satisfy the ABC
        contract. Returns END on error so an unexpected call does not re-enter a
        processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "output_format"

    # -- Output shape ----------------------------------------------------------

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        This dict is received by AllergenScreeningGraphNode.merge_output() in
        graph.py as the `sub_result` argument. Both methods are designed
        together to guarantee field-name consistency:

            Inner get_output()  emits: "formatted_answer", "citations",
                                       "screening_status", "overall_abstain",
                                       "status", ...
            Outer merge_output() reads: sub_result.get("formatted_answer"),
                                        sub_result.get("citations"),
                                        sub_result.get("screening_status"),
                                        sub_result.get("overall_abstain"),
                                        sub_result.get("status")

        Additional fields (intake_notes, error_log, trace_id,
        correlation_id, node_history) are surfaced for observability and so
        an inner validation rejection can name the offending field to the
        outer graph (AllergenScreeningGraphNode.on_subgraph_error surfaces
        only clean single-line messages).
        """
        return {
            # the reason must leave the subgraph or the outer graph cannot report it
            "error_code": state.get("error_code"),
            "formatted_answer": state.get("formatted_answer"),
            "citations": state.get("citations"),
            "screening_status": state.get("screening_status"),
            "overall_abstain": state.get("overall_abstain"),
            "status": state.get("status"),
            "intake_notes": state.get("intake_notes"),
            "error_log": state.get("error_log", []),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
