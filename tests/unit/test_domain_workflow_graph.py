# RET-C2-283 — Unit Tests: DomainWorkflowGraph (inner BaseGraph)
#
# Inner-graph composition + a full inner invoke() over the seeded KB. The
# inner graph runs the 5 domain nodes (all ANONYMOUS) — the outer
# boundary is the AgentBaseGraph backbone's concern and is covered in
# test_graph_composition.py / the PoB suite.
#
# The allergen term used to drive a real KB match is read PROGRAMMATICALLY
# from config/kb/allergen_kb.json (see test_retrieve_node.py's rationale) —
# zero hand-typed Japanese in this file.
#
# Mirrors docs/03_test_spec.md S3.1 (INT-01..INT-06).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import json
import pathlib

from langgraph.graph import END

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_status import AgentStatus

from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import AllergenScreeningGraphNode
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, from_json

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_KB = json.loads((_REPO_ROOT / "config" / "kb" / "allergen_kb.json").read_text(encoding="utf-8"))
_WHEAT = next(e for e in _KB if e["id"] == "kb-mand-003")
_WHEAT_PAYLOAD = f"ingredient declaration includes {_WHEAT['explicit_terms'][0]} as the base"


class TestInnerGraphConstruction:
    def test_int_01_inherits_base_graph(self):
        assert issubclass(DomainWorkflowGraph, BaseGraph)

    def test_int_01_registers_the_five_domain_nodes(self):
        inner = DomainWorkflowGraph()
        inner.register_nodes()
        assert set(inner._nodes.keys()) == {
            "input_validate",
            "retrieve",
            "rerank_filter",
            "generate_answer",
            "output_format",
        }
        assert isinstance(inner._nodes["input_validate"], InputValidateNode)
        assert isinstance(inner._nodes["retrieve"], RetrieveNode)
        assert isinstance(inner._nodes["rerank_filter"], RerankFilterNode)
        assert isinstance(inner._nodes["generate_answer"], GenerateAnswerNode)
        assert isinstance(inner._nodes["output_format"], OutputFormatNode)

    def test_inner_graph_name_and_schema(self):
        inner = DomainWorkflowGraph()
        assert inner.name == "ret_c2_283_allergen_compliance_workflow"
        assert inner.state_schema is State

    def test_initialize_finalize_are_not_registered(self):
        # Outer backbone concerns must not leak into the inner topology.
        inner = DomainWorkflowGraph()
        inner.register_nodes()
        assert "initialize" not in inner._nodes
        assert "finalize" not in inner._nodes


class TestConfigForwarding:
    def test_int_02_extra_initial_state_republishes_retrieval_block(self):
        inner = DomainWorkflowGraph(config={"configurable": {"retrieval": {"top_k": 2}}})
        extra = inner._extra_initial_state()
        assert set(extra.keys()) == {"retrieval_config", "screening_request"}
        assert isinstance(extra["retrieval_config"], str)  # JSON string (checkpoint safety)
        assert from_json(extra["retrieval_config"]) == {"top_k": 2}

    def test_extra_initial_state_with_no_config_is_empty_block(self):
        assert from_json(DomainWorkflowGraph()._extra_initial_state()["retrieval_config"]) == {}

    def test_extra_initial_state_seeds_the_bridged_label_request(self):
        # The screened caller record crosses the outer-to-inner boundary via
        # the ContextVar bridge — the framework's GraphNode does not forward
        # structured caller data into subgraph.invoke().
        from src.graph.context_bridge import set_label_request

        try:
            set_label_request('{"label_text": "wheat"}')
            extra = DomainWorkflowGraph()._extra_initial_state()
            assert extra["screening_request"] == '{"label_text": "wheat"}'
        finally:
            set_label_request(None)

    def test_extra_initial_state_with_no_bridged_request_is_none(self):
        from src.graph.context_bridge import set_label_request

        set_label_request(None)
        assert DomainWorkflowGraph()._extra_initial_state()["screening_request"] is None


class TestOutputShape:
    def test_int_03_get_output_shapes_the_merge_contract(self):
        inner = DomainWorkflowGraph()
        out = inner.get_output(
            {
                "formatted_answer": "ANSWER",
                "citations": "[]",
                "screening_status": "[]",
                "overall_abstain": False,
                "status": AgentStatus.SUCCESS.value,
                "node_history": ["InputValidateNode"],
            }
        )
        assert out["formatted_answer"] == "ANSWER"
        assert out["citations"] == "[]"
        assert out["screening_status"] == "[]"
        assert out["overall_abstain"] is False
        assert out["status"] == AgentStatus.SUCCESS.value
        assert out["node_history"] == ["InputValidateNode"]

    def test_route_returns_end_on_error(self):
        inner = DomainWorkflowGraph()
        assert inner.route({"status": AgentStatus.ERROR.value}) == END
        assert inner.route({"status": AgentStatus.SUCCESS.value}) == "output_format"


class TestInnerEndToEnd:
    def _invoke(self, payload: str) -> dict:
        # Same construction path the outer GraphNode uses: manifest-derived
        # config via _parent_config(); domain nodes take NO ctor args.
        inner = DomainWorkflowGraph(config=AllergenScreeningGraphNode()._parent_config())
        return inner.invoke(payload, session_id="inner-e2e")

    def test_int_04_full_inner_run_produces_the_formatted_answer(self):
        result = self._invoke(_WHEAT_PAYLOAD)
        assert result["status"] == AgentStatus.SUCCESS.value
        answer = result["formatted_answer"]
        assert answer.startswith("# Food Label Allergen Compliance — Advisory Screening")
        assert "[1]" in answer
        citations = from_json(result["citations"])
        assert citations and citations[0]["id"] == "kb-mand-003"

    def test_int_04_inner_node_history_is_the_linear_topology(self):
        history = self._invoke(_WHEAT_PAYLOAD)["node_history"]
        assert history == [
            "InputValidateNode",
            "RetrieveNode",
            "RerankFilterNode",
            "GenerateAnswerNode",
            "OutputFormatNode",
        ]

    def test_int_05_inner_output_never_carries_the_outer_advisory_stamp(self):
        """The inner graph's own OutputFormatNode does not stamp — that is
        exclusively the outer PostProcessNode's job (Safety Boundary #3)."""
        answer = self._invoke(_WHEAT_PAYLOAD)["formatted_answer"]
        assert "advisory only" not in answer

    def test_int_06_all_nine_mandatory_allergens_get_a_status_even_unmatched(self):
        """SAFETY: every mandatory allergen always gets a status — the
        caller can never see a partial mandatory-tier list."""
        result = self._invoke(_WHEAT_PAYLOAD)
        statuses = from_json(result["screening_status"])
        mandatory_ids = {s["id"] for s in statuses if s["tier"] == "mandatory"}
        all_mandatory_ids = {e["id"] for e in _KB if e["tier"] == "mandatory"}
        assert mandatory_ids == all_mandatory_ids

    def test_short_input_abstains_end_to_end(self):
        result = self._invoke("hi")
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["overall_abstain"] is True
        assert from_json(result["citations"]) == []
        assert from_json(result["screening_status"]) == []
