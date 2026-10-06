# RET-C2-283 — Unit Tests: nested Cat-2 graph composition (outer + end-to-end)
#
# Drives the REAL outer agent (FoodLabelAllergenComplianceAgent / Graph)
# end-to-end via AgentBaseGraph.invoke(). The e2e context is
# InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL) — the
# manifest's declared caller level; for_internal() is NEVER used (it would
# over-privilege the run and hide trust-gate regressions).
#
# The e2e payload is read PROGRAMMATICALLY from deploy/invoke_payload.json
# (the same payload the invoke-order boundary test proves) — zero hand-typed
# Japanese in this file; see test_retrieve_node.py's rationale.
#
# Mirrors docs/03_test_spec.md S3.2 (INT-07..INT-18).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import json
import pathlib

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import (
    _ADVISORY_NOTICE,
    AllergenScreeningGraphNode,
    FoodLabelAllergenComplianceAgent,
    Graph,
)
from src.nodes.post_process_node import PostProcessNode, _ADVISORY_STAMP
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json, to_json

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEPLOY_PAYLOAD = json.loads((_REPO_ROOT / "deploy" / "invoke_payload.json").read_text(encoding="utf-8"))
_VALID_PAYLOAD = _DEPLOY_PAYLOAD["input"]


def _run(user_input: str = _VALID_PAYLOAD, trust: TrustLevel = TrustLevel.VERIFIED_EXTERNAL) -> dict:
    ctx = InvocationContext(caller_trust_level=trust, caller_id="unit-suite")
    return Graph().invoke(user_input, ctx=ctx)


class TestOuterGraphConstruction:
    def test_int_07_inherits_agent_base_graph_directly(self):
        assert issubclass(FoodLabelAllergenComplianceAgent, AgentBaseGraph)

    def test_int_07_graph_alias(self):
        assert Graph is FoodLabelAllergenComplianceAgent

    def test_state_schema_is_state(self):
        assert FoodLabelAllergenComplianceAgent().state_schema is State

    def test_int_08_compile_fills_all_backbone_slots(self):
        agent = FoodLabelAllergenComplianceAgent()
        agent.compile()
        for slot in ("initialize", "pre_process", "main", "post_process", "finalize"):
            assert agent._nodes.get(slot) is not None, f"backbone slot not filled: {slot}"
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], AllergenScreeningGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_add_edges_is_not_overridden(self):
        # Backbone wiring belongs to the framework — the template must not
        # redefine it.
        assert "add_edges" not in FoodLabelAllergenComplianceAgent.__dict__


class TestMainSlotGraphNode:
    def test_int_09_get_subgraph_returns_the_inner_graph(self):
        subgraph = AllergenScreeningGraphNode().get_subgraph()
        assert isinstance(subgraph, DomainWorkflowGraph)
        assert subgraph.config["configurable"]["retrieval"], "inner config must carry the retrieval block"

    def test_int_10_extract_input_prefers_validated_input(self):
        node = AllergenScreeningGraphNode()
        assert node.extract_input({"validated_input": "VI", "user_input": "UI"}) == "VI"
        assert node.extract_input({"user_input": "UI"}) == "UI"

    def test_int_11_merge_output_maps_the_inner_contract(self):
        node = AllergenScreeningGraphNode()
        citations = to_json(
            [
                {
                    "ref": 1,
                    "id": "kb-mand-001",
                    "allergen_en": "Egg",
                    "allergen_ja": "J",
                    "tier": "mandatory",
                    "source": "s",
                }
            ]
        )
        delta = node.merge_output(
            {},
            {
                "formatted_answer": "ANSWER",
                "citations": citations,
                "screening_status": "[]",
                "overall_abstain": False,
                "status": AgentStatus.SUCCESS.value,
            },
        )
        # The inner formatted_answer surfaces as BOTH allergen_screening_result
        # and result (PostProcessNode's output gate reads state["result"]).
        assert delta == {
            "allergen_screening_result": "ANSWER",
            "result": "ANSWER",
            "citations": citations,
            "screening_status": "[]",
            "overall_abstain": False,
            "status": AgentStatus.SUCCESS.value,
            # The boundary now also carries the degraded-completion marker; on an
            # answered run neither side set one, so it crosses empty.
            "error_code": "",
        }

    def test_error_strategy_handles_inner_failures_and_hitl_is_contained(self):
        # "handle": an inner failure fails CLOSED via on_subgraph_error with
        # clean messages — a raw traceback never crosses the graph boundary.
        assert AllergenScreeningGraphNode.error_strategy == "handle"
        assert AllergenScreeningGraphNode.propagate_hitl is False

    def test_on_subgraph_error_surfaces_only_single_line_messages(self):
        from framework.errors import SubgraphError

        error = SubgraphError(
            agent_name="inner",
            error_log=[
                "InputValidateNode: top_k must be an integer in [1, 20]",
                "InputValidateNode: top_k must be an integer in [1, 20]",  # dupe
                '[RetrieveNode] boom\nTraceback (most recent call last):\n  File "/x.py"',
            ],
            trace_id="t",
        )
        delta = AllergenScreeningGraphNode().on_subgraph_error({}, error)
        assert delta["status"] == AgentStatus.ERROR.value
        assert delta["error_log"] == ["InputValidateNode: top_k must be an integer in [1, 20]"]

    def test_int_12_parent_config_never_empty_without_config(self):
        # Even with no runtime config the forwarded config carries the
        # fallback retrieval/llm blocks — never {}.
        cfg = AllergenScreeningGraphNode(parent_config={})._parent_config()
        assert cfg["configurable"]["retrieval"]["kb_path"] == "config/kb/allergen_kb.json"
        assert cfg["configurable"]["llm"]

    def test_parent_config_forwards_a_declared_override(self):
        cfg = AllergenScreeningGraphNode(
            parent_config={
                "retrieval": {
                    "score_threshold": 0.5,
                    "top_k": 3,
                    "kb_path": "config/kb/allergen_kb.json",
                    "abstain_min_chars": 4,
                }
            }
        )._parent_config()
        assert cfg["configurable"]["retrieval"]["score_threshold"] == 0.5


class TestEndToEndInvoke:
    """Full agent run: outer backbone + inner domain workflow, no LLM."""

    def test_int_13_invoke_returns_success(self):
        result = _run()
        assert (
            result.get("status") == AgentStatus.SUCCESS.value
        ), f"Expected success, got {result.get('status')}. result={result!r}"

    def test_int_13_output_is_the_gated_stamped_answer(self):
        output = _run().get("output")
        assert isinstance(output, str) and output.strip()
        assert output.startswith("# Food Label Allergen Compliance — Advisory Screening")
        assert _ADVISORY_STAMP in output

    def test_int_13_e2e_traverses_the_post_process_gate(self):
        history = _run().get("node_history", [])
        for cls_name in ("PreProcessNode", "AllergenScreeningGraphNode", "PostProcessNode"):
            assert cls_name in history, f"node_history missing {cls_name}: {history}"

    def test_int_14_anonymous_caller_is_denied_at_the_outer_boundary(self):
        """Trust gate at graph level: an ANONYMOUS invoke is refused by the
        VERIFIED_EXTERNAL pre_process slot. The error state short-circuits the
        main slot and routes past post_process to finalize — no domain answer
        is ever produced."""
        result = _run(trust=TrustLevel.ANONYMOUS)
        assert result.get("status") == AgentStatus.ERROR.value
        assert result["output"] is None
        # Closed-set error only: the framework's denial line stays internal.
        assert result["error"] == {"reason": "workflow_failed"}
        assert "error_log" not in result
        assert "trust gate denied" not in json.dumps(result, ensure_ascii=False, default=str)
        history = result.get("node_history", [])
        assert "PostProcessNode" not in history
        assert history[:2] == ["InitializeNode", "PreProcessNode"]


class TestInnerFailureContainment:
    """An unexpected inner-node crash fails CLOSED with no runtime detail."""

    def test_inner_exception_never_leaks_a_traceback_to_the_caller(self, monkeypatch):
        from src.nodes import retrieve_node as retrieve_module

        def _boom(self, state):
            raise RuntimeError("synthetic inner failure")

        monkeypatch.setattr(retrieve_module.RetrieveNode, "execute", _boom)
        result = _run()
        assert result.get("status") == AgentStatus.ERROR.value
        assert result["output"] is None
        assert result["error"] == {"reason": "workflow_failed"}
        assert "error_log" not in result
        assert "screening_status" not in result
        rendered = json.dumps(result, ensure_ascii=False, default=str)
        assert "Traceback" not in rendered
        assert 'File "' not in rendered
        assert "/src/" not in rendered
        assert "synthetic inner failure" not in rendered


class TestStructuredProductOutput:
    """Structured product contract: get_output() adds the tier-tagged
    structured payload on SUCCESS only, fail-closed on a gate violation."""

    def test_int_15_success_state_carries_the_structured_fields(self):
        agent = FoodLabelAllergenComplianceAgent()
        state = {
            "status": AgentStatus.SUCCESS.value,
            "screening_status": to_json([{"id": "kb-mand-001", "status": "disclosed"}]),
            "citations": to_json([{"ref": 1, "id": "kb-mand-001"}]),
            "overall_abstain": False,
            "formatted_output": "clean answer",
            "node_history": [],
        }
        out = agent.get_output(state)
        assert out["screening_status"] == [{"id": "kb-mand-001", "status": "disclosed"}]
        assert out["citations"] == [{"ref": 1, "id": "kb-mand-001"}]
        assert out["overall_abstain"] is False
        assert out["advisory_notice"] == _ADVISORY_NOTICE
        assert "error" not in out
        assert "error_log" not in out

    def test_int_16_non_success_state_withholds_the_structured_fields(self):
        agent = FoodLabelAllergenComplianceAgent()
        state = {
            "status": AgentStatus.ERROR.value,
            "screening_status": to_json([{"id": "kb-mand-001", "status": "disclosed"}]),
            "citations": "[]",
            "error_log": ["InputValidateNode: top_k must be an integer in [1, 20]"],
            "node_history": [],
        }
        out = agent.get_output(state)
        assert "screening_status" not in out
        assert "citations" not in out
        assert "advisory_notice" not in out
        # Closed-set error only; the internal channel is not projected.
        assert out["output"] is None
        assert out["error"] == {"reason": "workflow_failed"}
        assert "error_log" not in out
        assert "top_k" not in json.dumps(out)

    def test_int_17_fail_closed_withholds_structured_fields_on_gate_violation(self):
        """A credential-shaped string riding the structured citations payload
        must NOT bypass the output gate by avoiding the text path."""
        agent = FoodLabelAllergenComplianceAgent()
        state = {
            "status": AgentStatus.SUCCESS.value,
            "screening_status": "[]",
            "citations": to_json([{"ref": 1, "source": "sk-ABCDEF0123456789abcdef"}]),
            "overall_abstain": False,
            "formatted_output": "clean answer",
            "node_history": [],
        }
        out = agent.get_output(state)
        assert "citations" not in out
        assert "screening_status" not in out
        assert "advisory_notice" not in out
        # The base envelope (already gated by PostProcessNode) is untouched.
        assert out["output"] == "clean answer"


class TestStateRoundTrip:
    """Checkpoint-safety helpers: producers to_json() on write, consumers
    from_json()."""

    def test_to_from_json_list_round_trip(self):
        original = [{"id": "kb-mand-001", "score": 0.75, "allergen_en": "Egg"}]
        assert from_json(to_json(original)) == original

    def test_to_from_json_dict_round_trip(self):
        original = {"include_recommended": True, "top_k": 3}
        assert from_json(to_json(original)) == original

    def test_to_json_none_passes_through(self):
        assert to_json(None) is None

    def test_from_json_malformed_returns_default(self):
        assert from_json("{not valid json", default=[]) == []
        assert from_json(None, default={}) == {}
        assert from_json("", default=[]) == []
