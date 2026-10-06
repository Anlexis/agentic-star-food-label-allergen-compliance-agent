# RET-C2-283 — Unit Tests: manifest / config consistency
#
# config/agent.yaml (the flat registry manifest) and config/config.yaml (the
# runtime parameters the registry passes as Graph(config=...)) are live
# configuration, not documentation: AllergenScreeningGraphNode._parent_config()
# forwards the declared retrieval/llm blocks into the inner graph, and the
# class-name contract requires the declared entry point to BE the
# src/graph/graph.py agent class. These tests pin config <-> code consistency
# so a config drift fails fast in CI.
#
# Mirrors docs/03_test_spec.md S2.8 (CFG-01..CFG-08).
# Deterministic — no LLM, no network.

import json
import pathlib

import yaml

from framework.schemas.trust_level import TrustLevel

from src.graph.graph import (
    AllergenScreeningGraphNode,
    FoodLabelAllergenComplianceAgent,
    _runtime_config,
)
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = yaml.safe_load((_ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))
_RUNTIME = yaml.safe_load((_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))


class TestManifestIdentity:
    def test_cfg_01_agent_id_is_the_template_id(self):
        assert _MANIFEST["id"] == "RET-C2-283"
        assert _MANIFEST["enabled"] is True

    def test_cfg_02_declared_class_is_the_graph_class(self):
        # Class-name contract: manifest entry point == graph.py class == server import.
        assert _MANIFEST["class"] == ("src.graph.graph." + FoodLabelAllergenComplianceAgent.__name__)
        assert _MANIFEST["name"] == FoodLabelAllergenComplianceAgent().name

    def test_cfg_03_category_and_industry(self):
        assert _MANIFEST["category"] == "Cat 2"
        assert _MANIFEST["industry"] == "RET"
        assert _MANIFEST["base_type"] == "RAGAgent"
        # Rule-assembled screening — no model invocation in the pipeline.
        assert _MANIFEST["generation_mode"] == "deterministic"

    def test_cfg_03b_requires_blocks_match_the_code(self):
        # No ctx.secrets.require() call and no client construction in src/ —
        # declaring an unprovisioned secret/extra would fail the deployment
        # at compile time.
        assert _MANIFEST["requires"]["secrets"] == []
        assert _MANIFEST["requires"]["extras"] == []


class TestManifestSecurity:
    def test_cfg_04_required_trust_level_matches_outer_gate_nodes(self):
        declared = TrustLevel(_MANIFEST["required_trust_level"])
        assert declared is TrustLevel.VERIFIED_EXTERNAL
        assert PreProcessNode.required_trust_level is declared
        assert PostProcessNode.required_trust_level is declared

    def test_cfg_05_max_retry_within_framework_ceiling(self):
        max_retry = _RUNTIME["max_retry"]
        assert isinstance(max_retry, int)
        assert 0 <= max_retry < 10  # AgentBaseGraph MAX_RETRY_CEILING

    def test_cfg_05b_timeout_is_declared_in_seconds(self):
        assert isinstance(_RUNTIME["timeout_s"], int)
        assert _RUNTIME["timeout_s"] > 0

    def test_hitl_is_not_enabled(self):
        # HITL auto-waiver contract: this template declares no HITL.
        assert (_RUNTIME.get("hitl") or {}).get("enabled", False) is False
        assert AllergenScreeningGraphNode.propagate_hitl is False


class TestRuntimeConfigPlumbing:
    def test_cfg_06_retrieval_block_matches_node_defaults(self):
        # Node module defaults mirror the declared block — a drift silently
        # changes tuning.
        retrieval = _RUNTIME["retrieval"]
        from src.nodes.rerank_filter_node import _DEFAULT_RETRIEVAL as rerank_defaults
        from src.nodes.retrieve_node import _DEFAULT_RETRIEVAL as retrieve_defaults

        assert retrieval["top_k"] == retrieve_defaults["top_k"] == rerank_defaults["top_k"]
        assert (
            retrieval["score_threshold"] == retrieve_defaults["score_threshold"] == rerank_defaults["score_threshold"]
        )
        assert retrieval["abstain_min_chars"] == rerank_defaults["abstain_min_chars"]
        assert retrieval["kb_path"] == retrieve_defaults["kb_path"]
        assert (_ROOT / retrieval["kb_path"]).is_file()

    def test_cfg_07_runtime_config_loads_the_declared_file(self):
        cfg = _runtime_config()
        assert cfg["max_retry"] == _RUNTIME["max_retry"]
        assert cfg["timeout_s"] == _RUNTIME["timeout_s"]
        assert cfg["retrieval"] == _RUNTIME["retrieval"]
        assert cfg["llm"] == _RUNTIME["llm"]

    def test_cfg_07b_parent_config_forwards_declared_blocks(self):
        node = AllergenScreeningGraphNode(parent_config=_runtime_config())
        cfg = node._parent_config()
        assert cfg["configurable"]["retrieval"] == _RUNTIME["retrieval"]
        assert cfg["configurable"]["llm"] == _RUNTIME["llm"]
        assert cfg["configurable"]["retrieval"], "_parent_config() must never forward an empty retrieval block"

    def test_cfg_07c_declared_retrieval_values_reach_the_inner_state(self):
        """A declared value is LIVE end-to-end: a non-default score_threshold
        declared in the runtime config must surface in the inner graph's
        seeded state, not silently degrade to a module default."""
        from src.graph.domain_workflow_graph import DomainWorkflowGraph
        from src.schemas.state import from_json

        custom = {
            "retrieval": {
                "score_threshold": 0.33,
                "top_k": 2,
                "kb_path": "config/kb/allergen_kb.json",
                "abstain_min_chars": 5,
            }
        }
        node = AllergenScreeningGraphNode(parent_config=custom)
        inner = DomainWorkflowGraph(config=node._parent_config())
        seeded = from_json(inner._extra_initial_state()["retrieval_config"])
        assert seeded["score_threshold"] == 0.33
        assert seeded["top_k"] == 2
        assert seeded["abstain_min_chars"] == 5


class TestSeededKnowledgeBase:
    def test_cfg_08_kb_is_a_well_formed_entry_list(self):
        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        assert isinstance(entries, list)
        assert len(entries) >= 5, "seeded KB must carry a usable corpus"
        for entry in entries:
            assert set(entry.keys()) == {
                "id",
                "allergen_en",
                "allergen_ja",
                "tier",
                "source",
                "explicit_terms",
                "implicit_hint_terms",
                "content",
            }
            assert entry["id"] and entry["allergen_en"] and entry["allergen_ja"]
            assert entry["tier"] in ("mandatory", "recommended")

    def test_cfg_08_kb_ids_are_unique(self):
        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        ids = [e["id"] for e in entries]
        assert len(ids) == len(set(ids))

    def test_cfg_08_nine_mandatory_specified_raw_materials(self):
        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        mandatory = [e for e in entries if e["tier"] == "mandatory"]
        assert len(mandatory) == 9
