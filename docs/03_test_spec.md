# Test Specification — RET-C2-283

**Template ID:** RET-C2-283
**Template Name:** FoodLabelAllergenComplianceAgent
**Category:** Cat 2 (nested RAG) — Food Label Allergen Compliance Advisory Screening

This document defines the test cases for the implementation (state, nodes,
inner/outer graphs, manifest, server). The test code lives in `tests/unit/` +
`tests/proof_of_boundary/`; this spec is the contract those tests implement.

> ⚠️ **This is a food-allergen safety template — the tests ARE the safety case.** A
> false "disclosed" could seriously harm someone. Section 3.3 below is dedicated to
> proving the hard safety invariants end-to-end, not just per-node logic.

## 1. Scope & Invocation Conventions

- Per-node unit tests for the 5 inner domain nodes + the 2 outer gate nodes.
- Manifest/config consistency (`config/agent.yaml` + `config/config.yaml` ↔ code)
  and seeded-KB integrity.
- Inner-graph (`DomainWorkflowGraph`) and outer-graph
  (`FoodLabelAllergenComplianceAgent`) composition / integration.
- A dedicated Safety-Boundary suite proving the hard invariants named in
  `docs/02_design.md` "Safety Boundary" end-to-end.
- Proof-of-Boundary (PoB): import isolation, State msgpack safety, invoke order
  (PB-6), HITL propagation (PB-7, auto-waived — non-HITL), server boot, and
  end-to-end business behaviour through the real ASGI `POST /invoke`.

**Trust-gate invocation canon.** Behavioural per-node tests invoke the node via
`node(state)` — through `BaseNode.__call__`, which runs the trust gate → input
gate → `execute()` → output gate. The state builder sets `caller_trust_level` to
`TrustLevel.VERIFIED_EXTERNAL.value` for the two outer gate slots
(PreProcessNode / PostProcessNode — the manifest's declared caller level) and
`TrustLevel.ANONYMOUS.value` for the five inner domain nodes. The
template-owned screens (control tokens, label_request contract, fail-closed
numerics) are ADDITIONALLY proven by calling `execute()` directly — the refusal
must hold even where no framework wrapper runs.

**No `execute(state, config=…)` signature in this template.**
`execute(self, state) -> dict` is the only signature here — every node reads
config exclusively from State (`retrieval_config`, `query_filters`), seeded by
`DomainWorkflowGraph._extra_initial_state()` / `InputValidateNode`. A test that
passes a 2nd `execute()` argument would `TypeError`.

**Input-gate masking expectations.** The framework input gate masks
`user_input`/`validated_input`/`llm_response` (e-mail, phone/SSN/CC digit groups,
Title-Case name bigrams) to `[MASKED]` before `execute()` runs. Positive-path
payloads for generic (non-KB-matching) tests are lowercase ASCII, PII-free label
text; domain fields (`label_text`, `grounded_answer`, `formatted_answer`,
`matched_provisions`, `screening_status`, …) are not scan targets. English
Title-Case label text travels on the `input_context["label_request"]` channel,
which the framework does not rewrite — the survival of Title-Case ingredient
names through that channel is itself pinned by tests (PRE-TC, E2E-EN below).

**Audit muting.** `shared.*` is never sys.modules-stubbed (the framework imports
`shared.security` at load time). The domain audit emitter is muted via an autouse
fixture patching `src.nodes.<mod>.emit_trace_event`; audit assertion tests
re-patch the same attribute with a spy and assert on `call.args[1]` (the event
payload), never the whole-call repr.

**Japanese-text safety (safety-critical).** Every Japanese/CJK string used
to build a test payload or assertion is either (a) read PROGRAMMATICALLY from the
real `config/kb/allergen_kb.json` / `deploy/invoke_payload.json` at
test-collection time, or (b) imported directly from the owning module's own
constant (`_ABSTAIN_ANSWER`, `_ADVISORY_STAMP`, `_ADVISORY_NOTICE`,
`_NO_TEXT_TIER_HEADER`), or (c) confined to the E2E suite's fixed fixtures.
No unit-test file hand-types a CJK literal for the safety-critical allergen
vocabulary — this removes transcription risk end to end. See
`test_retrieve_node.py` / `test_safety_boundary.py` module docstrings.

## 2. Unit Test Cases

### 2.1 PreProcessNode (outer pre_process slot; caller gate) — `test_pre_process_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| PRE-01 | Valid label text | lowercase ASCII ingredient text | `status=SUCCESS`, `validated_input` set |
| PRE-02 | Input is stripped | leading/trailing whitespace | `validated_input` trimmed |
| — | `enriched_context` | `input_context.channel` | carried through (JSON string); defaults to `"unknown"` |
| PRE-03 | Empty input | `""` | declined — the run COMPLETES carrying `EMPTY_INPUT`, `error_log` non-empty, no `validated_input` |
| PRE-04 | Whitespace only | `"   \n\t "` | declined — run COMPLETES carrying the reason code |
| PRE-05 | Missing `user_input` | key absent | declined — run COMPLETES carrying the reason code |
| PRE-06 | Non-string input | dict payload | the call-boundary contract still returns a graceful declined result carrying a reason code — never a raw traceback to the caller |
| PRE-07 | Domain audit | valid input | `pre_process_complete` emitted; payload (`call.args[1]`) carries `input_chars` |
| PRE-TC | Title-Case survival | English label with multi-word ingredient names via `label_request` | names survive screening verbatim into `screening_request` |
| PRE-PII | Personal data masked, not refused | label text + contact line | SUCCESS; e-mail/phone spans masked; ingredient names intact |
| PRE-TOK | Control tokens refused | `<\|im_start\|>` / `[INST]` / `<<SYS>>` in values or KEYS | `status=ERROR` — terminal — via direct `execute()`; token never echoed |
| PRE-BEN | Benign wording not refused | ordinary label prose ("storage instructions", "system of quality control", 日本語) | SUCCESS |
| PRE-NUM | Non-finite matrix | `top_k` ∈ NaN/Infinity/-Infinity/bool/0/21/3.5/str/list/dict | fail CLOSED by declining: run COMPLETES carrying `INVALID_REQUEST`, naming `top_k`; value never echoed |
| PRE-FLG | Non-boolean flag | `include_recommended` non-bool | fail CLOSED by declining: run COMPLETES carrying `INVALID_REQUEST`, naming the field |
| PRE-UNK | Unknown fields | stray/hostile field name | declined — run COMPLETES carrying `INVALID_REQUEST`; the field name never echoed |
| PRE-REQ | Contract shape | missing `label_text` / non-object record / oversize text | declined — run COMPLETES carrying `INVALID_REQUEST`, naming the field |
| PRE-DEG | Degrade path | no `label_request` at all | SUCCESS via the plain-text path; no `screening_request` |

Two stop outcomes, and the rows say which one they assert. A value the caller can correct —
no label text, a field past its bound, a malformed option, an unsupported field — ends in a
**completed** run carrying a reason code, so the caller can fix the value and send the request
again on the same conversation; the reason code is asserted alongside the status, because a
successful status on its own would also hold for a request that was simply screened. A control
sequence **terminates**: it is not a value to correct, and asserting a completed run there would
state that rewording the request is a route past the refusal. Neither path publishes
`validated_input` or `screening_request`, so "fail closed" holds on both.

### 2.2 InputValidateNode (inner node 1; ANONYMOUS) — `test_input_validate_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| VAL-01 | Plain text | free-text label | whole string becomes `label_text`; filters `{include_recommended: None, top_k: None}` |
| VAL-02 | Whitespace | ragged spacing/newlines | collapsed to single spaces |
| VAL-03 | JSON envelope | `{"text"/"label_text"/"ingredients", "include_recommended", "top_k"}` | all three parsed; every text-key alias accepted |
| VAL-04 | Malformed JSON | `{`-prefixed non-JSON | treated as plain-text label + parse note |
| VAL-05 | Malformed `top_k` | NaN / Infinity / 99 / −5 / 0 / 3.5 / bool / str / list / dict | fail CLOSED by declining: run COMPLETES carrying `INVALID_REQUEST`, naming `top_k`; value never echoed; no `label_text` produced |
| VAL-06 | In-range `top_k` | 7 | accepted into `query_filters` |
| VAL-07 | `include_recommended` non-boolean | `"yes"` | fail CLOSED by declining: run COMPLETES carrying `INVALID_REQUEST`, naming the field |
| VAL-08 | Oversize text | > 4000 chars | truncated to 4000 + note |
| VAL-09 | Empty request | `""` | `label_text=""` + "empty request" note (non-fatal) |
| VAL-10 | Domain audit | text + `top_k` override | `input_validate_complete` emitted; `has_top_k_override=True` |
| VAL-BR | Structured channel | bridged `screening_request` | supplies `label_text` + filters; beats the text path; malformed bridge falls back to text |
| VAL-TOK | Control tokens | token in plain text or an envelope field NAME | terminal refusal (`status=ERROR`) via direct `execute()`, no `label_text`; benign prose unaffected |
| VAL-PII | Personal data | e-mail in envelope text | masked + note |
| — | Checkpoint safety | any | `query_filters` is a JSON string, never a bare dict |

### 2.3 RetrieveNode (inner node 2; ANONYMOUS) — `test_retrieve_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| RET-01 | Explicit term match | KB `explicit_terms[0]` substring | score `1.0`, `matched_term` = the term |
| RET-02 | Implicit hint match | KB `implicit_hint_terms[0]` only | score `0.5` |
| RET-03 | No-match text keeps every mandatory entry | unrelated ASCII text | all 9 mandatory ids present, score `0.0` |
| RET-04 | No-match text drops every recommended entry | unrelated ASCII text | 0 recommended ids present |
| RET-05 | Empty text behaves like no-match | `""` | 9 mandatory ids, 0 recommended |
| RET-06 | Entry shape + excerpt cap | any hit | keys `{id,allergen_en,allergen_ja,tier,source,score,matched_term,excerpt}`; excerpt ≤ 280 chars |
| RET-07 | Deterministic ordering | mixed tiers | mandatory entries all precede recommended; mandatory sorted by id asc |
| RET-08 | Config precedence via State (no `execute(config=)`) | bogus `retrieval_config.kb_path` | `[]` + "not readable" note |
| RET-09 | Absent `retrieval_config` | — | falls back to the real KB module default |
| RET-10 | Notes accumulation | prior `intake_notes` | appended, never clobbered |
| RET-11 | Domain audit | any | `retrieve_complete` emitted; `kb_entries`/`candidates`/`text_chars` |
| — | KB integrity | — | 16 entries, unique ids, 9 mandatory / 7 recommended, required keys |

### 2.4 RerankFilterNode (inner node 3; ANONYMOUS) — SAFETY-CRITICAL — `test_rerank_filter_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| RRF-01 | Abstain gate | text < `abstain_min_chars` (8) | `overall_abstain=True`, `screening_status=[]` — no partial verdict, even with high-confidence candidates present |
| RRF-02 | No-abstain boundary | text length exactly 8 / one under | `False` / `True` respectively |
| RRF-03 | Threshold classification | score ≥ / between 0–threshold / == 0 | `disclosed` / `ambiguous` / `missing` |
| RRF-04 | Custom threshold via State | `retrieval_config.score_threshold` override | reclassifies accordingly |
| RRF-05 | Mandatory tier never capped | 9 mandatory entries + `top_k=1` | all 9 survive |
| RRF-06 | Recommended tier capped | 5 candidates + `top_k=2` | exactly 2 survive (highest-first, as delivered) |
| RRF-07 | `include_recommended=False` | mixed tiers | recommended dropped, mandatory unaffected; default (absent) is `True` |
| RRF-08 | Caller `top_k` | narrower wins; wider does not widen | enforced both directions |
| RRF-09 | Garbage entries | non-dict / uncoercible score | skipped / coerced to `0.0` |
| RRF-10 | Domain audit | any | `rerank_filter_complete` emitted |

### 2.5 GenerateAnswerNode (inner node 4; ANONYMOUS) — SAFETY-CRITICAL — `test_generate_answer_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| GEN-01 | Abstain path | `overall_abstain=True` | ONLY the fixed abstention message, `citations=[]` — even if `screening_status` is populated |
| GEN-02 | Citation markers | disclosed/ambiguous entries | `[n]` markers, sequential |
| GEN-03 | `missing` gets no ref | missing entry | no `[n]` marker, absent from citations |
| GEN-04 | Tier headers | mandatory-only / recommended-only / both | correct header(s) present, absent otherwise; a recommended entry never appears under the mandatory header |
| GEN-05 | No entries | empty `screening_status`, non-abstain | explicit no-coverage message |
| GEN-06 | No-verdict rule | full run | zero banned verdict tokens (`pass`/`fail`/`compliant`/`non-compliant`/`unsafe`); status vocabulary limited to `DISCLOSED`/`AMBIGUOUS`/`MISSING` |
| GEN-07 | Domain audit | any | `generate_answer_complete` emitted |

### 2.6 OutputFormatNode (inner node 5, terminal; ANONYMOUS) — `test_output_format_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| FMT-01 | Full compose | body + citations | header + body + `## Sources` rows; `status=SUCCESS` |
| FMT-02 | Blank source | citation without source | no dangling suffix on the source line |
| FMT-03 | No disclaimer here | any | this node NEVER adds the advisory stamp (safety-boundary-by-construction — see Safety Boundary #3) |
| FMT-04 | No citations | empty list | explicit "- none (…)" sources line |
| FMT-05 | Missing/empty body | no `grounded_answer` | fallback text; `status=SUCCESS` |
| FMT-06 | Domain audit | any | `output_format_complete` emitted |

### 2.7 PostProcessNode (outer post_process slot; output gate) — SAFETY-CRITICAL — `test_post_process_node.py`

| ID | Case | Input (`result`) | Expected |
|----|------|------------------|----------|
| POST-01 | Clean output | normal screening result | `formatted_output` = result + advisory stamp, `status=SUCCESS` |
| POST-02 | Empty result | `""` | forwarded as-is, unstamped, `status=SUCCESS` (non-fatal) |
| POST-03..06 | Credential leak | `sk-` API key / `password=` assignment / JWT (built at runtime) / Bearer token | `status=ERROR`, `formatted_output` = the closed-set envelope `{"reason": "output_withheld"}`, `result` = None, raw secret absent from the whole delta; the violation type on `error_log` |
| POST-07 | PII leak | e-mail / long digit run | same blocked behaviour as POST-03..06 |
| POST-08 | Stamp non-suppressible | every non-empty clean body | the fixed `_ADVISORY_STAMP` is present verbatim, unconditionally |
| — | Blocked path never leaks | any violation | response is EXACTLY the closed-set envelope (never a partial/unstamped result, never a stub carrying text); the envelope is truthy and its reason is in `ERROR_REASONS`; `status=ERROR` + `output_withheld` together are an unambiguous "do not trust this output" signal — the literal stamp text does not ride this path |
| POST-10 | Domain audit | clean path | `post_process_complete` emitted; `output_chars`; on a block `post_process_output_blocked` carries `{reason, violation}` and never the value |

### 2.8 Manifest / config consistency + KB integrity — `test_config_manifest.py`

| ID | Case | Expected |
|----|------|----------|
| CFG-01 | Template id | manifest `id` = `RET-C2-283`, `enabled: true` |
| CFG-02 | Class-name contract | manifest `class` == `src.graph.graph.FoodLabelAllergenComplianceAgent`; `name` == the agent's `name` property |
| CFG-03 | Classification | Cat 2 / RET / RAGAgent / `generation_mode: deterministic`; `requires.secrets`/`requires.extras` both `[]` (no `ctx.secrets.require()` call, no client construction in `src/`) |
| CFG-04 | Trust level | manifest `VERIFIED_EXTERNAL` == PreProcessNode & PostProcessNode `required_trust_level` |
| CFG-05 | Runtime params | `config/config.yaml` `max_retry` int in `[0, 10)` (framework ceiling); `timeout_s` positive int; hitl not enabled, `propagate_hitl=False` |
| CFG-06 | Retrieval block | `top_k`/`score_threshold`/`abstain_min_chars` mirror node module defaults; `kb_path` exists |
| CFG-07 | Config plumbing | `_runtime_config()` loads the declared file; `_parent_config()` forwards retrieval + llm blocks, never `{}`; a declared non-default value reaches the inner seeded state |
| CFG-08 | KB integrity | 16 entries, unique ids, required keys, 9 mandatory (特定原材料) |

### 2.9 Closed-set ERROR envelope — `test_error_envelope_closed_set.py`

The caller-visible error is a closed set (docs/02 "Output boundary"). A
runtime-assembled sentinel (a name, an e-mail, a token-shaped fragment —
deliberately not credential-shaped, so a filter would pass it) is seeded into
`error_log` next to entries of the framework's own shapes (a trust-gate denial,
a wrapped exception with traceback) and into every pre-gate field that can
survive in state.

| ID | Case | Expected |
|----|------|----------|
| ENV-01 | `get_output()` on every terminated shape | an error status routed to finalize, a timeout / pending status, an output-gate refusal, a foreign / non-string / empty value in the reason slot, a surviving pre-gate `result` / `formatted_output`: `output: null`, `error: {"reason": …}` drawn from `ERROR_REASONS`, truthy, no `error_log` key, no structured field, and a fixed key set; the sentinel and the framework markers appear nowhere in the returned mapping, walking nested keys and values |
| ENV-02 | Success envelope | unchanged (`output`, the structured fields), no `error` key — and no `error_log` key even when the internal channel is non-empty |
| ENV-03 | `PostProcessNode` on every gate rule × both entry points | `formatted_output` = `{"reason": "output_withheld"}` (truthy, single key), `result` = None, exactly one `error_log` line naming the rule and never the value, the seeded internal entries not re-emitted; the block audit event carries `{reason, violation}` only |
| ENV-04 | Already-errored state | direct `execute()` returns `{"reason": "workflow_failed"}` with `result` cleared and re-emits nothing; through the framework pipeline `execute()` never runs on an errored incoming state |
| ENV-05 | Builder | the vocabulary is exactly `{workflow_failed, output_withheld}`; an unknown reason is refused without being echoed; the probe finds the sentinel where it lives (verify the verifier) |

The closed-set envelope covers runs that **terminate**. A run declined because the caller can
correct the value does not reach it: `get_output()` returns the base envelope as it stands, whose
body is the fixed reason sentence `PostProcessNode` placed in the caller-facing slot, with a
successful status and no `error` key. The closed-set discipline is unchanged by that — the
sentence is a module constant, no structured field is released, and nothing a node authored
reaches the caller. The end-to-end proof of that path is in PB-E2E.

## 3. Integration / Composition

### 3.1 Inner graph — `test_domain_workflow_graph.py`

| ID | Case | Expected |
|----|------|----------|
| INT-01 | Composition | inherits `BaseGraph`; registers exactly the 5 domain nodes; no initialize/finalize |
| INT-02 | Config forwarding | `_extra_initial_state()` republishes the retrieval block as the JSON-string `retrieval_config` and seeds the bridged `screening_request` |
| INT-03 | Output shape | `get_output()` emits `formatted_answer`/`citations`/`screening_status`/`overall_abstain`/`status`/`error_log`/…; `route()` → END on error |
| INT-04 | Inner e2e (wheat match) | full inner `invoke()` → SUCCESS; formatted answer cites `kb-mand-003`; inner `node_history` = the 5 domain nodes in linear order |
| INT-05 | Inner output unstamped | inner `formatted_answer` never carries the outer advisory stamp |
| INT-06 | All-mandatory coverage | every one of the 9 mandatory allergens gets a status, even when unmatched |
| — | Abstain end-to-end | short input → inner `overall_abstain=True`, empty `citations`/`screening_status` |

### 3.2 Outer graph + e2e — `test_graph_composition.py`

| ID | Case | Expected |
|----|------|----------|
| INT-07 | Outer composition | inherits `AgentBaseGraph` (direct framework inheritance); `Graph` alias; `add_edges()` NOT overridden |
| INT-08 | Backbone slots | `compile()` fills all 5; pre/main/post are PreProcessNode / AllergenScreeningGraphNode / PostProcessNode |
| INT-09 | `get_subgraph()` | returns `DomainWorkflowGraph` carrying the forwarded retrieval config |
| INT-10 | `extract_input()` | prefers `validated_input`, falls back to `user_input` |
| INT-11 | `merge_output()` | inner `formatted_answer` → outer `allergen_screening_result` AND `result`; `citations`/`screening_status`/`overall_abstain`/`status` mapped; changed keys only |
| INT-12 | Config fallback | `_parent_config()` never `{}` even with no runtime config; declared overrides forwarded |
| — | Error strategy | `error_strategy == "handle"`; `on_subgraph_error()` dedupes and surfaces only single-line messages; a forced inner exception produces an ERROR response with no traceback / runtime path anywhere in it |
| INT-13 | e2e happy path (deploy payload) | VERIFIED_EXTERNAL invoke → SUCCESS; `output` = gated, stamped answer; PostProcessNode traversed |
| INT-14 | e2e trust denial | ANONYMOUS invoke → ERROR; empty `output`; PostProcessNode NOT traversed |
| INT-15/16 | Structured `get_output()` | on SUCCESS: `screening_status`/`citations`/`overall_abstain`/`advisory_notice` present; on non-SUCCESS: withheld entirely |
| INT-17 | Structured fail-closed | a credential-shaped string riding `citations` → the whole structured payload is withheld (never bypasses the output gate via the structured path) |
| — | Checkpoint-safety helpers | `to_json`/`from_json` round-trip; None/malformed handling |

### 3.3 Safety-Boundary proof suite — SAFETY-CRITICAL — `test_safety_boundary.py`

End-to-end (full `Graph().invoke()`) proofs of the hard invariants named in
`docs/02_design.md` "Safety Boundary" — this is the safety case, not just
per-node coverage:

| ID | Invariant | Proof |
|----|-----------|-------|
| SAFE-01 | Abstention is real | short input → SUCCESS + `overall_abstain=True` + empty structured fields; output routes the caller to "a qualified human reviewer"; the advisory stamp still rides the abstain path |
| SAFE-02 | Never an authoritative verdict | a full mixed-match run's output carries zero banned verdict tokens; `screening_status` status vocabulary limited to disclosed/missing/ambiguous |
| SAFE-03 | Tiers never conflated | every returned entry's `tier` matches the KB ground truth for that id; all 9 mandatory ids always present; `include_recommended=False` never narrows the mandatory tier |
| SAFE-04 | Advisory stamp non-suppressible | text output carries `_ADVISORY_STAMP` verbatim and the structured payload carries the fixed `advisory_notice`, regardless of which allergens were found (incl. the abstain path) |
| SAFE-05 | Allergen names round-trip byte-identical | a KB entry's `allergen_ja` survives a full outer invoke byte-for-byte (string equality AND explicit UTF-8 byte comparison), in both `screening_status` and `citations`, with no cross-contamination between neighbouring ids |
| — | Defence in depth | a uniquely tagged raw input marker never appears in the final output — the answer is grounded ONLY in KB-sourced fields, never raw caller text |

## 4. Proof-of-Boundary

| ID | Case | Expected |
|----|------|----------|
| PB-IMPORT | `test_import_isolation.py` | no platform-internal SDK import anywhere under `src/` |
| PB-STATE | `test_state_safety.py` | `State` has no credential-named fields and no `BaseModel` / `InvocationContext` annotations |
| PB-6 | `test_pb_invoke_order.py` | full `Graph().invoke()` with `InvocationContext(caller_trust_level=VERIFIED_EXTERNAL)` (never `for_internal()`) over the payload read programmatically from `deploy/invoke_payload.json` → SUCCESS with outer `node_history` exactly `[InitializeNode, PreProcessNode, AllergenScreeningGraphNode, PostProcessNode, FinalizeNode]`; inner domain nodes never leak into the outer history |
| PB-7 | `test_pb7_hitl_interrupt_propagation.py` | **Auto-waived — non-HITL** (`AllergenScreeningGraphNode.propagate_hitl = False`, no `hitl.enabled` block); skip stub retained |
| PB-BOOT | `test_server_boot.py` | `import src.api.server` does not raise; module-level agent is `FoodLabelAllergenComplianceAgent`, compiled; fresh ctor→`compile()` fills the 5 backbone slots; `/health` reports the agent |
| PB-E2E | `test_invoke_e2e.py` | end-to-end business behaviour through the real ASGI `POST /invoke` (Bearer auth): Japanese plain-text screening; English Title-Case ingredient names surviving the structured channel to DISCLOSED statuses; ambiguous + abstain severity paths; served runtime config live; fail-closed non-finite `top_k` matrix incl. a raw `NaN` JSON literal on the wire — the run **completes**, the body is the fixed reason sentence, no structured field is released, no `error_log`, field name absent; over-long `label_text` likewise, with no traceback in the body; control-token refusal **terminates** without echo; personal-data masking; 401 auth boundary; 413 size cap; `/health`. **Closed-set envelope at the wire:** a node is made to author a runtime-assembled sentinel into `error_log` during a real `/invoke`, on five paths — a caller-data rejection at pre_process and an inner rejection carried out of the subgraph both **complete** with the reason sentence as the body; an inner node raising (framework-wrapped message + traceback), an output-gate refusal (`output_withheld`) and a trust denial all **terminate** with `error: {"reason": …}`, `output: null`. On every one of the five, no `error_log` key is projected and the sentinel, `Traceback`, `File "`, the exception class, "trust gate denied" and "output blocked" are absent from every nested value and from the raw JSON; a verify-the-verifier case shows the same sentinel does enter the node's own `error_log` |

> **Gate checklist:** PB-IMPORT, PB-STATE, PB-6, PB-BOOT and PB-E2E are mandatory.
> PB-7 applies only to HITL-enabled templates — this template is non-HITL, so PB-7
> is **Auto-waived — non-HITL** and its skip must not block the gate.

## 5. Test Execution Summary

- Execution date: 2026-09-15
- Runner: real SDK wheel (`agenticstar-agentcore==1.0.2`), `python -m pytest tests/`
- Total tests (full `tests/` sweep, includes the PoB suite): 339
- Pass: 338 / Fail: 0 / Skip: 1 (PB-7 conditional stub — auto-waived, non-HITL)
- Determinism: no LLM, no network, no live OCR — retrieval is deterministic
  substring matching over the seeded KB; answer assembly is rule-based
