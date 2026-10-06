# Template Design Specification — RET-C2-283

**Template ID:** RET-C2-283
**Template Name:** FoodLabelAllergenComplianceAgent
**Category:** Cat 2 (multi-step domain workflow — RAG pattern)
**Industry:** RET

## Position in the Framework Architecture

| Aspect | Value |
|--------|-------|
| Agent class | `FoodLabelAllergenComplianceAgent` (alias `Graph`) |
| L1 Base (framework base class) | AgentBaseGraph — direct framework inheritance |
| Inner graph base | `BaseGraph` — `DomainWorkflowGraph` |
| Pattern | Cat 2 two-layer nested architecture (outer fixed 5-node backbone + `GraphNode` in the `main` slot wrapping an inner `BaseGraph` domain workflow) |

**Three-layer separation**
- State: flat `TypedDict` composition (no Pydantic — msgpack incompatible);
  structured fields stored as JSON strings via `to_json()` / `from_json()`
- Node: framework inheritance via `FunctionNode` (override
  `execute(self, state) -> dict` only — no `config` parameter)
- Graph: composition (`register_nodes()` for node substitution); outer
  `add_edges()` is NOT overridden

## Purpose

Retail food-label allergen compliance **advisory screening**. Given
**normalized label/ingredient text** (not an image — OCR is explicitly out of
scope for v1), the agent retrieves relevant 食品表示基準 (Food Labeling
Standards) allergen provisions from a seeded, versioned knowledge base
spanning both legal tiers — mandatory (特定原材料, 9 items) and recommended
(特定原材料に準ずるもの, a representative subset) — and returns a cited,
tier-tagged advisory screening result flagging each allergen as disclosed /
missing / ambiguous, for a qualified human to confirm.

**This agent is an advisory screening assistant, NOT the compliance
authority.** It abstains on low-confidence input and never issues an
authoritative pass/fail verdict. See "Safety Boundary" below.

## Caller-Data Contract (input_context channel)

Label text can be submitted on two channels:

- **`input` (plain text)** — the input text itself is the label text. Suited
  to Japanese labels. The framework's input gate masks Title-Case name-like
  runs in input-text fields before any node executes, so **English** label
  text on this channel arrives with multi-word ingredient names ("Whole Milk
  Powder", "Cashew Nut") already rewritten — and a disclosed allergen would
  then be falsely reported missing.
- **`input_context["label_request"]` (structured record — preferred)** —
  `{"label_text": str, "include_recommended": bool?, "top_k": int?}`. The
  framework does not rewrite `input_context`, so ingredient and product names
  survive to the matcher. Because the framework also does not scan this
  channel, `PreProcessNode` owns ALL screening for it (see Input Screening
  below).

The framework does not forward `input_context` across the outer→inner graph
boundary (`GraphNode` invokes the subgraph with the input string only), so the
screened record crosses via a per-invocation `ContextVar` bridge
(`src/graph/context_bridge.py`): the outer `extract_input()` stashes it and
`DomainWorkflowGraph._extra_initial_state()` seeds it into inner state as
`screening_request`.

The HTTP adapter (`src/api/server.py`) accepts `input_context` on
`POST /invoke` and enforces a coarse serialized-size cap (256 KiB) before the
graph runs.

### Input contract (label_request)

| Field | Type | Bounds | On violation |
|-------|------|--------|--------------|
| `label_text` | str (required) | 1–4000 chars; printable (newlines/tabs allowed, other control chars refused) | decline, naming the field |
| `include_recommended` | bool (optional) | strict boolean | decline, naming the field |
| `top_k` | int (optional) | finite integer in [1, 20] — NaN/Infinity refused | decline, naming the field |
| any other field | — | not part of the contract | decline (field name never echoed) |

These fail CLOSED with a field-naming reason: nothing is screened and the
rejected value is never echoed into the error log or the response. Each is a
value the caller can correct, so the run completes carrying the reason rather
than terminating — see *Declining a request without terminating the run* under
Security Design. A chat-template control sequence in any of these fields is the
other case and terminates. Absent optional fields fall back to the configured
defaults. Absent `label_request` degrades to the plain-text path.

## Architecture Overview

### Outer backbone (AgentBaseGraph)

```
START → initialize → pre_process → main → {route} → post_process → finalize → END
                                     ↓ (retry, up to the configured max_retry)
                                   pre_process
```

| Slot | Class | Responsibility | required_trust_level |
|------|-------|----------------|----------------------|
| initialize | InitializeNode (framework default) | session_id, trust_level, schema_version | — (framework) |
| pre_process | `PreProcessNode` | caller trust gate + label-request screening → `validated_input`, `screening_request` | `TrustLevel.VERIFIED_EXTERNAL` |
| main | `AllergenScreeningGraphNode` (`GraphNode`) | delegates to inner `DomainWorkflowGraph`; maps inner `formatted_answer`/`citations`/`screening_status`/`overall_abstain` → outer state | — (GraphNode delegation) |
| post_process | `PostProcessNode` | output content gate — module-level `_security_gate_output()` RECURSIVELY scans `result` for credentials/PII → ERROR + the closed-set envelope `{"reason": "output_withheld"}` (`result` cleared, violation type on `error_log` only); on the clean path appends the non-suppressible advisory stamp | `TrustLevel.VERIFIED_EXTERNAL` |
| finalize | FinalizeNode (framework default) | response_metadata, total_time_ms | — (framework) |

### Inner graph (DomainWorkflowGraph — BaseGraph, linear)

```
START → input_validate → retrieve → rerank_filter → generate_answer → output_format → END
```

All five inner domain nodes declare `required_trust_level = TrustLevel.ANONYMOUS`
(the external trust gate lives on the outer backbone gate node; a stricter inner
level would deny a real VERIFIED_EXTERNAL invoke at runtime).

| Node | Responsibility | required_trust_level | Input State | Output State |
|------|----------------|----------------------|-------------|---------------|
| `InputValidateNode` | Resolve the bridged `screening_request` (preferred) or parse the (possibly JSON-enveloped) input text; normalise whitespace; cap length; fail-closed validation of `include_recommended`/`top_k`; control-token screen + personal-data masking on the text path | `TrustLevel.ANONYMOUS` | `screening_request` \| `validated_input` \| `user_input` | `label_text`, `query_filters`, `intake_notes` |
| `RetrieveNode` | Deterministic explicit/implicit term matching over the seeded KB (`config/kb/allergen_kb.json`): substring-match each entry's `explicit_terms`/`implicit_hint_terms` against the label text | `TrustLevel.ANONYMOUS` | `label_text`, `retrieval_config` | `matched_provisions`, `intake_notes` |
| `RerankFilterNode` | Classify each match into disclosed/ambiguous/missing by `score_threshold`; apply `include_recommended` scope + `top_k` cap (mandatory tier never capped); compute the SAFETY-CRITICAL `overall_abstain` gate | `TrustLevel.ANONYMOUS` | `matched_provisions`, `query_filters`, `retrieval_config`, `label_text` | `screening_status`, `overall_abstain` |
| `GenerateAnswerNode` | Rule-based, cited, tier-grouped advisory body assembly from `screening_status` only (v1 deterministic — LLM synthesis seam documented below); emits ONLY the abstention message when `overall_abstain` | `TrustLevel.ANONYMOUS` | `screening_status`, `overall_abstain` | `grounded_answer`, `citations` |
| `OutputFormatNode` | Compose the final advisory body: grouped statuses + Sources list. **Does NOT add the advisory disclaimer** — that lives in `post_process`, see Safety Boundary | `TrustLevel.ANONYMOUS` | `grounded_answer`, `citations` | `formatted_answer`, `status` |

### Data Flow

```
input_context["label_request"] (structured record) + user_input (caller text)
  → PreProcessNode (trust gate + screening)        → validated_input, screening_request
  → AllergenScreeningGraphNode.extract_input       → stashes screening_request (ContextVar bridge)
                                                     → inner DomainWorkflowGraph.invoke(validated_input)
        → _extra_initial_state()                   → seeds retrieval_config + screening_request
        → input_validate                           → label_text / query_filters
        → retrieve                                 → matched_provisions
        → rerank_filter                            → screening_status / overall_abstain
        → generate_answer                          → grounded_answer / citations
        → output_format                            → formatted_answer
     get_output() → {formatted_answer, citations, screening_status, overall_abstain, status, error_log, ...}
  → AllergenScreeningGraphNode.merge_output        → result = formatted_answer, allergen_screening_result,
                                                        citations, screening_status, overall_abstain
  → PostProcessNode (output gate)                  → formatted_output (gated + advisory-stamped)
```

On the plain-text path (no `label_request`), the caller may still supply a
JSON envelope (`{"text": ..., "include_recommended": ..., "top_k": ...}`)
inside the input text; it passes through `validated_input` as a string and the
FIRST inner node (`InputValidateNode`) parses it back with the same
fail-closed field validation as the structured channel.

### Runtime config forwarding (`_parent_config`)

Declared runtime parameters live in `config/config.yaml` (separate from the
static registry manifest `config/agent.yaml`). The platform registry loads it
and passes it as `Graph(config=...)`; the standalone server mirrors that
construction via `_runtime_config()`. `register_nodes()` hands the same dict
to the main-slot node, and `AllergenScreeningGraphNode._parent_config()`
forwards the `retrieval` + `llm` blocks under `config["configurable"]`
(never `{}`):

```
{"configurable": {"retrieval": {score_threshold, top_k, kb_path, abstain_min_chars}, "llm": {...}}}
```

`get_subgraph()` passes this into `DomainWorkflowGraph(config=...)`; the inner
graph republishes the `retrieval` block into the inner initial state as the
JSON-string field `retrieval_config` (via `_extra_initial_state()`), so the
declared `score_threshold` / `top_k` / `abstain_min_chars` are live at
runtime. `RetrieveNode` and `RerankFilterNode` read `retrieval_config` from
state, falling back to module defaults that mirror the declared values.
**There is no `config` parameter on any node's `execute()`** — config reaches
inner nodes exclusively through State seeding, never a second positional
argument.

### State Definition

| Field | Type | Purpose | Layer |
|-------|------|---------|-------|
| `validated_input` | `NotRequired[str]` | screened caller text | outer |
| `screening_request` | `NotRequired[Optional[str]]` (JSON) | screened `label_request` record (bridged to the inner graph) | both |
| `enriched_context` | `NotRequired[Optional[str]]` (JSON) | channel metadata written by `PreProcessNode` | outer |
| `allergen_screening_result` | `NotRequired[str]` | final answer, mapped from inner `formatted_answer` | outer |
| `label_text` | `NotRequired[str]` | normalised label/ingredient text (NOT an image; no OCR) | inner |
| `query_filters` | `NotRequired[Optional[str]]` (JSON) | `{"include_recommended": bool\|None, "top_k": int\|None}` | inner |
| `retrieval_config` | `NotRequired[Optional[str]]` (JSON) | forwarded runtime `retrieval` block | inner |
| `matched_provisions` | `NotRequired[Optional[str]]` (JSON) | per-allergen match candidates (all mandatory + matched recommended) | inner |
| `screening_status` | `NotRequired[Optional[str]]` (JSON) | final per-allergen `disclosed`/`missing`/`ambiguous` statuses | inner |
| `overall_abstain` | `NotRequired[bool]` | SAFETY BOUNDARY — True when input confidence is too low for a verdict | inner |
| `grounded_answer` | `NotRequired[str]` | rule-assembled advisory body | inner |
| `citations` | `NotRequired[Optional[str]]` (JSON) | `[{ref, id, allergen_en, allergen_ja, tier, source}]` | inner |
| `formatted_answer` | `NotRequired[str]` | final body (no disclaimer — see Safety Boundary) | inner |
| `intake_notes` | `NotRequired[Optional[str]]` (JSON) | validation / parse notes (no PII) | inner |
| `trace_id` / `correlation_id` | `Optional[str]` | framework-managed tracing | both |

**State Constraints (mandatory):**
- Flat `TypedDict` only (primitives + JSON-serialisable types).
- Structured fields (dict / list[dict]) stored as JSON STRINGS via `to_json()` /
  `from_json()` — used consistently by every producer AND consumer (msgpack
  checkpoint safety). `overall_abstain` is the one domain field that is a bare
  scalar (`bool`) and is typed `NotRequired[bool]` accordingly — no JSON
  serialisation needed for a scalar.
- Domain fields are `NotRequired[...]` (valid TypedDict before any node writes).
- `formatted_output` is NOT re-declared (backbone field stays framework-owned).
- No JWT, API keys, credentials, or raw personal identifiers in State.
- `InvocationContext` via `config["configurable"]` only (never in State).
- No Pydantic models / dataclasses / arbitrary Python objects.

## Security Design

### Caller trust

Every node declares `required_trust_level` (see tables above);
`PreProcessNode` (VERIFIED_EXTERNAL) declines empty / non-string `user_input`
before the inner workflow runs — a value the caller can correct, so the run
completes carrying the reason and the workflow does no work. The standalone
server elevates authenticated Bearer callers to VERIFIED_EXTERNAL
(`INVOKE_AUTH_TOKEN`); a caller the trust gate denies is a terminating case.

### Input screening

The framework `FunctionNode` default scan masks personal data in
`user_input` / `validated_input` / `llm_response` at every node boundary and
rejects high-confidence injection content in the actionable input fields. The
template additionally owns its own screens — they hold even where no
framework wrapper runs, and they are the ONLY screens on the `input_context`
channel (which the framework does not scan):

- **Control-token screen** (`src/nodes/validation.py`): chat-template control
  tokens (`<|...|>`, `[INST]`, `<<SYS>>`) are refused as a CLASS, post-parse,
  over every string — values AND keys, at any nesting depth — on both
  channels. Matched tokens go to the audit trail, never back to the caller.
- **Finite-number rule**: every caller-controlled number goes through
  `finite_in_range` (rejects bools, non-numerics, NaN/±Infinity — which parse
  fine via `float()` AND arrive via raw JSON literals — and out-of-range
  magnitudes). Fail CLOSED with a field-naming reason, on the declining path:
  nothing is screened, and the run completes so the caller can correct the
  figure.
- **Personal-data masking on label text**: high-precision patterns (card
  numbers, national IDs, e-mail addresses, phone numbers) are MASKED, not
  refused — a manufacturer's contact line is legitimate on a retail label,
  and the screening needs the rest of the text. The broad Title-Case name
  heuristic is deliberately NOT applied on this channel: ingredient and
  product names are legitimately Title-Case, and masking them destroys the
  allergen match itself (the reason the structured channel exists).

### Output gate

`PostProcessNode` calls the module-level `_security_gate_output()` scan from
`execute()` — API keys / JWT / Bearer tokens / credential assignments **and
PII** (e-mail, long digit runs) in the final `result` refuse the response:
the node returns `AgentStatus.ERROR` with `formatted_output` set to the
closed-set error envelope `{"reason": "output_withheld"}`, `result` cleared,
and the violation *type* on `error_log` only. Unlike a generic
top-level-string-only scan, this gate **recurses into nested dict/list/tuple
content** (signature `content: Any`) — the same function is reused by
`FoodLabelAllergenComplianceAgent.get_output()` to defensively re-scan the
structured payload (`screening_status`, `citations`) before it is surfaced to
a programmatic caller. No `_extra_security_gate_input` /
`_extra_security_gate_output` instance methods are defined on any node.

### Output boundary — no numeric rewriting layer

The advisory output contains **no monetary aggregates and no
caller-controlled figures**: every rendered value is KB-derived (allergen
names, tier/status labels, match scores in [0, 1], citation reference
numbers, provision sources) or a fixed template string. Label text itself is
never echoed into the output. Consequently the output boundary applies **no
numeric rounding/aggregation grid** — there is no figure whose precision
could leak anything, and coarsening match scores would only obscure the
screening result. The enforced output invariants are instead:

- the credential/PII content gate above, on BOTH representations (rendered
  text and structured payload), each fail-closed;
- the non-suppressible advisory stamp on every clean text response and the
  constant `advisory_notice` field on every SUCCESS structured response;
- the abstention contract (empty `screening_status`, abstention message) on
  low-confidence input;
- the caller-visible error is a closed set: on any status that **terminates**,
  the invoke body carries `output: null` and `error: {"reason": …}` with the
  reason drawn from `ERROR_REASONS` — `output_withheld` when the gate above
  refused the response, `workflow_failed` for every other terminating outcome
  (a control-sequence refusal, a trust denial, an inner-workflow error; the
  backbone routes all of them straight to `finalize`). `error_log` is
  never projected: it is the internal channel (state reducer, audit trail)
  and carries node- and framework-authored text, including a wrapped
  exception's message and traceback with runtime paths;
  `on_subgraph_error()` carries an inner rejection onto it as single-line
  entries only. Not publishing it is the contract — filtering it would not
  be.

### Declining a request without terminating the run

A request that is not screened stops in one of two ways, chosen by what the
caller can do about it:

- **A value the caller can correct** — no label text, text past the size
  bound, a malformed `top_k` or `include_recommended`, an unsupported
  `label_request` field. The run **completes**, carrying a reason code
  (`EMPTY_INPUT`, `QUESTION_TOO_LONG`, `INVALID_REQUEST`) through state. No
  `screening_request` or `label_text` is published, so nothing is screened;
  every node downstream of the stop passes the code along untouched instead
  of doing its own work, and `PostProcessNode` turns it into the fixed
  caller-facing sentence held in `src/services/failure_message.py`.
  `get_output()` recognises the code and returns the base envelope as it
  stands: a successful status, the sentence as the body, no `error` key and
  no structured field. Terminating instead would end the calling surface's
  turn and surface only a closed-set label, leaving what to correct reachable
  solely from the audit trail — the caller could not fix the value and send
  the request again on the same conversation. The reason code is an internal
  state field and is never published in the envelope: the caller reads the
  sentence, not the code.
- **A refusal the agent owns** — a chat-template control sequence on either
  channel, an output-gate violation, a trust denial, an inner-workflow error.
  These **terminate**, and the closed-set envelope above is what the caller
  receives. Rewording the request is not a route past them, so presenting
  them as correctable would be a false statement about the agent's behaviour.

The branch is chosen at the call site, by the reason code the stop carries,
not by inspecting the message — so a new refusal defaults to terminating
rather than to completing. The closed-set discipline holds on both paths: the
declined body is a module constant, and nothing a node authored reaches the
caller either way.

### Audit logging

Every node's `execute()` emits exactly ONE domain-specific
`emit_trace_event("<node>_complete", {small non-PII payload}, state)` on its
success path, plus screening-decision events on the rejection paths
(`pre_process_validation_failed`, `pre_process_injection_blocked`,
`pre_process_personal_data_masked`, `input_validate_rejected`). Nodes do NOT
emit `node_start` / `node_complete` / `node_error` — `BaseNode.__call__()`
emits those. Domain event names:

- `pre_process_complete`
- `input_validate_complete`
- `retrieve_complete`
- `rerank_filter_complete`
- `generate_answer_complete`
- `output_format_complete`
- `post_process_complete`

## Safety Boundary

> This is a food-allergen template — a false "disclosed" could hurt someone.
> The following are hard design invariants, not aspirations.

1. **Never an authoritative verdict.** No node, at any layer, ever emits a
   pass/fail, compliant/non-compliant, or safe/unsafe determination. Output is
   always a set of factual, per-allergen `disclosed` / `missing` / `ambiguous`
   statements. `GenerateAnswerNode` and `OutputFormatNode` contain no such
   verdict vocabulary anywhere in their code or output templates.
2. **Abstain on low confidence, for real.** `RerankFilterNode` computes
   `overall_abstain` from the normalised input length
   (`abstain_min_chars`, default 8). When True, it emits an EMPTY
   `screening_status` (`to_json([])`) — not a partial or hedged list — so
   `GenerateAnswerNode` structurally cannot synthesise a per-allergen verdict
   from it; it emits only the abstention message and empty citations. This is
   mechanically enforced (data-level), not just a prompt/wording choice, and
   is testable directly (feed a short/empty label text, assert
   `overall_abstain=True` and `screening_status=="[]"`).
3. **The advisory stamp is non-suppressible.** "This screening result is
   advisory only — human review required." is appended by `PostProcessNode`
   (the one output gate every response passes through unconditionally on its
   clean path) — deliberately NOT by `OutputFormatNode` or any other domain
   node, so no input or code path can produce a stamped-result response that
   skips it. The structured `get_output()` payload carries the equivalent
   `advisory_notice` field as a fixed constant on every SUCCESS response.
4. **No OCR claim, anywhere.** v1 takes normalized text only. No node, doc,
   README, or config references a live OCR call or claims image-input
   capability.
5. **Mandatory tier can never be narrowed away.** The caller-controlled
   `include_recommended` filter only ever affects the RECOMMENDED tier's
   surfacing; the 9 mandatory allergens are always screened and always
   present in `screening_status` (as `disclosed`/`missing`/`ambiguous`),
   regardless of caller input.
6. **Ingredient names must survive intake.** An English label's Title-Case
   ingredient runs are proof of disclosure; any intake step that rewrites
   them turns a disclosed allergen into a false "missing". The structured
   channel + masking design above exists for this invariant, and tests pin
   named ingredients ("Whole Milk Powder", "Cashew Nut") surviving
   end-to-end to a disclosed status.

## v1 Implementation Note — LLM synthesis

v1 of this template is **deterministic end-to-end**: matching is
explicit/implicit substring scoring over the seeded KB and
`GenerateAnswerNode` assembles the advisory body rule-based from the
classified `screening_status` (grouped per-allergen lines + citations). There
is NO live LLM call, NO live OCR model call, and NO live vector store in v1 —
the `llm` config block is forwarded through `_parent_config()` for
forward-compatibility but is not consumed by any v1 node, and no
`system_prompt` is read at runtime. The LLM synthesis upgrade seam is
documented in `config/prompts/allergen_synthesis_prompt.md`: a v2
`GenerateAnswerNode` swaps the rule-based assembly for an LLM call that
synthesises over the same `screening_status` input and emits the same
`grounded_answer` / `citations` state contract — including the same
abstain-on-low-confidence and no-verdict safety rules — so no other node
changes.

## Composition Pattern

- **Pattern:** `GraphNode` (subgraph) in the outer `main` slot.
- **Composition target:** `DomainWorkflowGraph` (inner `BaseGraph`).
- **Error strategy:** `handle` — an inner failure fails the invocation CLOSED
  via `on_subgraph_error()`, surfacing only clean, single-line, field-naming
  messages; a raw exception trace never crosses the graph boundary.
- Inner domain nodes run at `TrustLevel.ANONYMOUS`; outer pre/post_process run
  at `TrustLevel.VERIFIED_EXTERNAL`.

## Import Isolation Confirmation
- [x] Template does not import the platform-internal SDK layer.
- [x] Import targets: `framework/` and `shared/` only.
- [x] Framework base classes only in base positions (`AgentBaseGraph`,
      `BaseGraph`, `GraphNode`, `FunctionNode`).

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Base type | AgentBaseGraph | AutonomousBaseGraph | **AgentBaseGraph** | Fixed multi-step RAG-style workflow, no autonomous loop |
| Composition pattern | Standalone slots | GraphNode → inner BaseGraph | **GraphNode → inner BaseGraph** | 5-step domain workflow exceeds a single `main` node; nested keeps the outer backbone untouched |
| English label channel | Plain input text | `input_context` structured record + ContextVar bridge | **Structured record** | The framework's Title-Case masking on input-text fields rewrites multi-word ingredient names before matching — a disclosed allergen would be falsely reported missing; `input_context` is not rewritten, and the template screens it itself |
| Personal data in label text | Refuse the request | Mask the spans, keep the label | **Mask** | A manufacturer's contact line is legitimate on a retail label; refusing would block real work, masking preserves both hygiene and the allergen match |
| Caller numeric validation | Clamp/ignore out-of-contract values | Finite+bounded parse, fail closed | **Fail closed** | NaN/Infinity parse fine and compare False; silently clamping hides caller errors — a field-naming reason is actionable and never echoes the value |
| Reporting a fail-closed caller value | Terminate the run | Complete the run carrying a reason code | **Complete** | Terminating ends the calling surface's turn and surfaces a closed-set label only, leaving what to correct reachable solely from the audit trail; completing lets the caller fix the value and send the request again on the same conversation. Refusals the agent owns — a control sequence, an output-gate violation, a trust denial — still terminate, because rewording is not a route past them |
| Inner error strategy | Re-raise inner failures | `handle` + single-line messages on the internal channel | **handle** | Fail closed; the inner rejection's single-line message is carried onto the outer `error_log` for the audit trail (exception traces carry runtime paths and are dropped). A terminating inner failure gives the caller the closed-set `error.reason` only; an inner rejection of a correctable caller value carries its reason code out of the subgraph instead, and the run completes |
| Advisory stamp placement | OutputFormatNode (domain layer) | PostProcessNode (output gate, backbone) | **PostProcessNode** | Every response passes through the output gate unconditionally, making the stamp genuinely non-suppressible — a domain-layer stamp could be skipped by a future node change |
| Answer synthesis | Rule-based assembly | LLM call | **Rule-based (v1)** | Deterministic assembly is testable and auditable for a safety-critical advisory tool; v2 swaps the LLM in at the documented seam |
| Label-text ingestion | Live OCR model call | Normalized text input (OCR out of scope) | **Normalized text (v1)** | Keeps the build deterministic and network-free |
| KB storage | External vector store | Seeded JSON KB | **Seeded JSON KB (v1)** | Self-contained, deterministic CI; the match contract (`matched_provisions` JSON) is store-agnostic for a later vector-store upgrade |
| Abstain mechanism | Prompt-level hedging only | Data-level empty-verdict gate (RerankFilterNode) | **Data-level gate** | A mechanically enforced empty `screening_status` cannot be bypassed by wording drift in a later prompt/LLM change |
