# Food Label Allergen Compliance Agent

AI agent for reading food labels and checking allergen compliance, built with Agentic Star.

> **Category**: Cat 2 (domain-specific retrieval pipeline)
> **Industry**: Retail
> **Template ID**: RET-C2-283

## Overview

Advisory screening of retail food-label ingredient text against the Japanese Food Labeling
Standards (食品表示基準) allergen rules. Given normalized label/ingredient text — Japanese or
English — the agent matches it against a seeded, versioned allergen knowledge base covering both
legal tiers (the 9 mandatory 特定原材料 and a recommended-tier subset) and returns a cited,
tier-tagged result flagging each allergen as `disclosed`, `missing` or `ambiguous`, plus a
structured payload for programmatic callers.

It is an **advisory screening assistant, not the compliance authority**: every response carries a
non-suppressible "advisory only — human review required" stamp, the agent abstains on input too
sparse to screen reliably, and it never issues a pass/fail verdict. v1 is deterministic and
network-free — no OCR, no LLM call, no vector store (see `docs/02_design.md`).

English label text should be submitted on the structured `input_context.label_request` channel
(see the caller-data contract in `docs/02_design.md` — the plain-text channel is subject to the
platform's name-masking and would corrupt multi-word English ingredient names before matching).

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent fails its
start-up checks and refuses to start rather than running in a partially working state. This is
intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
docs/         design and operational documentation
```

See `docs/` for the design specification and the test specification.

## Customising

1. Adjust `config/` for your own environment and policies.
2. Replace the knowledge sources and sample data with your own.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
