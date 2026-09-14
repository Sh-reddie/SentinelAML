# SentinelAML

A multi-agent transaction-fraud investigation system that goes one step past
scoring a transaction: it investigates the flagged case, drafts the
Suspicious Activity Report (SAR) narrative a compliance analyst would
otherwise write by hand, and runs that draft through an automated
compliance-critic loop before handing it to a human for final sign-off.

Most fraud-detection portfolio projects stop at "here's a model that scores
a transaction 0-1." That's necessary but not the bottleneck -- in practice,
analysts spend most of their time writing the narrative, not scoring the
transaction. SentinelAML targets that second, harder problem.

## Why this design

Financial-crime narratives get filed with regulators. An LLM that
occasionally invents a supporting fact is not a curiosity here, it's a
liability. So the architecture treats "does the narrative only claim things
it can prove" as a first-class, code-enforced check rather than something
the LLM is trusted to self-report -- see `src/agents/critic.py`.

## Architecture

```
                              ┌────────────┐
                    escalate  │   triage   │  dismiss
              ┌──────────────┤            ├───────────────┐
              │               └────────────┘               │
              v                                             v
      ┌───────────────┐                            ┌─────────────────┐
      │  investigator  │                            │ finalize_dismiss │──> END
      └──────┬───────┘                            └─────────────────┘
              v
      ┌───────────────┐   revise    ┌────────────┐
      │   narrative    │<────────────┤   critic   │
      └───────┬────────┘             └─────┬──────┘
              │                             │ approved
              v                             v
      (loops back to narrative      ┌──────────────────┐
       up to MAX_REVISIONS times)   │ finalize_approved │──> END
                                     └──────────────────┘
                                     max revisions hit -> finalize_escalate (human review) -> END
```

Four agents, wired as a [LangGraph](https://github.com/langchain-ai/langgraph) state machine:

1. **Triage** (`src/agents/triage.py`) -- runs on every transaction. Combines
   four deterministic AML rules (`src/rules.py`: structuring, rapid fan-out,
   round-trip layering, new-account-large-transaction) with LLM judgment on
   whether to escalate. Cheap, runs on 100% of volume.
2. **Investigator** (`src/agents/investigator.py`) -- only for escalated
   cases. Builds a factual, txn_id-anchored timeline from the subject
   transaction and its related transactions.
3. **Narrative writer** (`src/agents/narrative.py`) -- drafts the SAR from
   the dossier, citing a txn_id for every claim. Re-entered on each revision
   with the critic's prior issues, so a rewrite actually targets what failed.
4. **Critic** (`src/agents/critic.py`) -- the part that makes this more than
   a demo. Every `txn_id` the narrative cites is checked in code against the
   set of ids that actually exist in the case -- not asked of the LLM, verified.
   Only after that passes does an LLM judge qualitative completeness
   (missing elements, speculation stated as fact). Rejections loop back to
   the narrative writer, capped at `MAX_REVISIONS` before handing off to a
   human reviewer instead of looping forever.

## Two LLM modes, one interface

Every agent depends only on `LLMClient.structured(...)` (`src/llm.py`):

- `MockLLMClient` -- deterministic, offline, zero cost. Implements the same
  decision logic a prompted model would follow, in plain Python. This is
  what tests and CI run against, so the whole graph -- routing, the revision
  loop, the citation guard -- is exercised on every commit with no API key.
- `RealLLMClient` -- wraps a LangChain chat model (Anthropic or OpenAI) with
  structured output (`with_structured_output`), using the prompts in
  `src/prompts.py`.

Swap providers with one environment variable:

```bash
cp .env.example .env
# LLM_PROVIDER=mock (default) | anthropic | openai
```

## Data

No external dataset or real bank data -- `data/generate_synthetic_transactions.py`
generates a seeded, labeled case set (`data/cases.json`): benign cases with no
rule triggers, and suspicious cases deliberately constructed to trip each of
the four rules. Fully reproducible, zero privacy/compliance surface area.

```bash
python -m data.generate_synthetic_transactions
```

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env          # LLM_PROVIDER=mock works with no key at all

pytest -v                     # 11 tests, all pass in mock mode
python -m eval.eval_harness   # runs the full pipeline over the labeled case set

uvicorn src.api:app --reload  # POST /investigate, GET /health
# or: docker build -t sentinelaml . && docker run -p 8000:8000 sentinelaml
```

## Eval results (mock LLM, 24 labeled cases, reproducible via the command above)

| | Precision | Recall | F1 |
|---|---|---|---|
| Rules-only baseline | 1.00 | 1.00 | 1.00 |
| Full pipeline (triage decision) | 0.80 | 1.00 | 0.89 |

Mean citation accuracy across every SAR draft produced: **1.00**. Outcome mix:
4 dismissed, 20 approved, 0 escalated to human review, 0 average revisions
needed -- expected in mock mode, since the deterministic mock never
hallucinates a citation in the first place. `tests/test_critic_citation_guard.py`
exercises the rejection path directly, by constructing a draft that cites a
transaction id that does not exist in the case, so the critic's code-verified
guard is tested independent of which LLM (mock or real) is behind it.

The pipeline's recall matches the rules baseline (nothing the rules would
catch gets missed) but trades some precision for factoring account risk
context into the escalation call, not only hard-coded rule hits -- a
deliberate, explainable tradeoff for a fraud workflow where a missed true
positive is costlier than an extra human review. Run the harness against a
real model (`LLM_PROVIDER=anthropic` or `openai`) to compare live numbers
against this mock baseline.

## What's not here yet

- The related-transaction lookup is passed in by the caller rather than
  queried from a transaction store -- the next real step would be a tool-use
  agent that pulls this context itself.
- No persistence layer for case history or analyst overrides.
- `RealLLMClient` is wired but not eval'd against live numbers in this repo
  (see above) -- that's the natural next data point to add.

## Stack

Python, LangChain, LangGraph, FastAPI, Pydantic, MLflow, Docker, pytest.
