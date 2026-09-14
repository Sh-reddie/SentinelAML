"""LLM client abstraction.

Two implementations, one interface:

- MockLLMClient: deterministic, offline, zero-cost. Implements the same
  decision logic a prompted model would be asked to follow, in plain Python.
  This is what `pytest` and CI run against, so the whole graph is exercised
  on every commit without an API key or a bill.
- RealLLMClient: wraps a LangChain chat model (Anthropic or OpenAI) with
  structured output, using the prompts in prompts.py.

Every agent depends only on `LLMClient.structured(...)`, never on a provider
or on whether it's talking to the mock -- that's what makes the swap safe.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    @abstractmethod
    def structured(self, *, kind: str, payload: Dict[str, Any], schema: Type[T]) -> T:
        """Run one structured call for agent `kind`, returning an instance of `schema`."""


class RealLLMClient(LLMClient):
    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
        self.provider = provider
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            self._model = ChatAnthropic(model=model or os.getenv("LLM_MODEL") or "claude-sonnet-4-5")
        elif provider == "openai":
            from langchain_openai import ChatOpenAI

            self._model = ChatOpenAI(model=model or os.getenv("LLM_MODEL") or "gpt-4o-mini")
        else:
            raise ValueError(f"Unknown LLM provider: {provider!r}")

    def structured(self, *, kind: str, payload: Dict[str, Any], schema: Type[T]) -> T:
        from .prompts import build_prompt

        system, user = build_prompt(kind, payload)
        structured_model = self._model.with_structured_output(schema)
        return structured_model.invoke([("system", system), ("user", user)])


class MockLLMClient(LLMClient):
    """No network calls. See module docstring."""

    def structured(self, *, kind: str, payload: Dict[str, Any], schema: Type[T]) -> T:
        handler = getattr(self, f"_handle_{kind}", None)
        if handler is None:
            raise ValueError(f"MockLLMClient has no handler for kind={kind!r}")
        return schema(**handler(payload))

    @staticmethod
    def _handle_triage(payload: Dict[str, Any]) -> Dict[str, Any]:
        triggered = payload["triggered_rules"]
        account = payload["account"]
        risk_bump = {"low": 0.0, "medium": 0.15, "high": 0.3}.get(account["risk_rating"], 0.0)
        risk_score = min(1.0, 0.2 * len(triggered) + risk_bump)
        decision = "escalate" if triggered or risk_score > 0.25 else "dismiss"
        rationale = (
            f"Triggered rules: {', '.join(triggered) if triggered else 'none'}; "
            f"account risk rating: {account['risk_rating']}."
        )
        return {
            "txn_id": payload["transaction"]["txn_id"],
            "risk_score": round(risk_score, 2),
            "decision": decision,
            "triggered_rules": triggered,
            "rationale": rationale,
        }

    @staticmethod
    def _handle_investigate(payload: Dict[str, Any]) -> Dict[str, Any]:
        txn = payload["transaction"]
        related = payload["related_transactions"]
        timeline = [
            {
                "txn_id": txn["txn_id"],
                "detail": (
                    f"Subject transaction: {txn['amount']} {txn['currency']} via "
                    f"{txn['channel']} to {txn['counterparty_id']} on {txn['timestamp']}."
                ),
            }
        ]
        for r in related:
            timeline.append(
                {
                    "txn_id": r["txn_id"],
                    "detail": (
                        f"Related transaction: {r['amount']} {r['currency']} via "
                        f"{r['channel']} between {r['account_id']} and {r['counterparty_id']} "
                        f"on {r['timestamp']}."
                    ),
                }
            )
        rules = payload["triage"]["triggered_rules"]
        pattern_summary = (
            f"{len(related) + 1} transaction(s) reviewed for account {txn['account_id']}. "
            f"Pattern indicators: {', '.join(rules) if rules else 'none beyond the subject transaction'}."
        )
        return {"subject_account": txn["account_id"], "timeline": timeline, "pattern_summary": pattern_summary}

    @staticmethod
    def _handle_narrative(payload: Dict[str, Any]) -> Dict[str, Any]:
        dossier = payload["dossier"]
        notes = payload.get("revision_notes") or []
        cited = [item["txn_id"] for item in dossier["timeline"]]
        lines = [f"On {item['txn_id']}: {item['detail']}" for item in dossier["timeline"]]
        narrative = (
            f"Subject account {dossier['subject_account']} was reviewed following automated triage. "
            f"{dossier['pattern_summary']} " + " ".join(lines)
        )
        if notes:
            narrative += " Revision addressed: " + "; ".join(notes)
        return {"narrative": narrative, "cited_txn_ids": cited, "version": 1}

    @staticmethod
    def _handle_critic(payload: Dict[str, Any]) -> Dict[str, Any]:
        sar = payload["sar_draft"]
        valid_ids = set(payload["valid_txn_ids"])
        cited_ids = set(sar["cited_txn_ids"])
        issues = []
        if not cited_ids:
            issues.append("Narrative cites no transactions.")
        if not cited_ids.issubset(valid_ids):
            issues.append("Narrative cites unknown transaction id(s).")
        if len(sar["narrative"]) < 40:
            issues.append("Narrative is too short to contain the required elements.")
        approved = not issues
        return {"approved": approved, "issues": issues, "citation_accuracy": 1.0 if not issues else 0.0}


def get_llm_client() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "mock")
    if provider == "mock":
        return MockLLMClient()
    return RealLLMClient(provider=provider)
