"""Triage agent: decide escalate vs. dismiss for one transaction.

Deliberately cheap and rule-anchored -- this node runs on every transaction,
so it should not be where the expensive reasoning happens. It hands off to
the investigator only when there's a real signal.
"""
from __future__ import annotations

from ..llm import LLMClient
from ..rules import triggered_rules
from ..schemas import CaseState, TriageResult


def make_triage_node(llm: LLMClient):
    def triage_node(state: CaseState) -> dict:
        txn = state["transaction"]
        account = state["account"]
        related = state.get("related_transactions", [])

        rules_hit = triggered_rules(txn, account, related)
        payload = {
            "transaction": txn.model_dump(),
            "account": account.model_dump(),
            "triggered_rules": rules_hit,
        }
        result = llm.structured(kind="triage", payload=payload, schema=TriageResult)
        return {"triage": result}

    return triage_node


def route_after_triage(state: CaseState) -> str:
    triage = state["triage"]
    return "investigate" if triage.decision == "escalate" else "dismiss"
