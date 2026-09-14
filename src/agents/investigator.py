"""Investigator agent: turns a triaged transaction into an evidence timeline.

Only runs for escalated cases -- this is the "multi-step reasoning over
tools/context" node: it consumes the related-transaction context that was
pulled for this case and produces a structured dossier the narrative writer
can cite from. It does not decide guilt; it assembles facts.
"""
from __future__ import annotations

from ..llm import LLMClient
from ..schemas import CaseState, InvestigationDossier


def make_investigator_node(llm: LLMClient):
    def investigator_node(state: CaseState) -> dict:
        payload = {
            "transaction": state["transaction"].model_dump(),
            "account": state["account"].model_dump(),
            "related_transactions": [t.model_dump() for t in state.get("related_transactions", [])],
            "triage": state["triage"].model_dump(),
        }
        dossier = llm.structured(kind="investigate", payload=payload, schema=InvestigationDossier)
        return {"dossier": dossier}

    return investigator_node
