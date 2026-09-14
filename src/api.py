"""FastAPI service exposing the investigation graph as `/investigate`."""
from __future__ import annotations

from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from .graph import build_graph
from .schemas import Account, CriticFeedback, InvestigationDossier, SARDraft, Transaction, TriageResult

load_dotenv()

app = FastAPI(
    title="SentinelAML",
    description="Multi-agent transaction-fraud investigation and SAR-drafting service.",
    version="0.1.0",
)

_graph = build_graph()


class InvestigateRequest(BaseModel):
    transaction: Transaction
    account: Account
    related_transactions: List[Transaction] = []


class InvestigateResponse(BaseModel):
    final_status: str
    triage: Optional[TriageResult] = None
    dossier: Optional[InvestigationDossier] = None
    sar_draft: Optional[SARDraft] = None
    critic_feedback: Optional[CriticFeedback] = None
    revision_count: int = 0


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/investigate", response_model=InvestigateResponse)
def investigate(request: InvestigateRequest) -> InvestigateResponse:
    initial_state = {
        "transaction": request.transaction,
        "account": request.account,
        "related_transactions": request.related_transactions,
        "revision_count": 0,
    }
    final_state = _graph.invoke(initial_state)
    return InvestigateResponse(**final_state)
