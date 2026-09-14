"""Typed data contracts shared across the pipeline.

Keeping these as pydantic models (rather than passing raw dicts between agents)
is what makes the LangGraph state machine, the LLM structured-output calls, and
the eval harness all validate against the same shapes.
"""
from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    txn_id: str
    account_id: str
    counterparty_id: str
    amount: float
    currency: str = "USD"
    timestamp: str
    channel: Literal["wire", "ach", "card", "cash", "mobile"]
    country: str


class Account(BaseModel):
    account_id: str
    customer_name: str
    risk_rating: Literal["low", "medium", "high"]
    kyc_country: str
    account_age_days: int


class TriageResult(BaseModel):
    txn_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    decision: Literal["dismiss", "escalate"]
    triggered_rules: List[str]
    rationale: str


class EvidenceItem(BaseModel):
    txn_id: str
    detail: str


class InvestigationDossier(BaseModel):
    subject_account: str
    timeline: List[EvidenceItem]
    pattern_summary: str


class SARDraft(BaseModel):
    narrative: str
    cited_txn_ids: List[str]
    version: int = 1


class CriticFeedback(BaseModel):
    approved: bool
    issues: List[str] = Field(default_factory=list)
    # citation_accuracy is computed in code (not by the LLM) -- see agents/critic.py
    citation_accuracy: float = 1.0


class CaseState(TypedDict, total=False):
    transaction: Transaction
    account: Account
    related_transactions: List[Transaction]
    triage: Optional[TriageResult]
    dossier: Optional[InvestigationDossier]
    sar_draft: Optional[SARDraft]
    critic_feedback: Optional[CriticFeedback]
    revision_count: int
    final_status: str
