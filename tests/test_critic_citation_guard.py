"""Unit tests for the critic's code-verified citation check -- the part of
the system that catches a hallucinated citation regardless of what the LLM
(mock or real) claims about its own output."""
from src.agents.critic import make_critic_node
from src.llm import MockLLMClient
from src.schemas import Account, EvidenceItem, InvestigationDossier, SARDraft, Transaction, TriageResult


def _base_state():
    txn = Transaction(
        txn_id="TXN-1", account_id="ACC-1", counterparty_id="CP-1", amount=9500.0,
        currency="USD", timestamp="2026-08-01T10:00:00", channel="cash", country="US",
    )
    account = Account(account_id="ACC-1", customer_name="Test", risk_rating="medium", kyc_country="US", account_age_days=200)
    dossier = InvestigationDossier(
        subject_account="ACC-1",
        timeline=[EvidenceItem(txn_id="TXN-1", detail="Cash deposit just under the reporting threshold.")],
        pattern_summary="Single structuring indicator.",
    )
    triage = TriageResult(txn_id="TXN-1", risk_score=0.6, decision="escalate", triggered_rules=["structuring_near_threshold"], rationale="test")
    return {"transaction": txn, "account": account, "dossier": dossier, "triage": triage, "revision_count": 0}


def test_citation_to_real_transaction_is_approved():
    state = _base_state()
    state["sar_draft"] = SARDraft(narrative="On TXN-1 a cash deposit of 9500 USD was made, just under the reporting threshold.", cited_txn_ids=["TXN-1"])

    critic_node = make_critic_node(MockLLMClient())
    result = critic_node(state)

    assert result["critic_feedback"].citation_accuracy == 1.0
    assert result["critic_feedback"].approved is True


def test_citation_to_nonexistent_transaction_is_rejected():
    state = _base_state()
    state["sar_draft"] = SARDraft(
        narrative="On TXN-1 and TXN-999 large suspicious transfers occurred.",
        cited_txn_ids=["TXN-1", "TXN-999"],
    )

    critic_node = make_critic_node(MockLLMClient())
    result = critic_node(state)

    feedback = result["critic_feedback"]
    assert feedback.approved is False
    assert feedback.citation_accuracy == 0.5
    assert any("citation" in issue.lower() for issue in feedback.issues)
