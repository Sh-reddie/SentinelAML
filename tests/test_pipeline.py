from src.graph import build_graph
from src.llm import MockLLMClient
from src.schemas import Account, Transaction


def _run(transaction, account, related=None):
    graph = build_graph(llm=MockLLMClient())
    return graph.invoke(
        {"transaction": transaction, "account": account, "related_transactions": related or [], "revision_count": 0}
    )


def test_benign_case_is_dismissed_without_a_sar():
    txn = Transaction(
        txn_id="TXN-B0", account_id="ACC-B0", counterparty_id="CP-1", amount=120.0,
        currency="USD", timestamp="2026-08-01T09:00:00", channel="card", country="US",
    )
    account = Account(account_id="ACC-B0", customer_name="Jane Doe", risk_rating="low", kyc_country="US", account_age_days=900)

    result = _run(txn, account)

    assert result["final_status"] == "dismissed"
    assert result["triage"].decision == "dismiss"
    assert result.get("sar_draft") is None


def test_structuring_case_escalates_and_produces_a_cited_sar():
    txn = Transaction(
        txn_id="TXN-S0", account_id="ACC-S0", counterparty_id="CP-1", amount=9600.0,
        currency="USD", timestamp="2026-08-05T09:00:00", channel="cash", country="US",
    )
    related = [
        Transaction(
            txn_id="TXN-S1", account_id="ACC-S0", counterparty_id="CP-2", amount=9400.0,
            currency="USD", timestamp="2026-08-05T05:00:00", channel="cash", country="US",
        )
    ]
    account = Account(account_id="ACC-S0", customer_name="John Roe", risk_rating="medium", kyc_country="US", account_age_days=400)

    result = _run(txn, account, related)

    assert result["triage"].decision == "escalate"
    assert "structuring_near_threshold" in result["triage"].triggered_rules
    assert result["final_status"] in ("approved", "escalated_to_human_review")

    sar = result["sar_draft"]
    valid_ids = {"TXN-S0", "TXN-S1"}
    assert set(sar.cited_txn_ids).issubset(valid_ids)
    assert result["critic_feedback"].citation_accuracy == 1.0


def test_max_revisions_caps_the_critic_loop():
    # A dossier with zero timeline entries makes it impossible for the mock
    # narrative agent to ever cite a valid transaction -- this forces the
    # critic to reject every revision, which must terminate rather than loop.
    from src.agents.critic import MAX_REVISIONS

    txn = Transaction(
        txn_id="TXN-X0", account_id="ACC-X0", counterparty_id="CP-1", amount=15000.0,
        currency="USD", timestamp="2026-08-20T09:00:00", channel="wire", country="US",
    )
    account = Account(account_id="ACC-X0", customer_name="New Corp", risk_rating="high", kyc_country="AE", account_age_days=3)

    result = _run(txn, account)

    assert result["revision_count"] <= MAX_REVISIONS
    assert result["final_status"] in ("approved", "escalated_to_human_review")
