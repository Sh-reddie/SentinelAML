from src.rules import baseline_decision, triggered_rules
from src.schemas import Account, Transaction


def _account(**kw):
    defaults = dict(account_id="ACC-1", customer_name="Test User", risk_rating="low", kyc_country="US", account_age_days=1000)
    defaults.update(kw)
    return Account(**defaults)


def _txn(**kw):
    defaults = dict(
        txn_id="TXN-1", account_id="ACC-1", counterparty_id="CP-1", amount=100.0,
        currency="USD", timestamp="2026-08-01T10:00:00", channel="ach", country="US",
    )
    defaults.update(kw)
    return Transaction(**defaults)


def test_benign_transaction_triggers_nothing():
    txn = _txn(amount=250.0)
    account = _account()
    assert triggered_rules(txn, account, []) == []
    assert baseline_decision(txn, account, []) == "dismiss"


def test_structuring_rule_fires_just_under_threshold():
    txn = _txn(amount=9500.0)
    account = _account()
    assert "structuring_near_threshold" in triggered_rules(txn, account, [])


def test_structuring_rule_does_not_fire_well_under_threshold():
    txn = _txn(amount=500.0)
    account = _account()
    assert "structuring_near_threshold" not in triggered_rules(txn, account, [])


def test_rapid_fan_out_fires_with_five_counterparties_in_an_hour():
    txn = _txn(txn_id="TXN-0", counterparty_id="CP-0", timestamp="2026-08-01T10:00:00")
    related = [
        _txn(txn_id=f"TXN-{i}", counterparty_id=f"CP-{i}", timestamp=f"2026-08-01T10:{i:02d}:00")
        for i in range(1, 5)
    ]
    account = _account()
    assert "rapid_fan_out" in triggered_rules(txn, account, related)


def test_new_account_large_transaction_rule():
    txn = _txn(amount=6000.0)
    account = _account(account_age_days=5)
    assert "new_account_large_transaction" in triggered_rules(txn, account, [])


def test_round_trip_layering_rule():
    txn = _txn(account_id="ACC-1", counterparty_id="ACC-2", amount=5000.0, timestamp="2026-08-01T10:00:00")
    reverse = _txn(txn_id="TXN-2", account_id="ACC-2", counterparty_id="ACC-1", amount=4980.0, timestamp="2026-08-02T10:00:00")
    account = _account()
    assert "round_trip_layering" in triggered_rules(txn, account, [reverse])
