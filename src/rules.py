"""Deterministic rule checks that seed the triage agent.

These are intentionally simple and auditable -- in a real AML program the rule
set is the part compliance signs off on, and the LLM's job is to reason over
*why* rules fired and write it up, not to invent new rules on the fly.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List

from .schemas import Account, Transaction

STRUCTURING_THRESHOLD = 10_000.0
STRUCTURING_MARGIN = 0.9  # flag txns >= 90% of the threshold, just under it
FAN_OUT_MIN_COUNTERPARTIES = 5
FAN_OUT_WINDOW_HOURS = 1
NEW_ACCOUNT_DAYS = 30
NEW_ACCOUNT_LARGE_AMOUNT = 5_000.0


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


@dataclass
class Rule:
    name: str
    check: Callable[[Transaction, Account, List[Transaction]], bool]


def _structuring(txn: Transaction, account: Account, related: List[Transaction]) -> bool:
    return STRUCTURING_THRESHOLD * STRUCTURING_MARGIN <= txn.amount < STRUCTURING_THRESHOLD


def _rapid_fan_out(txn: Transaction, account: Account, related: List[Transaction]) -> bool:
    window_start = _parse(txn.timestamp)
    recent = [
        r for r in related
        if r.account_id == txn.account_id
        and abs((_parse(r.timestamp) - window_start).total_seconds()) <= FAN_OUT_WINDOW_HOURS * 3600
    ]
    counterparties = {r.counterparty_id for r in recent} | {txn.counterparty_id}
    return len(counterparties) >= FAN_OUT_MIN_COUNTERPARTIES


def _new_account_large_txn(txn: Transaction, account: Account, related: List[Transaction]) -> bool:
    return account.account_age_days <= NEW_ACCOUNT_DAYS and txn.amount >= NEW_ACCOUNT_LARGE_AMOUNT


def _round_trip_layering(txn: Transaction, account: Account, related: List[Transaction]) -> bool:
    # Funds sent to a counterparty who, within the related set, sends a similar
    # amount back to the original account -- a classic layering pattern.
    for r in related:
        if (
            r.account_id == txn.counterparty_id
            and r.counterparty_id == txn.account_id
            and abs(r.amount - txn.amount) <= max(50.0, txn.amount * 0.05)
        ):
            return True
    return False


RULES: List[Rule] = [
    Rule("structuring_near_threshold", _structuring),
    Rule("rapid_fan_out", _rapid_fan_out),
    Rule("new_account_large_transaction", _new_account_large_txn),
    Rule("round_trip_layering", _round_trip_layering),
]


def triggered_rules(txn: Transaction, account: Account, related: List[Transaction]) -> List[str]:
    return [rule.name for rule in RULES if rule.check(txn, account, related)]


def baseline_decision(txn: Transaction, account: Account, related: List[Transaction]) -> str:
    """A rules-only baseline (no LLM) that the eval harness compares the full
    pipeline against, so the README can report an actual lift number instead
    of a vibe."""
    return "escalate" if triggered_rules(txn, account, related) else "dismiss"
