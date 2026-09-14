"""Generates a synthetic, labeled case set for SentinelAML.

No external dataset dependency (no Kaggle download, no real bank data, no
privacy/compliance question to answer) -- everything here is fabricated with
a fixed seed so the eval harness is reproducible. Each "case" is a subject
transaction plus its related transactions, labeled `benign` or `suspicious`
by construction: suspicious cases are deliberately built to trip one of the
four rules in src/rules.py, benign cases are built to trip none of them.

Run directly to (re)write data/cases.json, data/accounts.csv and
data/transactions.csv:

    python -m data.generate_synthetic_transactions
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from faker import Faker

SEED = 20260908
CHANNELS = ["wire", "ach", "card", "cash", "mobile"]
COUNTRIES = ["US", "US", "US", "GB", "AE", "SG", "IN"]

DATA_DIR = Path(__file__).parent
fake = Faker()


def _account(rng: random.Random, account_id: str, *, age_days: int | None = None, risk: str | None = None) -> Dict[str, Any]:
    return {
        "account_id": account_id,
        "customer_name": fake.name(),
        "risk_rating": risk or rng.choice(["low", "low", "medium", "high"]),
        "kyc_country": rng.choice(COUNTRIES),
        "account_age_days": age_days if age_days is not None else rng.randint(60, 3000),
    }


def _txn(rng: random.Random, txn_id: str, account_id: str, counterparty_id: str, amount: float, ts: datetime, **kw) -> Dict[str, Any]:
    return {
        "txn_id": txn_id,
        "account_id": account_id,
        "counterparty_id": counterparty_id,
        "amount": round(amount, 2),
        "currency": "USD",
        "timestamp": ts.isoformat(),
        "channel": kw.get("channel") or rng.choice(CHANNELS),
        "country": kw.get("country") or rng.choice(COUNTRIES),
    }


def _benign_case(rng: random.Random, idx: int) -> Dict[str, Any]:
    base_ts = datetime(2026, 8, 1) + timedelta(days=idx, hours=rng.randint(0, 23))
    acct = _account(rng, f"ACC-B{idx:03d}")
    subject = _txn(rng, f"TXN-B{idx:03d}-0", acct["account_id"], f"CP-{fake.uuid4()[:8]}", rng.uniform(20, 4000), base_ts)
    related = [
        _txn(
            rng,
            f"TXN-B{idx:03d}-{j}",
            acct["account_id"],
            f"CP-{fake.uuid4()[:8]}",
            rng.uniform(20, 3000),
            base_ts - timedelta(days=rng.randint(1, 10)),
        )
        for j in range(1, rng.randint(1, 3))
    ]
    return {
        "case_id": f"benign-{idx:03d}",
        "label": "benign",
        "account": acct,
        "transaction": subject,
        "related_transactions": related,
    }


def _structuring_case(rng: random.Random, idx: int) -> Dict[str, Any]:
    acct = _account(rng, f"ACC-S{idx:03d}")
    base_ts = datetime(2026, 8, 5) + timedelta(days=idx)
    subject = _txn(rng, f"TXN-S{idx:03d}-0", acct["account_id"], f"CP-{fake.uuid4()[:8]}", rng.uniform(9100, 9900), base_ts, channel="cash")
    related = [
        _txn(
            rng, f"TXN-S{idx:03d}-{j}", acct["account_id"], f"CP-{fake.uuid4()[:8]}",
            rng.uniform(9100, 9900), base_ts - timedelta(hours=j * 4), channel="cash",
        )
        for j in range(1, 3)
    ]
    return {"case_id": f"structuring-{idx:03d}", "label": "suspicious", "account": acct, "transaction": subject, "related_transactions": related}


def _fan_out_case(rng: random.Random, idx: int) -> Dict[str, Any]:
    acct = _account(rng, f"ACC-F{idx:03d}")
    base_ts = datetime(2026, 8, 10) + timedelta(days=idx)
    subject = _txn(rng, f"TXN-F{idx:03d}-0", acct["account_id"], f"CP-{fake.uuid4()[:8]}", rng.uniform(500, 2000), base_ts, channel="mobile")
    related = [
        _txn(
            rng, f"TXN-F{idx:03d}-{j}", acct["account_id"], f"CP-{fake.uuid4()[:8]}",
            rng.uniform(500, 2000), base_ts + timedelta(minutes=j * 5), channel="mobile",
        )
        for j in range(1, 5)
    ]
    return {"case_id": f"fanout-{idx:03d}", "label": "suspicious", "account": acct, "transaction": subject, "related_transactions": related}


def _round_trip_case(rng: random.Random, idx: int) -> Dict[str, Any]:
    acct = _account(rng, f"ACC-R{idx:03d}")
    counterparty = f"ACC-RCP{idx:03d}"
    base_ts = datetime(2026, 8, 15) + timedelta(days=idx)
    amount = rng.uniform(4000, 15000)
    subject = _txn(rng, f"TXN-R{idx:03d}-0", acct["account_id"], counterparty, amount, base_ts, channel="wire")
    related = [
        _txn(rng, f"TXN-R{idx:03d}-1", counterparty, acct["account_id"], amount * rng.uniform(0.97, 1.0), base_ts + timedelta(days=1), channel="wire"),
    ]
    return {"case_id": f"roundtrip-{idx:03d}", "label": "suspicious", "account": acct, "transaction": subject, "related_transactions": related}


def _new_account_case(rng: random.Random, idx: int) -> Dict[str, Any]:
    acct = _account(rng, f"ACC-N{idx:03d}", age_days=rng.randint(1, 25))
    base_ts = datetime(2026, 8, 20) + timedelta(days=idx)
    subject = _txn(rng, f"TXN-N{idx:03d}-0", acct["account_id"], f"CP-{fake.uuid4()[:8]}", rng.uniform(5100, 20000), base_ts, channel="wire")
    return {"case_id": f"newacct-{idx:03d}", "label": "suspicious", "account": acct, "transaction": subject, "related_transactions": []}


def generate_cases(n_each: int = 8) -> List[Dict[str, Any]]:
    rng = random.Random(SEED)
    Faker.seed(SEED)
    cases: List[Dict[str, Any]] = []
    for i in range(n_each):
        cases.append(_benign_case(rng, i))
    for i in range(n_each // 2):
        cases.append(_structuring_case(rng, i))
        cases.append(_fan_out_case(rng, i))
        cases.append(_round_trip_case(rng, i))
        cases.append(_new_account_case(rng, i))
    rng.shuffle(cases)
    return cases


def write_all(n_each: int = 8) -> Path:
    cases = generate_cases(n_each)
    out_path = DATA_DIR / "cases.json"
    out_path.write_text(json.dumps(cases, indent=2))

    accounts_seen: Dict[str, Dict[str, Any]] = {}
    txn_rows: List[Dict[str, Any]] = []
    for case in cases:
        accounts_seen[case["account"]["account_id"]] = case["account"]
        txn_rows.append(case["transaction"])
        txn_rows.extend(case["related_transactions"])

    import csv

    with (DATA_DIR / "accounts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(accounts_seen.values())).keys()))
        writer.writeheader()
        writer.writerows(accounts_seen.values())

    with (DATA_DIR / "transactions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(txn_rows[0].keys()))
        writer.writeheader()
        writer.writerows(txn_rows)

    return out_path


if __name__ == "__main__":
    path = write_all()
    print(f"Wrote {path}, data/accounts.csv, data/transactions.csv")
