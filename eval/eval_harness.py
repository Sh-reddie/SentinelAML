"""Evaluates the full pipeline against the synthetic labeled case set.

Reports, and writes to eval/eval_report.json:
  - Triage precision/recall/F1 vs. the rules-only baseline (src/rules.baseline_decision)
    -- this is the number that answers "did adding an LLM on top of the rules help?"
  - Mean citation accuracy across every SAR draft that was produced
  - The critic-loop outcome mix: approved / escalated_to_human_review, and mean
    revisions needed before approval

Run with:

    python -m eval.eval_harness

Uses LLM_PROVIDER=mock by default (see src/llm.py) so this runs for free and
deterministically; set LLM_PROVIDER=anthropic or openai with a real key to
eval against the live model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from data.generate_synthetic_transactions import generate_cases
from src.graph import build_graph
from src.rules import baseline_decision
from src.schemas import Account, Transaction

EVAL_DIR = Path(__file__).parent


def _prf1(tp: int, fp: int, fn: int, tn: int = 0) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def run_eval(n_each: int = 8) -> Dict[str, Any]:
    cases = generate_cases(n_each)
    graph = build_graph()

    baseline_confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    pipeline_confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    citation_accuracies: List[float] = []
    outcomes: Dict[str, int] = {"dismissed": 0, "approved": 0, "escalated_to_human_review": 0}
    revisions: List[int] = []

    for case in cases:
        txn = Transaction(**case["transaction"])
        account = Account(**case["account"])
        related = [Transaction(**r) for r in case["related_transactions"]]
        should_escalate = case["label"] == "suspicious"

        baseline_escalates = baseline_decision(txn, account, related) == "escalate"
        _accumulate(baseline_confusion, predicted=baseline_escalates, actual=should_escalate)

        final_state = graph.invoke(
            {"transaction": txn, "account": account, "related_transactions": related, "revision_count": 0}
        )
        pipeline_escalates = final_state["triage"].decision == "escalate"
        _accumulate(pipeline_confusion, predicted=pipeline_escalates, actual=should_escalate)

        outcomes[final_state["final_status"]] = outcomes.get(final_state["final_status"], 0) + 1
        if final_state.get("critic_feedback"):
            citation_accuracies.append(final_state["critic_feedback"].citation_accuracy)
        if final_state["final_status"] in ("approved", "escalated_to_human_review"):
            revisions.append(final_state.get("revision_count", 0))

    report = {
        "n_cases": len(cases),
        "baseline_rules_only": _prf1(**baseline_confusion),
        "pipeline_triage": _prf1(**pipeline_confusion),
        "mean_citation_accuracy": round(sum(citation_accuracies) / len(citation_accuracies), 3) if citation_accuracies else None,
        "outcomes": outcomes,
        "mean_revisions_to_resolution": round(sum(revisions) / len(revisions), 2) if revisions else None,
        "llm_provider": os.getenv("LLM_PROVIDER", "mock"),
    }

    (EVAL_DIR / "eval_report.json").write_text(json.dumps(report, indent=2))
    _maybe_log_mlflow(report)
    return report


def _accumulate(confusion: Dict[str, int], *, predicted: bool, actual: bool) -> None:
    if predicted and actual:
        confusion["tp"] += 1
    elif predicted and not actual:
        confusion["fp"] += 1
    elif not predicted and actual:
        confusion["fn"] += 1
    else:
        confusion["tn"] += 1


def _maybe_log_mlflow(report: Dict[str, Any]) -> None:
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed -- skipping run tracking (metrics were still written to eval_report.json)")
        return

    with mlflow.start_run(run_name="sentinelaml-eval"):
        mlflow.log_param("llm_provider", report["llm_provider"])
        mlflow.log_param("n_cases", report["n_cases"])
        mlflow.log_metrics(
            {
                "baseline_precision": report["baseline_rules_only"]["precision"],
                "baseline_recall": report["baseline_rules_only"]["recall"],
                "baseline_f1": report["baseline_rules_only"]["f1"],
                "pipeline_precision": report["pipeline_triage"]["precision"],
                "pipeline_recall": report["pipeline_triage"]["recall"],
                "pipeline_f1": report["pipeline_triage"]["f1"],
                "mean_citation_accuracy": report["mean_citation_accuracy"] or 0.0,
            }
        )


if __name__ == "__main__":
    result = run_eval()
    print(json.dumps(result, indent=2))
