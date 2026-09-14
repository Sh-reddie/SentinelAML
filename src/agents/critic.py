"""Critic agent: the loop that keeps the narrative writer honest.

Citation accuracy is checked in code, not left to the LLM's self-report --
every txn_id the narrative cites is verified against the set of txn_ids
that actually exist in this case. That is the difference between a system
that catches hallucinated evidence and one that hopes it doesn't happen.
The LLM call layered on top only judges qualitative completeness
(missing elements, speculation stated as fact).
"""
from __future__ import annotations

from ..llm import LLMClient
from ..schemas import CaseState, CriticFeedback

MAX_REVISIONS = 3


def _citation_accuracy(cited_txn_ids: list[str], valid_txn_ids: set[str]) -> float:
    if not cited_txn_ids:
        return 0.0
    correct = sum(1 for tid in cited_txn_ids if tid in valid_txn_ids)
    return correct / len(cited_txn_ids)


def make_critic_node(llm: LLMClient):
    def critic_node(state: CaseState) -> dict:
        sar = state["sar_draft"]
        dossier = state["dossier"]
        valid_txn_ids = {item.txn_id for item in dossier.timeline}

        accuracy = _citation_accuracy(sar.cited_txn_ids, valid_txn_ids)

        payload = {
            "sar_draft": sar.model_dump(),
            "dossier": dossier.model_dump(),
            "valid_txn_ids": sorted(valid_txn_ids),
        }
        feedback = llm.structured(kind="critic", payload=payload, schema=CriticFeedback)

        # Code-verified citation accuracy always wins over whatever the LLM
        # self-reported -- never trust the model's own hallucination check.
        feedback.citation_accuracy = accuracy
        if accuracy < 1.0:
            feedback.approved = False
            if "citation" not in " ".join(feedback.issues).lower():
                feedback.issues.append(
                    f"Citation accuracy {accuracy:.0%}: narrative cites a txn_id not present in this case."
                )

        return {
            "critic_feedback": feedback,
            "revision_count": state.get("revision_count", 0) + (0 if feedback.approved else 1),
        }

    return critic_node


def route_after_critic(state: CaseState) -> str:
    feedback = state["critic_feedback"]
    if feedback.approved:
        return "approved"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        return "escalate_to_human"
    return "revise"
