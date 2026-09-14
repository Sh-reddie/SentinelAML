"""Narrative-writer agent: drafts the SAR narrative from the dossier.

Re-entered on each revision loop -- `revision_notes` carries the critic's
issues from the previous pass so the rewrite actually targets them instead
of starting from scratch.
"""
from __future__ import annotations

from ..llm import LLMClient
from ..schemas import CaseState, SARDraft


def make_narrative_node(llm: LLMClient):
    def narrative_node(state: CaseState) -> dict:
        revision_count = state.get("revision_count", 0)
        prior_issues = state["critic_feedback"].issues if state.get("critic_feedback") else []
        payload = {
            "dossier": state["dossier"].model_dump(),
            "triage": state["triage"].model_dump(),
            "revision_notes": prior_issues,
        }
        draft = llm.structured(kind="narrative", payload=payload, schema=SARDraft)
        draft.version = revision_count + 1
        return {"sar_draft": draft}

    return narrative_node
