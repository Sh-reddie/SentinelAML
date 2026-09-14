"""Wires the four agents into a LangGraph state machine.

    triage --dismiss--> finalize_dismiss --> END
      |
    escalate
      v
  investigator -> narrative -> critic --approved--> finalize_approved --> END
                       ^            |
                       |__ revise __|
                                    |
                          max revisions hit
                                    v
                          finalize_escalate --> END

The revise loop is capped (see agents/critic.py:MAX_REVISIONS) so a stubborn
disagreement between narrative and critic degrades to "kick it to a human"
instead of spinning forever -- a graph that can loop needs an explicit exit,
not an implicit one.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from .agents.critic import make_critic_node, route_after_critic
from .agents.investigator import make_investigator_node
from .agents.narrative import make_narrative_node
from .agents.triage import make_triage_node, route_after_triage
from .llm import LLMClient, get_llm_client
from .schemas import CaseState


def _finalize(status: str):
    def _node(state: CaseState) -> dict:
        return {"final_status": status}

    return _node


def build_graph(llm: LLMClient | None = None):
    llm = llm or get_llm_client()

    graph = StateGraph(CaseState)
    graph.add_node("triage", make_triage_node(llm))
    graph.add_node("investigator", make_investigator_node(llm))
    graph.add_node("narrative", make_narrative_node(llm))
    graph.add_node("critic", make_critic_node(llm))
    graph.add_node("finalize_dismiss", _finalize("dismissed"))
    graph.add_node("finalize_approved", _finalize("approved"))
    graph.add_node("finalize_escalate", _finalize("escalated_to_human_review"))

    graph.set_entry_point("triage")
    graph.add_conditional_edges(
        "triage", route_after_triage, {"investigate": "investigator", "dismiss": "finalize_dismiss"}
    )
    graph.add_edge("investigator", "narrative")
    graph.add_edge("narrative", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"approved": "finalize_approved", "revise": "narrative", "escalate_to_human": "finalize_escalate"},
    )
    graph.add_edge("finalize_dismiss", END)
    graph.add_edge("finalize_approved", END)
    graph.add_edge("finalize_escalate", END)

    return graph.compile()
