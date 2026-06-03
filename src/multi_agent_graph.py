"""
Multi-Agent orchestrator for BiteBot (agents-as-tools).

create_multi_agent_graph() builds the three agents and returns the supervisor
(the root agent). run_multi_agent_system() invokes the supervisor once; the
supervisor delegates to a specialist agent via a tool, which invokes that agent
through its public .invoke() so New Relic registers it as an agent entity.

Filename/function names kept stable so app.py imports don't change.
"""

import logging
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.supervisor_agent import create_supervisor
from src.discovery_and_reservation_agent import create_discovery_and_reservation_agent
from src.customer_support_agent import create_support_agent
from src.tools import set_active_session, set_support_context, get_tool_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DELEGATE_TO_AGENT = {
    "delegate_to_restaurant": "restaurant",
    "delegate_to_support":    "support",
}


def create_multi_agent_graph():
    """Build specialists + supervisor; return the supervisor as the root agent."""
    logger.info("Assembling multi-agent system (agents-as-tools)...")
    restaurant_agent = create_discovery_and_reservation_agent()
    support_agent    = create_support_agent()
    supervisor       = create_supervisor(restaurant_agent, support_agent)
    logger.info("✅ Multi-agent system ready")
    return supervisor


def _last_text_reply(messages):
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            return m.content
    return "I'm here to help! What would you like to know?"


def run_multi_agent_system(
    graph,                       # the supervisor agent
    user_message:         str,
    thread_id:            str,
    conversation_history: list = None,
    reservations:         list = None,
) -> dict:
    """Run one turn. Returns: output, agent_used, reservations, reservation_json."""
    # Make session data available to tools before anything runs.
    set_active_session(thread_id)
    set_support_context(reservations or [])

    # Rebuild recent conversation as clean role-tagged messages.
    messages = []
    if conversation_history:
        for msg in conversation_history[-8:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_message))

    try:
        result = graph.invoke(
            {"messages": messages},
            config={"metadata": {"user_message": user_message, "system": "multi_agent"}},
        )
        msgs = result.get("messages", [])

        # The specialist's real answer is the return value of the delegate tool
        # (a ToolMessage). Prefer that over the supervisor's relay so formatting
        # from the specialist is preserved exactly.
        output     = None
        agent_used = "restaurant"
        for m in reversed(msgs):
            if isinstance(m, ToolMessage) and m.name in _DELEGATE_TO_AGENT:
                output     = m.content
                agent_used = _DELEGATE_TO_AGENT[m.name]
                break

        if not output:
            output = _last_text_reply(msgs)

        reservation_json     = get_tool_context("reservation")
        updated_reservations = get_tool_context("reservations") or (reservations or [])

        logger.info(f"[MULTI-AGENT] {agent_used} replied: {output[:100]}...")

        return {
            "output":           output,
            "agent_used":       agent_used,
            "reservations":     updated_reservations,
            "reservation_json": reservation_json,
        }

    except Exception as e:
        logger.error(f"[MULTI-AGENT] Error: {e}", exc_info=True)
        return {
            "output":           "I apologize, but I encountered an error. Please try again.",
            "agent_used":       "restaurant",
            "reservations":     reservations or [],
            "reservation_json": None,
        }