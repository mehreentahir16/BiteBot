"""
Supervisor Agent for BiteBot  (agents-as-tools pattern).

The supervisor is a create_agent. Its tools are thin wrappers that invoke the
specialist agents via their public .invoke(). Because each specialist is reached
through .invoke(), New Relic's agent instrumentation fires for it, and the calls
nest as: supervisor -> delegate tool -> specialist agent -> specialist's tools.

This is the pattern that produces agent + agent-to-agent + agent-to-tool entities
in the New Relic map. (Nesting compiled agents as StateGraph nodes does NOT, because
LangGraph inlines the subgraph and never calls the sub-agent's .invoke().)
"""

import logging
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import InjectedState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are the routing supervisor for BiteBot.

You never answer the user yourself. You pick the right specialist and call exactly
ONE delegate tool:

- delegate_to_restaurant: searching for restaurants, details/reviews/menus,
  checking availability, making a NEW reservation, and follow-ups about a restaurant
  just discussed ("book the first one", "reviews about that place").

- delegate_to_support: an EXISTING reservation the user already has
  ("my reservation", "my booking", "change my...", "cancel my...").

After the specialist responds, relay their reply to the user VERBATIM - do not
summarize, shorten, or rewrite it. When unsure, prefer delegate_to_restaurant."""


def _conversation_only(messages):
    """
    Keep just the real conversation (human turns + substantive assistant turns).
    Drops the supervisor's own tool-call/tool messages so the specialist sees a
    clean history.
    """
    cleaned = []
    for m in messages:
        if isinstance(m, HumanMessage):
            cleaned.append(m)
        elif isinstance(m, AIMessage) and m.content and not m.tool_calls:
            cleaned.append(m)
    return cleaned


def create_supervisor(restaurant_agent, support_agent):
    """
    Build the supervisor agent. Takes the two specialist agents so it can call
    them as tools.
    """

    @tool("delegate_to_restaurant",
          description=("Delegate to the restaurant discovery & reservation specialist for "
                       "searching, details, availability, and NEW reservations."))
    def delegate_to_restaurant(state: Annotated[dict, InjectedState]) -> str:
        convo = _conversation_only(state["messages"])
        logger.info("[SUPERVISOR] -> restaurant_agent")
        result = restaurant_agent.invoke({"messages": convo})
        return result["messages"][-1].content

    @tool("delegate_to_support",
          description=("Delegate to the customer support specialist for an EXISTING "
                       "reservation (view, modify, or cancel)."))
    def delegate_to_support(state: Annotated[dict, InjectedState]) -> str:
        convo = _conversation_only(state["messages"])
        logger.info("[SUPERVISOR] -> support_agent")
        result = support_agent.invoke({"messages": convo})
        return result["messages"][-1].content

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        model_kwargs={"metadata": {"agent_name": "supervisor", "agent_type": "routing"}},
        tags=["supervisor", "routing"],
    )

    logger.info("Creating supervisor agent (agents-as-tools)")

    return create_agent(
        model,
        [delegate_to_restaurant, delegate_to_support],
        system_prompt=SYSTEM_PROMPT,
        name="supervisor",
    )