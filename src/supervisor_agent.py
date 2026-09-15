"""
Supervisor for BiteBot multi-agent system.

Routes user requests to either:
- Restaurant Agent (search, book, reviews)
- Support Agent (modify, cancel reservations)
"""

import logging
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    """Structured output for routing decision."""

    agent: Literal["restaurant", "support"] = Field(
        description="Which agent should handle this request"
    )
    reasoning: str = Field(description="Brief explanation of routing decision")


_ROUTING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a routing supervisor for BiteBot, a restaurant assistant.

Route user requests to the appropriate agent:

**RESTAURANT AGENT** - Handles:
- Searching for restaurants
- Getting restaurant details, reviews, menu info
- Checking availability
- Making NEW reservations
- Questions about restaurants ("is it good?", "what do people say?")

**SUPPORT AGENT** - Handles:
- Viewing existing reservations
- Modifying reservations (change date/time/party size)
- Cancelling reservations
- Questions about existing bookings

SUPPORT AGENT must ONLY be used when a reservation already exists in the system."""),
    ("human", "{input}"),
])


def create_supervisor():
    """Create the supervisor chain for routing."""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # Chain input is a plain string keyed as "input" — NR captures just that
    # string rather than a serialised list of message objects.
    return _ROUTING_PROMPT | model.with_structured_output(RouteDecision)


def route_request(supervisor, user_message: str, conversation_history: list) -> str:
    """
    Route a user message to the appropriate agent.

    Args:
        supervisor: The supervisor chain
        user_message: Current user message
        conversation_history: Unused — agents own their memory via MemorySaver

    Returns:
        "restaurant" or "support"
    """
    try:
        response = supervisor.invoke({"input": user_message})
        logger.info(f"Routing to '{response.agent}' agent - {response.reasoning}")
        return response.agent

    except Exception as e:
        logger.error(f"Routing error: {e}", exc_info=True)
        return "restaurant"