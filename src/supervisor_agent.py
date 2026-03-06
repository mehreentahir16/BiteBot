"""
Supervisor for BiteBot multi-agent system.

Routes user requests to either:
- Restaurant Agent (search, book, reviews)
- Support Agent (modify, cancel reservations)
"""

import logging
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    """Structured output for routing decision."""

    agent: Literal["restaurant", "support"] = Field(
        description="Which agent should handle this request"
    )
    reasoning: str = Field(description="Brief explanation of routing decision")


def create_supervisor():
    """Create the supervisor LLM for routing."""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Use structured output for reliable routing
    return model.with_structured_output(RouteDecision)


def route_request(supervisor, user_message: str, conversation_history: list) -> str:
    """
    Route a user message to the appropriate agent.

    Args:
        supervisor: The supervisor LLM
        user_message: Current user message
        conversation_history: Recent conversation for context

    Returns:
        "restaurant" or "support"
    """
    # Build routing prompt
    routing_prompt = f"""You are a routing supervisor for BiteBot, a restaurant assistant.

        Route user requests to the appropriate agent:

        **RESTAURANT AGENT** - Handles:
        - Searching for restaurants
        - Getting restaurant details, reviews, menu info
        - Checking availability
        - Making a NEW reservations
        - Questions about restaurants ("is it good?", "what do people say?")

        **SUPPORT AGENT** - Handles:
        - Viewing existing reservations
        - Modifying reservations (change date/time/party size)
        - Cancelling reservations
        - Questions about existing bookings

        SUPPORT AGENT must ONLY be used when a reservation already exists in the system.
        Recent conversation context:
        {format_history(conversation_history[-4:]) if conversation_history else "No prior context"}

        Current user message: "{user_message}"

        Which agent should handle this request?"""

    try:
        response = supervisor.invoke([HumanMessage(content=routing_prompt)])

        logger.info(f"Routing to '{response.agent}' agent - {response.reasoning}")
        return response.agent

    except Exception as e:
        logger.error(f"Routing error: {e}", exc_info=True)
        # Default to restaurant agent on error
        return "restaurant"


def format_history(messages: list) -> str:
    """Format conversation history for routing context."""
    if not messages:
        return "None"

    formatted = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:100]  # Truncate for brevity
        formatted.append(f"{role}: {content}")

    return "\n".join(formatted)