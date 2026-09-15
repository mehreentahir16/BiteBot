"""
Customer Support Agent for BiteBot.

Handles reservation modifications, cancellations, and inquiries.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from langgraph.checkpoint.memory import MemorySaver

from src.tools import support_tools, set_active_session, set_support_context, get_support_context

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared checkpointer for support agent memory
_support_checkpointer = MemorySaver()


def create_support_agent():
    """Create the customer support agent."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key not found.")

    model = ChatOpenAI(model="gpt-5-mini", temperature=0.3)  # Lower temp for support

    system_prompt = """You are BiteBot's Customer Support Assistant.

        Your role is to help customers with their existing reservations:
        - View reservation details
        - Modify reservation date/time/party size
        - Cancel reservations
        - Answer questions about their bookings

        PERSONALITY:
        - Professional, empathetic, and solution-oriented
        - Clear and concise communication
        - Always confirm changes before executing them

        TOOLS:
        - view_reservation_tool: Look up reservations (no confirmation number needed)
        - modify_reservation_tool: Change date, time, or party size (no confirmation number needed)
        - cancel_reservation_tool: Cancel a reservation (no confirmation number needed)

        SMART RESERVATION HANDLING:
        All tools work WITHOUT a confirmation number! They automatically:
        - If customer has 1 reservation → use it directly
        - If customer has multiple → list them and ask which one
        - If customer has 0 → inform them politely

        WORKFLOW:
        1. When customer says "my reservation" or "cancel my reservation":
        - Just call the tool without a confirmation_number
        - The tool will handle everything automatically
        
        2. If customer provides a confirmation number:
        - Pass it to the tool for direct lookup
        
        3. For modifications:
        - Always verify what they want to change
        - Confirm the change with them before calling modify_reservation_tool
        - Example: "Just to confirm, you want to change your party size from 2 to 4. Is that correct?"

        4. For cancellations:
        - Always confirm before canceling
        - Example: "Are you sure you want to cancel your reservation at [Restaurant] on [Date]?"

        Be helpful and understanding!"""

    logger.info(f"Creating support agent with {len(support_tools)} tools")
    for t in support_tools:
        logger.info(f"  - {t.name}")

    agent = create_agent(
        model,
        support_tools,
        system_prompt=system_prompt,
        checkpointer=_support_checkpointer,
        name="customer_support_agent",
    )
    return agent


def run_support_agent(
    agent, user_message: str, thread_id: str, reservations: list
) -> dict:
    """
    Run the support agent with access to session reservations.

    Args:
        agent: The support agent
        user_message: User's message
        thread_id: Session thread ID
        reservations: List of reservations from Flask session

    Returns:
        Dict with output and updated reservations list
    """
    try:
        
        set_active_session(thread_id)
        set_support_context(reservations)

        logger.info(f"Processing support request for thread: {thread_id}")
        logger.info(f"Session has {len(reservations)} reservations")

        config = {"configurable": {"thread_id": thread_id}}
        response = agent.invoke(
            {"messages": [HumanMessage(content=user_message)]}, config=config
        )
        
        messages = response.get("messages", [])
        
        if not messages:
            logger.warning("No messages returned from support agent")
            return {
                "output": "I apologize, but I encountered an issue. Please try again.",
                "reservations": reservations,
            }

        # Get final output from last message
        output = messages[-1].content if messages[-1].content else "How can I help with your reservation?"

        logger.info(f"Support response generated: {output[:200]}...")

        # Get updated reservations from tool context (tools may have modified them)
        updated_reservations = get_support_context()

        return {
            "output": output,
            "reservations": updated_reservations,
        }
        
    except Exception as e:
        logger.error(f"Error in run_support_agent: {str(e)}", exc_info=True)
        return {
            "output": "I apologize, but I encountered an error. Please try again.",
            "reservations": reservations,
        }