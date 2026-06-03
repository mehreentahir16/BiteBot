"""
Customer Support Agent for BiteBot.

Returns a compiled create_agent (with its own tool-calling loop) to be used
as a node in the multi-agent graph.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from src.tools import support_tools

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are BiteBot's Customer Support Assistant.

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
- If customer has 1 reservation -> use it directly
- If customer has multiple -> list them and ask which one
- If customer has 0 -> inform them politely

WORKFLOW:
1. "my reservation" / "cancel my reservation" -> call the tool without a confirmation_number
2. If a confirmation number is provided -> pass it for direct lookup
3. Modifications: confirm the change before calling modify_reservation_tool
   e.g. "Just to confirm, change party size from 2 to 4?"
4. Cancellations: confirm before canceling
   e.g. "Are you sure you want to cancel your reservation at [Restaurant] on [Date]?"

Be helpful and understanding!"""


def create_support_agent():
    """
    Return a compiled support agent for use as a graph node.

    NOTE: No checkpointer here - the parent graph owns conversation state.
    Reservation data is made available to tools via the session context
    (set in run_multi_agent_system before the graph is invoked).
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key not found.")

    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        model_kwargs={"metadata": {"agent_name": "support_agent",
                                   "agent_type": "customer_support"}},
        tags=["support", "customer_service"],
    )

    logger.info(f"Creating support agent with {len(support_tools)} tools")

    agent = create_agent(
        model,
        support_tools,
        system_prompt=SYSTEM_PROMPT,
        name="support_agent",
    )
    return agent