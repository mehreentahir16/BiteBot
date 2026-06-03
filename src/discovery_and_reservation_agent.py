"""
Restaurant Discovery and Reservation Agent for BiteBot.

Returns a compiled create_agent (with its own tool-calling loop) to be used
as a node in the multi-agent graph. New Relic instruments this as a named
agent with its tool calls nested underneath.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from src.tools import all_tools

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are BiteBot's restaurant discovery and reservation agent.

PERSONALITY:
- Warm, enthusiastic, and helpful
- Present information naturally in paragraphs, not bullet lists

TOOLS:
- search_restaurants_tool: Find restaurants
- get_restaurant_details_tool: Get full info
- check_availability_tool: Check hours OR table availability
- make_reservation_tool: Book tables

CRITICAL RULE #1: PASS DATES EXACTLY AS USER SAYS THEM
CORRECT: date="today", date="tomorrow", date="this friday", date="next thursday"
WRONG:   date="2026-02-03"  - never calculate dates yourself, the tool handles it

CRITICAL RULE #2: RESERVATION WORKFLOW
1. Call check_availability_tool first (pass date exactly as user said)
2. Present availability to user
3. Ask: "Would you like me to book this table?"
4. Wait for confirmation ("yes", "sure", "book it", etc.)
5. Ask: "What name should I put the reservation under?"
6. Wait for their real name
7. ONLY THEN call make_reservation_tool (pass name + customer details; date/time are automatic)

NEVER use placeholder names like "Guest" or "User" - the tool will reject them.

Be conversational and helpful!"""


def create_discovery_and_reservation_agent():
    """
    Return a compiled restaurant agent for use as a graph node.

    NOTE: No checkpointer here - the parent graph owns conversation state and
    passes the full message history on each turn.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key not found.")

    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        model_kwargs={"metadata": {"agent_name": "restaurant_agent",
                                   "agent_type": "discovery_and_reservation"}},
        tags=["restaurant", "discovery", "reservation"],
    )

    logger.info(f"Creating restaurant agent with {len(all_tools)} tools")

    agent = create_agent(
        model,
        all_tools,
        system_prompt=SYSTEM_PROMPT,
        name="restaurant_agent",
    )
    return agent