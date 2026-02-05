"""
LangChain agent setup for BiteBot restaurant assistant.

Memory architecture
-------------------
* MemorySaver (checkpointer) owns the full per-session conversation.
  We only hand agent.invoke() the NEW message; the checkpointer supplies
  the rest via thread_id.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from src.tools import all_tools, set_tool_context, get_tool_context, set_active_session

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Checkpointer – one instance, shared across the whole app.
# Each Flask session gets a unique thread_id so conversations are isolated.
# NOTE: MemorySaver is in-process only.  App restart clears all threads.
# For production persistence swap in SqliteSaver or a Redis checkpointer.
_checkpointer = MemorySaver()

# Agent factory
def create_discovery_and_reservation_agent():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key not found.")

    model = ChatOpenAI(model="gpt-4o", temperature=0.7)

    system_prompt = """You are BiteBot, a friendly and conversational restaurant assistant.

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
        WRONG:   date="2026-02-03"  — never calculate dates yourself, the tool handles it

        CRITICAL RULE #2: RESERVATION WORKFLOW

        1. Call check_availability_tool first (pass date exactly as user said)
        2. Present availability to user
        3. Ask: "Would you like me to book this table?"
        4. Wait for confirmation ("yes", "sure", "book it", etc.)
        5. Ask: "What name should I put the reservation under?"
        6. Wait for their real name
        7. ONLY THEN call make_reservation_tool (just pass name + customer details, date/time are automatic)

        NEVER use placeholder names like "Guest" or "User" — the tool will reject them.

        Be conversational and helpful!"""

    logger.info(f"Creating agent with {len(all_tools)} tools")
    for t in all_tools:
        logger.info(f"  - {t.name}")

    agent = create_agent(
        model,
        all_tools,
        system_prompt=system_prompt,
        checkpointer=_checkpointer,
    )
    return agent

# Run
def run_agent(agent, user_message: str, thread_id: str, tool_context: dict = None) -> dict:
    """Invoke the agent with a single new message.

    The checkpointer owns conversation history. We do NOT replay it.

    Args:
        agent:        The compiled agent graph.
        user_message: What the user typed this turn.
        thread_id:    One per Flask session; scopes the agent's memory.
        tool_context: Inter-tool state (e.g. availability) persisted in
                      Flask session separately from the agent's memory.

    Returns:
        output           -- the assistant's reply (clean text).
        tool_context     -- updated inter-tool state to persist in session.
        reservation_json -- raw JSON dict if a reservation was confirmed, else None.
    """
    try:
        # Restore tool context from previous request
        if tool_context:
            for key, value in tool_context.items():
                if value is not None:
                    set_tool_context(key, value)

        logger.info(f"Processing message for thread: {thread_id}")

        # Bind this execution context to the session's thread_id. ContextVar propagates into the worker threads that langgraph uses to run tools, so they'll land in the right bucket.
        set_active_session(thread_id)

        config = {"configurable": {"thread_id": thread_id}}
        response = agent.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )
        
        # Get all messages from response
        messages = response.get("messages", [])
        
        if not messages:
            logger.warning("No messages returned from agent")
            return {
                "output": "I apologize, but I encountered an issue processing your request.",
                "tool_context": {"availability": None},
                "reservation_json": None,
            }

        # The last message is always the agent's final response
        output = messages[-1].content if messages[-1].content else "I'm here to help! What would you like to know?"

        logger.info(f"Response generated: {output[:200]}...")

        # Get updated tool context (includes any reservation data)
        updated_tool_context = {
            "availability": get_tool_context("availability"),
        }
        
        # Get reservation data if it exists (stored by make_reservation_tool)
        reservation_json = get_tool_context("reservation")

        return {
            "output": output,
            "tool_context": updated_tool_context,
            "reservation_json": reservation_json,
        }
        
    except Exception as e:
        logger.error(f"Error in run_agent: {str(e)}", exc_info=True)
        return {
            "output": "I apologize, but I encountered an error. Please try again.",
            "tool_context": {"availability": None},
            "reservation_json": None,
        }