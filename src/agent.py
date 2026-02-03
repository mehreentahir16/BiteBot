"""
LangChain agent setup for BiteBot restaurant assistant.

This module creates the agent that orchestrates the LLM, tools, and memory.
"""

import os
import logging
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.tools import all_tools, set_conversation_context, set_tool_context, get_tool_context

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

def create_bitebot_agent():
    """
    Create and return the BiteBot restaurant agent.
    
    Returns:
        Agent: The configured LangChain agent
    """
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError(
            "OpenAI API key not found. Please create a .env file with your OPENAI_API_KEY"
        )
    
    # Initialize the LLM
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.5,
    )
    
    # System prompt for the agent
    system_prompt = """You are BiteBot, a friendly and conversational restaurant assistant.

        PERSONALITY:
        - Warm, enthusiastic, and helpful
        - Present information naturally in paragraphs, not bullet lists

        TOOLS:
        - search_restaurants_tool: Find restaurants
        - get_restaurant_details_tool: Get full info
        - check_availability_tool: Check hours OR table availability
        - make_reservation_tool: Book tables 

        🚨 CRITICAL RULES FOR RESERVATIONS:

        1. When calling check_availability_tool, PASS DATES EXACTLY AS USER SAYS THEM:
            ✅ CORRECT:
            - User: "today at 7pm" → date="today", time="7pm"
            - User: "tomorrow evening" → date="tomorrow", time="evening"
            - User: "next Friday at 6:30" → date="next Friday", time="6:30"
            - User: "this Thursday" → date="Thursday"
            - User: "February 15th at 7pm" → date="February 15th", time="7pm"
            ❌ WRONG - Never calculate dates:
            - User: "tomorrow" → date="2026-02-03" ❌ NO! Use date="tomorrow"
            - User: "next Friday" → date="2026-02-07" ❌ NO! Use date="next Friday"
        The tool will parse ALL date formats automatically. Just pass what the user said!
        2. NEVER call make_reservation_tool immediately after check_availability_tool
        3. After checking availability, you MUST:
            a. Present the available table to the user
            b. Ask: "Would you like me to book this table for you?"
        c. WAIT for user to explicitly confirm (words like: "yes", "book it", "confirm", "go ahead", "sure", "please do")

        4. If user confirms, you MUST ask for their name:
        "Great! What name should I put the reservation under?"

        5. ONLY after you have BOTH confirmation AND real name, call make_reservation_tool

        6. NEVER use placeholder names like "Guest", "User", "Customer". If user hasn't given a name, you MUST ask for it.

        EXAMPLE - CORRECT FLOW:
        User: "Book a table for 4 today at 7pm"
        You: [Call check_availability_tool with date=today, time=7pm, party_size=4]
        You: "Great news! Vetri Cucina has a table available today at 7pm for 4 people. Would you like me to book it?"
        User: "Yes please"
        You: "Perfect! What name should I put the reservation under?"
        User: "Sarah Johnson"
        You: [NOW call make_reservation_tool with customer_name="Sarah Johnson"]
        Remember, a successful reservation requires BOTH confirmation + actual name!

        EXAMPLE - INCORRECT (DON'T DO THIS):
        User: "Book a table for 4 today at 7pm"
        You: [Call check_availability_tool with date=today, time=7pm, party_size=4]
        You: [Call make_reservation_tool immediately] ❌ WRONG - didn't ask for confirmation or name!

        EXAMPLE - INCORRECT (DON'T DO THIS):
        User: "Book a table for 4 today at 7pm"
        You: [Call check_availability_tool with date=today, time=7pm, party_size=4]
        You: "Great news! Vetri Cucina has a table available today at 7pm for 4 people. Would you like me to book it?"
        User: "Yes please"
        You: [Call make_reservation_tool immediately] ❌ WRONG - didn't ask for name! 

        Remember: Both confirmation and an actual Name are required for booking!"""
    
    logger.info(f"Creating agent with {len(all_tools)} tools")
    for tool in all_tools:
        logger.info(f"  - Tool: {tool.name}")
    
    # Create the agent with the new API
    agent = create_agent(
        model,
        all_tools,
        system_prompt=system_prompt
    )
    
    return agent

def _serialize_message(msg) -> dict:
    """Convert a LangChain message object to a plain dict for session storage."""
    msg_type = type(msg).__name__

    if msg_type == 'HumanMessage':
        return {'role': 'user', 'content': msg.content}

    if msg_type == 'AIMessage':
        return {'role': 'assistant', 'content': msg.content}

    if msg_type == 'ToolMessage':
        return {
            'role': 'tool',
            'content': msg.content,
            'name': msg.name,
            'tool_call_id': msg.tool_call_id,
        }

    # Fallback
    return {'role': 'unknown', 'content': str(msg)}


def run_agent(agent, user_message: str, conversation_history: list, tool_context: dict = None) -> dict:
    """
    Invoke the agent and return the output + new messages to persist.

    conversation_history: list of plain dicts (persisted in Flask session).
    tool_context: dict of shared state between tools (persisted in Flask session).
    Returns: { 'output': str, 'new_messages': list[dict], 'tool_context': dict }
    """
    # Give tools access to the current history so they can read ToolMessages
    set_conversation_context(conversation_history)

    # Restore tool context from previous request (e.g. availability checked last turn)
    if tool_context:
        for key, value in tool_context.items():
            if value is not None:
                set_tool_context(key, value)

    logger.info(f"[AGENT] Invoking agent with {len(conversation_history)} messages")
    logger.info(f"[AGENT] Latest user message: {user_message}")

    # Trim to a sliding window before sending to the LLM.
    # The only state that needs to survive across turns is in tool_context
    # (e.g. availability for reservations) — that's handled separately.
    # Sending the full history from message one just burns tokens.
    MAX_HISTORY_MESSAGES = 12  # ~6 exchanges
    if len(conversation_history) > MAX_HISTORY_MESSAGES:
        logger.info(f"[AGENT] Trimming history: {len(conversation_history)} → {MAX_HISTORY_MESSAGES} messages")
        conversation_history = conversation_history[-MAX_HISTORY_MESSAGES:]

    lc_messages = []
    for msg in conversation_history:
        role = msg.get('role')
        if role == 'user':
            lc_messages.append(HumanMessage(content=msg['content']))
        elif role == 'assistant':
            lc_messages.append(AIMessage(content=msg['content']))
        elif role == 'tool':
            lc_messages.append(ToolMessage(
                content=msg['content'],
                name=msg.get('name', ''),
                tool_call_id=msg.get('tool_call_id', '')
            ))

    response = agent.invoke({'messages': lc_messages})
    response_messages = response.get('messages', [])

    logger.info(f"[AGENT] Response received – {len(response_messages)} messages")

    # log everything for debugging
    for i, msg in enumerate(response_messages):
        msg_type = type(msg).__name__
        content_preview = str(getattr(msg, 'content', ''))[:120]
        logger.info(f"[AGENT] Message {i}: {msg_type} | {content_preview}")
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            logger.info(f"[AGENT]   Tool calls: {msg.tool_calls}")

    # --- extract only the NEW messages (everything after what we sent in) ---
    # The agent echoes back the input messages first, then appends new ones.
    new_lc_messages = response_messages[len(lc_messages):]

    # Serialize new messages for session storage
    new_messages = [_serialize_message(m) for m in new_lc_messages]

    # The final output is the last AIMessage
    output = ""
    for msg in reversed(response_messages):
        if type(msg).__name__ == 'AIMessage' and msg.content:
            output = msg.content
            break

    logger.info(f"[AGENT] Output: {output[:200]}")

    # Snapshot tool context so it persists across requests
    updated_tool_context = {
        'availability': get_tool_context('availability'),
    }

    # Pull reservation JSON out of ToolMessages if make_reservation_tool ran.
    # The IMPORTANT line lives in the ToolMessage, not the LLM's final reply.
    import re as _re
    reservation_json = None
    for msg in response_messages:
        if type(msg).__name__ == 'ToolMessage':
            match = _re.search(r'IMPORTANT: This reservation data includes: ({.*})', str(msg.content))
            if match:
                reservation_json = match.group(1)

    return {
        'output': output,
        'new_messages': new_messages,
        'tool_context': updated_tool_context,
        'reservation_json': reservation_json,
    }