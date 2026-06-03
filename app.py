"""
BiteBot Flask Application

A conversational interface for restaurant discovery and reservations.
Multi-agent graph architecture — supervisor routes to restaurant or support agent.
"""

from flask import Flask, render_template, request, jsonify, session
import os
import uuid
import logging
from datetime import datetime
from dotenv import load_dotenv
import newrelic.agent

from src.multi_agent_graph import create_multi_agent_graph, run_multi_agent_system

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24).hex())

# Single multi-agent system shared across all sessions.
# Internally contains supervisor, restaurant, and support agents.
multi_agent_system = None

try:
    print("Initializing BiteBot multi-agent system...")
    multi_agent_system = create_multi_agent_graph()
    print("✅ BiteBot ready!")
except Exception as e:
    print(f"❌ Error initializing multi-agent system: {e}")
    multi_agent_system = None


@app.route('/')
def index():
    """Render the main chat interface."""
    if 'messages'     not in session:
        session['messages']     = []
    if 'reservations' not in session:
        session['reservations'] = []
    if 'thread_id'    not in session:
        session['thread_id']    = str(uuid.uuid4())

    # tool_context is no longer needed — graph state handles it
    # Kept for any existing templates that reference it
    if 'tool_context' not in session:
        session['tool_context'] = {}

    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages through the multi-agent graph."""
    if not multi_agent_system:
        return jsonify({
            'error': 'Multi-agent system not initialized. Please check your configuration.'
        }), 500

    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        # ── Session data ──────────────────────────────────────────────────────
        ui_messages  = session.get('messages', [])
        reservations = session.get('reservations', [])

        # thread_id must be resolved before calling the graph
        thread_id = session.get('thread_id')
        if not thread_id:
            thread_id = str(uuid.uuid4())
            session['thread_id'] = thread_id

        # ── Invoke the multi-agent graph ──────────────────────────────────────
        # One call handles: supervisor routing → agent → tool execution
        # New Relic automatically instruments the full chain
        response = run_multi_agent_system(
            multi_agent_system,
            user_message,
            thread_id,
            conversation_history=ui_messages,
            reservations=reservations,
        )

        assistant_message = response.get('output', 'Sorry, I encountered an error.')
        agent_used        = response.get('agent_used', 'restaurant')

        logger.info(f"Agent used: {agent_used}")

        # ── Reservation handling ──────────────────────────────────────────────

        # Support agent may have modified existing reservations
        updated_reservations = response.get('reservations', reservations)

        # Restaurant agent may have created a new reservation
        reservation_json = response.get('reservation_json')
        if reservation_json:
            existing_ids = [r['reservation_id'] for r in updated_reservations]
            if reservation_json['reservation_id'] not in existing_ids:
                updated_reservations.append(reservation_json)
                logger.info(f"New reservation saved: {reservation_json['reservation_id']}")

        session['reservations'] = updated_reservations

        # ── Update UI message history ─────────────────────────────────────────
        ui_messages.append({'role': 'user',      'content': user_message})
        ui_messages.append({'role': 'assistant', 'content': assistant_message})
        session['messages']  = ui_messages
        session.modified = True

        # Capture trace ID for feedback correlation
        trace_id = newrelic.agent.current_trace_id()

        return jsonify({
            'message':      assistant_message,
            'trace_id':     trace_id,
            'reservations': session.get('reservations', []),
            'agent':        agent_used,
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        newrelic.agent.notice_error()
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.route('/reset', methods=['POST'])
def reset():
    """Reset the conversation — new thread_id gives a fresh agent memory."""
    session['messages']     = []
    session['reservations'] = []
    session['tool_context'] = {}
    session['thread_id']    = str(uuid.uuid4())
    session.modified = True
    return jsonify({'status': 'ok'})


@app.route('/reservations', methods=['GET'])
def get_reservations():
    """Get all reservations for the current session."""
    return jsonify({
        'reservations': session.get('reservations', [])
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status':              'healthy' if multi_agent_system else 'degraded',
        'multi_agent_system':  multi_agent_system is not None,
        'timestamp':           datetime.now().isoformat()
    })


@app.route('/feedback', methods=['POST'])
def record_feedback():
    """Record user feedback on agent responses."""
    data     = request.json
    trace_id = data.get('trace_id')
    rating   = data.get('rating')
    category = data.get('category', None)
    message  = data.get('message',  None)

    newrelic.agent.record_llm_feedback_event(
        trace_id=trace_id,
        rating=rating,
        category=category,
        message=message,
        metadata={'session_id': session.get('thread_id')}
    )

    return jsonify({'status': 'recorded'})


if __name__ == '__main__':
    port  = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)