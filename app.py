"""
BiteBot Flask Application

A conversational interface for restaurant discovery and reservations.
"""

from flask import Flask, render_template, request, jsonify, session
import os
import uuid
import logging
from datetime import datetime
from dotenv import load_dotenv
import newrelic.agent

from src.discovery_and_reservation_agent import create_discovery_and_reservation_agent, run_agent
from src.customer_support_agent import create_support_agent, run_support_agent
from src.supervisor_agent import create_supervisor, route_request

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24).hex())

# Initialize multi-agent system (shared across sessions)
restaurant_agent = None
support_agent = None
supervisor = None

try:
    print("Initializing BiteBot multi-agent system...")
    restaurant_agent = create_discovery_and_reservation_agent()
    print("Restaurant agent initialized")
    support_agent = create_support_agent()
    print("Support agent initialized")
    supervisor = create_supervisor()
    print("Supervisor initialized")
    print("✅ BiteBot ready!")
except Exception as e:
    print(f"❌ Error initializing agents: {e}")
    restaurant_agent = None
    support_agent = None
    supervisor = None


@app.route('/')
def index():
    """Render the main chat interface."""
    if 'messages' not in session:
        session['messages'] = []
    if 'reservations' not in session:
        session['reservations'] = []
    if 'tool_context' not in session:
        session['tool_context'] = {}
    if 'thread_id' not in session:
        session['thread_id'] = str(uuid.uuid4())

    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with multi-agent routing."""
    if not all([restaurant_agent, support_agent, supervisor]):
        return jsonify({
            'error': 'Agents not initialized. Please check your configuration.'
        }), 500

    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        # Get session data
        ui_messages = session.get('messages', [])
        tool_context = session.get('tool_context', {})
        reservations = session.get('reservations', [])

        # Route to appropriate agent
        agent_choice = route_request(supervisor, user_message, ui_messages)
        logger.info(f"Routing to {agent_choice} agent")

        # thread_id scopes the agent's memory to this session.
        thread_id = session.get('thread_id')
        if not thread_id:
            thread_id = str(uuid.uuid4())
            session['thread_id'] = thread_id

        # Capture trace ID for feedback correlation
        trace_id = newrelic.agent.current_trace_id()

        # Call the selected agent
        if agent_choice == "support":
            response = run_support_agent(
                support_agent, user_message, thread_id, reservations
            )
            assistant_message = response.get('output', 'Sorry, I encountered an error.')
            # Support agent may have modified reservations
            session['reservations'] = response.get('reservations', reservations)

        else:  # restaurant
            response = run_agent(
                restaurant_agent, user_message, thread_id, tool_context
            )
            assistant_message = response.get('output', 'Sorry, I encountered an error.')

            # Store reservation if created
            reservation_json = response.get('reservation_json')
            if reservation_json:
                # reservation_json is already a dict (not a JSON string)
                reservation = reservation_json
                existing_ids = [r['reservation_id'] for r in reservations]
                if reservation['reservation_id'] not in existing_ids:
                    reservations.append(reservation)
                    session['reservations'] = reservations
                    logger.info(f"Reservation saved: {reservation['reservation_id']}")

            # Update tool context
            session['tool_context'] = response.get('tool_context', {})

        # session['messages'] is for the chat UI only — the agent's memory
        # lives in the checkpointer keyed by thread_id.
        ui_messages.append({'role': 'user',      'content': user_message})
        ui_messages.append({'role': 'assistant', 'content': assistant_message})
        session['messages'] = ui_messages
        session.modified = True

        return jsonify({
            'message': assistant_message,
            'trace_id': trace_id,
            'reservations': session.get('reservations', []),
            'agent': agent_choice  # Let frontend know which agent responded
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.route('/reset', methods=['POST'])
def reset():
    """Reset the conversation."""
    session['messages'] = []
    session['reservations'] = []
    session['tool_context'] = {}
    session['thread_id'] = str(uuid.uuid4())  # new thread = fresh agent memory
    session.modified = True
    return jsonify({'status': 'ok'})


@app.route('/reservations', methods=['GET'])
def get_reservations():
    """Get all reservations."""
    return jsonify({
        'reservations': session.get('reservations', [])
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for monitoring."""
    all_initialized = all([restaurant_agent, support_agent, supervisor])
    return jsonify({
        'status': 'healthy' if all_initialized else 'degraded',
        'restaurant_agent': restaurant_agent is not None,
        'support_agent': support_agent is not None,
        'supervisor': supervisor is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/feedback', methods=['POST'])
def record_feedback():
    """Record user feedback on agent responses."""
    data = request.json
    trace_id = data.get('trace_id')  # Received from frontend
    rating = data.get('rating')  # e.g., 'thumbs_up' or 'thumbs_down'
    category = data.get('category', None)  # Optional: e.g., 'inaccurate', 'helpful'
    message = data.get('message', None)  # Optional: freeform text
    
    newrelic.agent.record_llm_feedback_event(
        trace_id=trace_id,
        rating=rating,
        category=category,
        message=message,
        metadata={'session_id': session.get('thread_id')}
    )
    
    return jsonify({'status': 'recorded'})


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)