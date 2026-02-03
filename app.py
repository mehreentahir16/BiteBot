"""
BiteBot Flask Application

A conversational interface for restaurant discovery and reservations.
"""

from flask import Flask, render_template, request, jsonify, session
import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from src.agent import create_bitebot_agent, run_agent

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24).hex())

# Initialize agent (shared across sessions)
try:
    agent = create_bitebot_agent()
    print("✅ BiteBot initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing agent: {e}")
    agent = None


@app.route('/')
def index():
    """Render the main chat interface."""
    if 'messages' not in session:
        session['messages'] = []
    if 'reservations' not in session:
        session['reservations'] = []
    if 'tool_context' not in session:
        session['tool_context'] = {}

    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    if not agent:
        return jsonify({
            'error': 'Agent not initialized. Please check your configuration.'
        }), 500

    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        # Get conversation history and tool context from session
        conversation_history = session.get('messages', [])
        tool_context = session.get('tool_context', {})

        # Add user message to history
        conversation_history.append({
            'role': 'user',
            'content': user_message
        })

        # Get agent response (passes tool_context so tools can share state across turns)
        response = run_agent(agent, user_message, conversation_history, tool_context)
        assistant_message = response.get('output', 'Sorry, I encountered an error.')

        # Store reservation if make_reservation_tool fired this turn
        reservation_json = response.get('reservation_json')
        if reservation_json:
            try:
                reservation = json.loads(reservation_json)
                reservations = session.get('reservations', [])
                existing_ids = [r['reservation_id'] for r in reservations]
                if reservation['reservation_id'] not in existing_ids:
                    reservations.append(reservation)
                    session['reservations'] = reservations
            except json.JSONDecodeError:
                pass

        # Clean up the output (remove IMPORTANT line)
        clean_message = re.sub(r'\n*IMPORTANT: This reservation data includes: {.*?}\n*', '', assistant_message)

        # Only persist the final assistant message — NOT the intermediate
        # AIMessage+ToolMessage pairs.  Those have tool_calls metadata that
        # _serialize_message strips, which makes OpenAI reject the replay
        # ("tool message must follow a message with tool_calls").
        # tool_context handles inter-tool state across turns now.
        conversation_history.append({
            'role': 'assistant',
            'content': clean_message
        })

        # Update session
        session['messages'] = conversation_history
        session['tool_context'] = response.get('tool_context', {})
        session.modified = True

        return jsonify({
            'message': clean_message,
            'reservations': session.get('reservations', [])
        })

    except Exception as e:
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
    return jsonify({
        'status': 'healthy',
        'agent_initialized': agent is not None,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)