document.addEventListener('DOMContentLoaded', function() {
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const chatMessages = document.getElementById('chatMessages');
    const welcomeMessage = document.getElementById('welcomeMessage');
    const resetButton = document.getElementById('resetButton');
    const reservationsList = document.getElementById('reservationsList');

    // Load reservations on page load
    loadReservations();

    // Send message on button click
    sendButton.addEventListener('click', sendMessage);

    // Send message on Enter key
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Reset conversation
    resetButton.addEventListener('click', resetConversation);

    function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;

        // Hide welcome message on first interaction
        if (welcomeMessage) {
            welcomeMessage.style.display = 'none';
        }

        // Add user message to chat
        addMessage(message, 'user');

        // Clear input
        messageInput.value = '';
        sendButton.disabled = true;

        // Show loading indicator
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message assistant loading-message';
        loadingDiv.innerHTML = '<span class="loading"></span> <span class="loading"></span> <span class="loading"></span>';
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Send to server
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading indicator
            chatMessages.removeChild(loadingDiv);

            if (data.error) {
                addMessage(data.error, 'assistant', null);
            } else {
                // Pass trace_id to addMessage
                addMessage(data.message, 'assistant', data.trace_id);
                
                // Always update reservations
                if (data.reservations !== undefined) {
                    updateReservations(data.reservations);
                }
            }

            sendButton.disabled = false;
            messageInput.focus();
        })
        .catch(error => {
            // Remove loading indicator
            chatMessages.removeChild(loadingDiv);
            addMessage(`Error: ${error.message}`, 'assistant', null);
            sendButton.disabled = false;
            messageInput.focus();
        });
    }

    function addMessage(content, role, traceId = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        // Convert markdown-style formatting to HTML
        let htmlContent = content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');
        
        messageDiv.innerHTML = htmlContent;
        
        // Add feedback buttons for assistant messages only
        if (role === 'assistant' && traceId) {
            const feedbackDiv = document.createElement('div');
            feedbackDiv.className = 'feedback-buttons';
            feedbackDiv.innerHTML = `
                <button class="feedback-btn thumbs-up" data-trace-id="${traceId}" title="Helpful">
                    👍
                </button>
                <button class="feedback-btn thumbs-down" data-trace-id="${traceId}" title="Not helpful">
                    👎
                </button>
            `;
            messageDiv.appendChild(feedbackDiv);
            
            // Add event listeners to feedback buttons
            const thumbsUp = feedbackDiv.querySelector('.thumbs-up');
            const thumbsDown = feedbackDiv.querySelector('.thumbs-down');
            
            thumbsUp.addEventListener('click', () => sendFeedback(traceId, 'thumbs_up', thumbsUp, thumbsDown));
            thumbsDown.addEventListener('click', () => sendFeedback(traceId, 'thumbs_down', thumbsUp, thumbsDown));
        }
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function sendFeedback(traceId, rating, thumbsUpBtn, thumbsDownBtn) {
        // Disable both buttons after clicking
        thumbsUpBtn.disabled = true;
        thumbsDownBtn.disabled = true;
        
        // Highlight the selected button
        if (rating === 'thumbs_up') {
            thumbsUpBtn.style.opacity = '1';
            thumbsDownBtn.style.opacity = '0.3';
        } else {
            thumbsDownBtn.style.opacity = '1';
            thumbsUpBtn.style.opacity = '0.3';
        }
        
        fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                trace_id: traceId,
                rating: rating
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Feedback recorded:', data);
        })
        .catch(error => {
            console.error('Error recording feedback:', error);
            // Re-enable buttons on error
            thumbsUpBtn.disabled = false;
            thumbsDownBtn.disabled = false;
        });
    }

    function loadReservations() {
        fetch('/reservations')
            .then(response => response.json())
            .then(data => {
                updateReservations(data.reservations);
            })
            .catch(error => {
                console.error('Error loading reservations:', error);
            });
    }

    function updateReservations(reservations) {
        if (!reservations || reservations.length === 0) {
            reservationsList.innerHTML = '<p class="no-reservations">No reservations yet</p>';
            return;
        }

        reservationsList.innerHTML = '';
        reservations.forEach(reservation => {
            const card = document.createElement('div');
            card.className = 'reservation-card';
            card.innerHTML = `
                <h4>🍽️ ${reservation.restaurant_name}</h4>
                <p><strong>Date:</strong> ${reservation.date}</p>
                <p><strong>Time:</strong> ${reservation.time}</p>
                <p><strong>Party:</strong> ${reservation.party_size} people</p>
                <p><strong>Name:</strong> ${reservation.customer_name}</p>
                <p><strong>Confirmation:</strong> <span class="reservation-id">${reservation.reservation_id}</span></p>
            `;
            reservationsList.appendChild(card);
        });
    }

    function resetConversation() {
        if (!confirm('Are you sure you want to reset the conversation and clear all reservations?')) {
            return;
        }

        fetch('/reset', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            // Clear chat messages
            chatMessages.innerHTML = '';
            
            // Show welcome message again
            if (welcomeMessage) {
                welcomeMessage.style.display = 'block';
            }

            // Clear reservations
            reservationsList.innerHTML = '<p class="no-reservations">No reservations yet</p>';

            messageInput.focus();
        })
        .catch(error => {
            alert(`Error resetting conversation: ${error.message}`);
        });
    }
});