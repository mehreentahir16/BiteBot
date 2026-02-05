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
                addMessage(`Error: ${data.error}`, 'assistant');
            } else {
                addMessage(data.message, 'assistant');
                
                // Always update reservations, even if empty.
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
            addMessage(`Error: ${error.message}`, 'assistant');
            sendButton.disabled = false;
            messageInput.focus();
        });
    }

    function addMessage(content, role) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        // Convert markdown-style formatting to HTML
        let htmlContent = content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');
        
        messageDiv.innerHTML = htmlContent;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
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