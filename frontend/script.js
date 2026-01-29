const chatButton = document.getElementById('chat-button');
const chatWindow = document.getElementById('chat-window');
const minimizeBtn = document.getElementById('minimize-chat');
const sendBtn = document.getElementById('send-btn');
const userInput = document.getElementById('user-input');
const chatMessages = document.getElementById('chat-messages');
const endSessionBtn = document.getElementById('end-session-btn');
const opStickyBtn = document.getElementById('operator-btn-sticky');

// 1. Toggle/Minimize
chatButton.addEventListener('click', () => chatWindow.classList.toggle('hidden'));
minimizeBtn.addEventListener('click', () => chatWindow.classList.add('hidden'));

// 2. End Session Logic
endSessionBtn.addEventListener('click', () => {
    if(confirm("Are you sure you want to end this session and clear history?")) {
        chatMessages.innerHTML = `
            <div class="message bot-message">Session reset. How can I help you today?</div>
            <div id="faq-container" class="faq-grid"></div>
        `;
        showCategories();
    }
});

// 3. Operator Button
opStickyBtn.addEventListener('click', contactOperator);

async function contactOperator() {
    addMessage("I'd like to speak with an operator.", 'user');
    setTimeout(() => {
        addMessage("Notifying our team... A human operator will join shortly.", 'bot');
    }, 500);

    try {
        await fetch('/api/operator/call', { method: 'POST' });
    } catch (err) {
        console.error("Failed to notify operator");
    }
}

// 4. FAQ Logic
async function showCategories() {
    const faqContainer = document.getElementById('faq-container');
    if (!faqContainer) return;

    faqContainer.innerHTML = 'Loading...';
    try {
        const response = await fetch('/api/faq/categories');
        const categories = await response.json();
        faqContainer.innerHTML = '';
        categories.forEach(cat => {
            const btn = document.createElement('button');
            btn.className = 'faq-btn';
            btn.innerText = cat.label;
            btn.onclick = () => showQuestions(cat.id);
            faqContainer.appendChild(btn);
        });
    } catch (err) {
        faqContainer.innerHTML = 'Error loading options.';
    }
}

async function showQuestions(catId) {
    const faqContainer = document.getElementById('faq-container');
    faqContainer.innerHTML = 'Loading...';
    try {
        const response = await fetch(`/api/faq/questions/${catId}`);
        const questions = await response.json();
        faqContainer.innerHTML = '';

        const backBtn = document.createElement('button');
        backBtn.className = 'faq-btn back-btn';
        backBtn.innerText = '← Back';
        backBtn.onclick = showCategories;
        faqContainer.appendChild(backBtn);

        questions.forEach(item => {
            const btn = document.createElement('button');
            btn.className = 'faq-btn';
            btn.innerText = item.q;
            btn.onclick = () => {
                addMessage(item.q, 'user');
                setTimeout(() => addMessage(item.a, 'bot'), 400);
            };
            faqContainer.appendChild(btn);
        });
    } catch (err) {
        faqContainer.innerHTML = 'Error loading questions.';
    }
}

// 5. Helper to add messages
function addMessage(text, sender) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', `${sender}-message`);
    msgDiv.textContent = text;
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// 6. Main Send Logic (Unified)
async function handleSend() {
    const text = userInput.value.trim();
    if (!text) return;

    // Display user message
    addMessage(text, 'user');
    userInput.value = '';

    // Add typing indicator
    const typingId = 'typing-' + Date.now();
    const typingDiv = document.createElement('div');
    typingDiv.id = typingId;
    typingDiv.classList.add('message', 'bot-message');
    typingDiv.textContent = '...';
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        // Send to backend
        const response = await fetch('/api/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });

        const data = await response.json();

        // Remove typing indicator
        const typingElem = document.getElementById(typingId);
        if (typingElem) typingElem.remove();

        // Handle Logic
        if (data.status === 'direct') {
            // High confidence answer
            addMessage(data.answer, 'bot');
        } else if (data.status === 'pending') {
            // Low confidence -> Waiting for Telegram approval
            // 1. Show the "Wait" message
            addMessage("Let me double check that with our front desk staff...", 'bot');

            // 2. Start polling for the real answer
            pollForAnswer(data.request_id);
        }

    } catch (err) {
        const typingElem = document.getElementById(typingId);
        if (typingElem) typingElem.remove();
        addMessage("Sorry, I'm having trouble connecting to the server.", 'bot');
        console.error("API Error:", err);
    }
}

// 7. Polling Logic
function pollForAnswer(requestId) {
    // Check every 2 seconds
    const interval = setInterval(async () => {
        try {
            const res = await fetch(`/api/check_status/${requestId}`);
            const data = await res.json();

            if (data.status === "completed") {
                clearInterval(interval); // Stop checking
                addMessage(data.answer, 'bot'); // Show the Operator's answer
            }
        } catch (e) {
            console.error("Polling error", e);
            clearInterval(interval); // Stop on error to prevent infinite loop
        }
    }, 2000);
}

// Event Listeners
sendBtn.addEventListener('click', handleSend);
userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleSend(); });

// Initial Load
showCategories();