const chatButton = document.getElementById('chat-button');
const chatWindow = document.getElementById('chat-window');
const minimizeBtn = document.getElementById('minimize-chat');
const sendBtn = document.getElementById('send-btn');
const userInput = document.getElementById('user-input');
const chatMessages = document.getElementById('chat-messages');
const endSessionBtn = document.getElementById('end-session-btn');
const opStickyBtn = document.getElementById('operator-btn-sticky');

// Toggle/Minimize
chatButton.addEventListener('click', () => chatWindow.classList.toggle('hidden'));
minimizeBtn.addEventListener('click', () => chatWindow.classList.add('hidden'));

// End Session Logic
endSessionBtn.addEventListener('click', () => {
    if(confirm("Are you sure you want to end this session and clear history?")) {
        // Reset message area to initial state
        chatMessages.innerHTML = `
            <div class="message bot-message">Session reset. How can I help you today?</div>
            <div id="faq-container" class="faq-grid"></div>
        `;
        showCategories(); // Reload the buttons
    }
});

// Operator Button
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

function addMessage(text, sender) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', `${sender}-message`);
    msgDiv.textContent = text;
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function handleSend() {
    const text = userInput.value.trim();
    if (!text) return;

    // 1. Display user message
    addMessage(text, 'user');
    userInput.value = '';

    // 2. Add a temporary "typing..." indicator
    const typingId = 'typing-' + Date.now();
    const typingDiv = document.createElement('div');
    typingDiv.id = typingId;
    typingDiv.classList.add('message', 'bot-message');
    typingDiv.textContent = '...';
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        // 3. Send to our new backend API
        const response = await fetch('/api/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });

        const data = await response.json();

        // 4. Remove typing indicator
        document.getElementById(typingId).remove();

        // 5. Handle the response based on status
        if (data.status === 'direct') {
            addMessage(data.answer, 'bot');
        } else if (data.status === 'pending_approval') {
            // Show the "checking..." message to user
            addMessage(data.answer, 'bot');
            // Log suggestion for debugging (in Phase 4 this goes to Telegram)
            console.log("Suggestion awaiting approval:", data.suggested);
        }
    } catch (err) {
        document.getElementById(typingId).remove();
        addMessage("Sorry, I'm having trouble connecting to the server.", 'bot');
        console.error("API Error:", err);
    }
}

sendBtn.addEventListener('click', handleSend);
userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleSend(); });

// Initial Load
showCategories();