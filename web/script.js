const apiUrl = 'ws://127.0.0.1:8000/ws/chat';

let ws;

function connect() {
    ws = new WebSocket(apiUrl);
    ws.onopen = () => console.log('WebSocket connected');
    ws.onclose = () => {
        console.log('WebSocket closed, reconnecting...');
        setTimeout(connect, 1000);
    };
    ws.onerror = (error) => console.error('WebSocket error:', error);
}

connect();

const chat = document.getElementById('chat');
const promptInput = document.getElementById('prompt');
const sendButton = document.getElementById('send');

sendButton.addEventListener('click', sendMessage);
promptInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function sendRequest(prompt, skipCached = false) {
    return new Promise((resolve, reject) => {
        if (ws.readyState !== WebSocket.OPEN) {
            reject(new Error('WebSocket not connected'));
            return;
        }
        const message = { prompt, skip_cached: skipCached };
        ws.send(JSON.stringify(message));
        const onMessage = (event) => {
            ws.removeEventListener('message', onMessage);
            try {
                const data = JSON.parse(event.data);
                if (!data || typeof data.result !== 'string' || typeof data.cached !== 'boolean') {
                    throw new Error('Malformed API response');
                }
                resolve(data);
            } catch (error) {
                reject(error);
            }
        };
        ws.addEventListener('message', onMessage);
    });
}

function sendMessage() {
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    addMessage(prompt, 'user');
    promptInput.value = '';

    const loadingDiv = addMessage('Generating response...', 'loading');
    scrollToBottom();

    sendRequest(prompt)
        .then(data => {
            loadingDiv.remove();
            const result = data.result;
            const cached = data.cached;
            const displayText = result;
            const messageDiv = addMessage(displayText, 'ai', true);
            if (cached) {
                const cachedDiv = document.createElement('div');
                cachedDiv.className = 'cached-info';
                cachedDiv.textContent = 'Cached';
                messageDiv.appendChild(cachedDiv);
            }
            addSatisfactionQuestion(messageDiv, prompt);
            scrollToBottom();
        })
        .catch(error => {
            loadingDiv.remove();
            addMessage(`Error: ${error.message}`, 'error');
            scrollToBottom();
        });
}

function addMessage(text, type, isMarkdown = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    if (type === 'loading') {
        const spinner = document.createElement('div');
        spinner.className = 'spinner';
        messageDiv.appendChild(spinner);
    } else if (isMarkdown) {
        try {
            messageDiv.innerHTML = marked.parse(text);
        } catch (error) {
            messageDiv.textContent = `Error rendering markdown: ${error.message}`;
        }
    } else {
        messageDiv.textContent = text;
    }
    chat.appendChild(messageDiv);
    return messageDiv;
}

function addSatisfactionQuestion(messageDiv, prompt) {
    const questionDiv = document.createElement('div');
    questionDiv.className = 'satisfaction-question';
    questionDiv.innerHTML = 'Are you satisfied with the response? <button class="yes-btn">Yes</button> <button class="no-btn">No</button>';
    messageDiv.appendChild(questionDiv);
    questionDiv.querySelector('.yes-btn').addEventListener('click', () => {
        questionDiv.remove();
    });
    questionDiv.querySelector('.no-btn').addEventListener('click', () => {
        questionDiv.remove();
        sendRetry(prompt);
    });
}

function sendRetry(prompt) {
    const loadingDiv = addMessage('Generating new response...', 'loading');
    scrollToBottom();
    sendRequest(prompt, true)
        .then(data => {
            loadingDiv.remove();
            const result = data.result;
            const cached = data.cached;
            const displayText = result;
            const messageDiv = addMessage(displayText, 'ai', true);
            if (cached) {
                const cachedDiv = document.createElement('div');
                cachedDiv.className = 'cached-info';
                cachedDiv.textContent = 'Cached';
                messageDiv.appendChild(cachedDiv);
            }
            scrollToBottom();
        })
        .catch(error => {
            loadingDiv.remove();
            addMessage(`Error: ${error.message}`, 'error');
            scrollToBottom();
        });
}

function scrollToBottom() {
    chat.scrollTop = chat.scrollHeight;
}
