// Global variables
let isProcessing = false;

// DOM elements
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    updateStatus();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
}

// Send message to the API with streaming support
async function sendMessage() {
    if (isProcessing) return;
    
    const message = chatInput.value.trim();
    if (!message) return;
    
    isProcessing = true;
    sendButton.disabled = true;
    chatInput.value = '';
    
    // Add user message to chat
    addMessage('user', message);
    
    // Check if streaming is supported and use it
    const useStreaming = true; // Enable streaming by default
    
    if (useStreaming) {
        await sendMessageStream(message);
    } else {
        await sendMessageNormal(message);
    }
    
    isProcessing = false;
    sendButton.disabled = false;
    chatInput.focus();
}

// Streaming version of send message
async function sendMessageStream(message) {
    // Create message element for streaming
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    const textP = document.createElement('p');
    contentDiv.appendChild(textP);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    let fullText = '';
    let sources = null;
    
    try {
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const jsonStr = line.slice(6);
                    if (jsonStr.trim()) {
                        try {
                            const data = JSON.parse(jsonStr);
                            
                            if (data.event === 'token') {
                                fullText += data.content;
                                textP.innerHTML = formatMessage(fullText);
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            } else if (data.event === 'sources' && data.sources && data.sources.length > 0) {
                                sources = data.sources;
                                // Add sources to the message
                                const sourcesDiv = document.createElement('div');
                                sourcesDiv.className = 'sources';
                                sourcesDiv.innerHTML = '<small>Sources: </small>';
                                
                                // Deduplicate sources by title
                                const uniqueSources = [];
                                const seenTitles = new Set();
                                
                                for (const source of sources) {
                                    const title = source.metadata?.title || 'Document';
                                    if (!seenTitles.has(title)) {
                                        seenTitles.add(title);
                                        uniqueSources.push(source);
                                    }
                                }
                                
                                uniqueSources.forEach(source => {
                                    const sourceSpan = document.createElement('span');
                                    sourceSpan.className = 'source-item';
                                    sourceSpan.textContent = source.metadata?.title || 'Document';
                                    sourceSpan.title = `Score: ${(source.score * 100).toFixed(1)}%`;
                                    sourcesDiv.appendChild(sourceSpan);
                                });
                                
                                contentDiv.appendChild(sourcesDiv);
                            } else if (data.event === 'error') {
                                textP.innerHTML = `Sorry, I encountered an error: ${data.error}`;
                            }
                        } catch (e) {
                            console.error('Error parsing SSE data:', e);
                        }
                    }
                }
            }
        }
    } catch (error) {
        textP.innerHTML = 'Sorry, I couldn\'t connect to the server. Please check your connection and try again.';
        console.error('Streaming error:', error);
    }
}

// Normal (non-streaming) version of send message
async function sendMessageNormal(message) {
    // Show typing indicator
    const typingId = showTypingIndicator();
    
    try {
        // Send to API
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        if (response.ok) {
            // Add assistant response with sources
            addMessage('assistant', data.response, data.sources);
        } else {
            addMessage('assistant', `Sorry, I encountered an error: ${data.error || 'Please try again.'}`);
        }
    } catch (error) {
        removeTypingIndicator(typingId);
        addMessage('assistant', 'Sorry, I couldn\'t connect to the server. Please check your connection and try again.');
        console.error('Chat error:', error);
    }
}

// Add message to chat
function addMessage(type, content, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Parse content for better formatting
    const formattedContent = formatMessage(content);
    contentDiv.innerHTML = formattedContent;
    
    // Add sources if available
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';
        sourcesDiv.innerHTML = '<small>Sources: </small>';
        
        // Deduplicate sources by title
        const uniqueSources = [];
        const seenTitles = new Set();
        
        for (const source of sources) {
            const title = source.metadata?.title || 'Document';
            if (!seenTitles.has(title)) {
                seenTitles.add(title);
                uniqueSources.push(source);
            }
        }
        
        uniqueSources.forEach(source => {
            const sourceSpan = document.createElement('span');
            sourceSpan.className = 'source-item';
            sourceSpan.textContent = source.metadata?.title || 'Document';
            sourceSpan.title = `Score: ${(source.score * 100).toFixed(1)}%`;
            sourcesDiv.appendChild(sourceSpan);
        });
        
        contentDiv.appendChild(sourcesDiv);
    }
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Format message content
function formatMessage(content) {
    // Convert markdown-like formatting to HTML
    let formatted = content
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/^- (.*?)$/gm, '<li>$1</li>')
        .replace(/^(\d+)\. (.*?)$/gm, '<li>$2</li>');
    
    // Wrap in paragraph if not already
    if (!formatted.startsWith('<')) {
        formatted = `<p>${formatted}</p>`;
    }
    
    // Convert consecutive li tags to ul
    formatted = formatted.replace(/(<li>.*?<\/li>\s*)+/g, (match) => {
        return `<ul>${match}</ul>`;
    });
    
    return formatted;
}

// Show typing indicator
function showTypingIndicator() {
    const typingId = 'typing-' + Date.now();
    const typingDiv = document.createElement('div');
    typingDiv.id = typingId;
    typingDiv.className = 'message assistant';
    typingDiv.innerHTML = `
        <div class="message-content">
            <div class="loading-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return typingId;
}

// Remove typing indicator
function removeTypingIndicator(typingId) {
    const typingDiv = document.getElementById(typingId);
    if (typingDiv) {
        typingDiv.remove();
    }
}

// Update status
async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        if (response.ok && data.status === 'operational') {
            statusDot.style.backgroundColor = '#10b981';
            statusText.textContent = 'Connected';
        } else {
            statusDot.style.backgroundColor = '#f59e0b';
            statusText.textContent = 'Limited';
        }
    } catch (error) {
        statusDot.style.backgroundColor = '#ef4444';
        statusText.textContent = 'Offline';
    }
}

// Clear chat (utility function)
function clearChat() {
    const messages = chatMessages.querySelectorAll('.message.user, .message.assistant:not(:first-child)');
    messages.forEach(msg => msg.remove());
}