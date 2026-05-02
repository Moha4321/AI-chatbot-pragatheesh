// --- START OF FILE frontend/script.js ---

// We leave this empty! The browser will automatically use localhost OR your Cloudflare URL.
const API_BASE_URL = ""; 

// 1. Load or Create Session ID from localStorage
let sessionId = localStorage.getItem("sessionId");
if (!sessionId) {
    sessionId = "user_" + Math.random().toString(36).substring(2, 10);
    localStorage.setItem("sessionId", sessionId);
}

// 2. Load Chat History from localStorage
let chatHistory = JSON.parse(localStorage.getItem("chatHistory")) ||[];

const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-btn");
const resetButton = document.getElementById("reset-btn");

// Helper to render UI and scroll to bottom
function addMessageToUI(sender, text) {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", sender);
    msgDiv.innerText = text;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return msgDiv;
}

// Render the history when the page first loads
function renderHistory() {
    chatBox.innerHTML = "";
    if (chatHistory.length === 0) {
        addMessageToUI("bot", "Hello! I'm here to support your health goals. How are you feeling today?");
    } else {
        chatHistory.forEach(msg => addMessageToUI(msg.role, msg.content));
    }
}

// RESTART BUTTON LOGIC
if (resetButton) {
    resetButton.addEventListener("click", () => {
        // Clear the memory
        localStorage.removeItem("chatHistory");
        localStorage.removeItem("sessionId");
        
        // Generate a new user session for telemetry
        sessionId = "user_" + Math.random().toString(36).substring(2, 10);
        localStorage.setItem("sessionId", sessionId);
        
        // Reset the UI
        chatHistory =[];
        renderHistory();
    });
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // Save the past history to send to the backend BEFORE we add the new message
    const pastHistory = [...chatHistory]; 

    // Add user message to UI and History
    addMessageToUI("user", text);
    chatHistory.push({ role: "user", content: text });
    localStorage.setItem("chatHistory", JSON.stringify(chatHistory));

    userInput.value = "";
    sendButton.disabled = true;

    // Create an empty bubble for the bot's response
    const botMessageDiv = addMessageToUI("bot", "");
    let fullBotResponse = "";

    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                session_id: sessionId, 
                text: text,
                history: pastHistory // <--- Send memory to the LLM!
            })
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        
        let done = false;
        let buffer = ""; 
        
        while (!done) {
            const { value, done: readerDone } = await reader.read();
            done = readerDone;
            
            if (value) {
                buffer += decoder.decode(value, { stream: true });
                let delimiterIndex;
                
                while ((delimiterIndex = buffer.indexOf("\n\n")) >= 0) {
                    const chunkStr = buffer.slice(0, delimiterIndex);
                    buffer = buffer.slice(delimiterIndex + 2);
                    
                    if (chunkStr.startsWith("data: ")) {
                        const jsonStr = chunkStr.replace("data: ", "");
                        try {
                            const data = JSON.parse(jsonStr);
                            if (data.chunk === "[DONE]") {
                                done = true;
                                break;
                            }
                            botMessageDiv.innerText += data.chunk;
                            fullBotResponse += data.chunk; // Build the full string
                            chatBox.scrollTop = chatBox.scrollHeight;
                        } catch (e) {}
                    }
                }
            }
        }
        
        // Stream is complete! Save the Bot's final response to localStorage
        chatHistory.push({ role: "bot", content: fullBotResponse });
        localStorage.setItem("chatHistory", JSON.stringify(chatHistory));

    } catch (error) {
        console.error("Chat Error:", error);
        botMessageDiv.innerText = "Connection lost. Please try again.";
    } finally {
        sendButton.disabled = false;
        userInput.focus();
    }
}

sendButton.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", function (e) {
    if (e.key === "Enter") sendMessage();
});

// Boot up the UI!
window.onload = renderHistory;

// --- END OF FILE ---