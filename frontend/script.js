// --- START OF FILE frontend/script.js ---

const API_BASE_URL = ""; 

// 1. Session & History Setup
let sessionId = localStorage.getItem("sessionId");
if (!sessionId) {
    sessionId = "user_" + Math.random().toString(36).substring(2, 10);
    localStorage.setItem("sessionId", sessionId);
}

let chatHistory = JSON.parse(localStorage.getItem("chatHistory")) ||[];

const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-btn");
const resetButton = document.getElementById("reset-btn");

// 2. Render helper
function addMessageToUI(sender, text) {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", sender);
    msgDiv.innerText = text;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return msgDiv;
}

// 3. Initial Load
function renderHistory() {
    chatBox.innerHTML = "";
    if (chatHistory.length === 0) {
        addMessageToUI("bot", "Hello! I'm here to support your health goals. How are you feeling today?");
    } else {
        chatHistory.forEach(msg => addMessageToUI(msg.role, msg.content));
    }
}

// 4. Restart Button Logic
if (resetButton) {
    resetButton.addEventListener("click", () => {
        localStorage.removeItem("chatHistory");
        localStorage.removeItem("sessionId");
        sessionId = "user_" + Math.random().toString(36).substring(2, 10);
        localStorage.setItem("sessionId", sessionId);
        chatHistory =[];
        renderHistory();
    });
}

// 5. Send Message Logic
async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // Snapshot history before adding the new message
    const pastHistory = [...chatHistory]; 

    // Render User text instantly
    addMessageToUI("user", text);
    chatHistory.push({ role: "user", content: text });
    localStorage.setItem("chatHistory", JSON.stringify(chatHistory));

    userInput.value = "";
    sendButton.disabled = true;

    // Create an empty bot bubble
    const botMessageDiv = addMessageToUI("bot", "");
    let fullBotResponse = "";

    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                session_id: sessionId, 
                text: text,
                history: pastHistory
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
                            fullBotResponse += data.chunk; 
                            botMessageDiv.innerText = fullBotResponse; // Update bubble text
                            chatBox.scrollTop = chatBox.scrollHeight;
                        } catch (e) {
                            console.error("JSON Parse Error:", e);
                        }
                    }
                }
            }
        }
        
        // Save the final response
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

// 6. Listeners
sendButton.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", function (e) {
    if (e.key === "Enter") sendMessage();
});

// Boot UI
window.onload = renderHistory;

// --- END OF FILE ---