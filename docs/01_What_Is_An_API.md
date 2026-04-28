# Lesson 1: What is an API? (And how our app works)

Hey Pragatheesh! To build this app, your website (Frontend) needs to talk to my Mac M4 (Backend). They talk using an **API** (Application Programming Interface).

### The Restaurant Analogy
*   **The Customer (Frontend):** This is your HTML/JS website. The user types, "I'm stressed, I want a burger."
*   **The Kitchen (Backend):** This is my Mac M4 running the LLM and the Vector Math. It processes the text and cooks up a response.
*   **The Waiter (The API):** The frontend can't go into the kitchen. It hands the request to the Waiter (API). The API carries the message over the internet via a Cloudflare Tunnel, drops it in the Mac M4, waits for the LLM to reply, and carries the response back to the website.

**Your Job:** You will write the JavaScript `fetch()` command. This is literally you snapping your fingers and calling the waiter to take the user's message to the backend.