import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import json
import logging
from contextlib import asynccontextmanager
from typing import List, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Import your custom thesis modules!
from embedding_engine import EmbeddingEngine
from rag_engine import RAGEngine
from llm_engine import LLMEngine
from prompts import format_llama3_prompt
from study_logger import log_interaction

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Engine Singletons ---
embedder = None
rag = None
llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup sequence. Loads models into M4 Unified Memory exactly once.
    """
    global embedder, rag, llm
    logger.info("Starting up M4 AI Chatbot Server...")
    
    # 1. Load the MPS Embedding Engine & Knowledge Base
    embedder = EmbeddingEngine()
    
    # 2. Initialize RAG Orchestrator (Threshold=0.55, Lambda=0.7)
    rag = RAGEngine(intent_threshold=0.55, mmr_lambda=0.7, top_k=3)
    
    # 3. Load the 4-bit Llama-3 into Unified Memory (~4.5GB RAM)
    llm = LLMEngine()
    
    logger.info("All engines loaded successfully. Server is ready!")
    yield
    logger.info("Shutting down server. Freeing RAM...")

# Initialize FastAPI
app = FastAPI(lifespan=lifespan, title="Dietary Intervention RAG Chatbot")

# CORS is required so your GitHub Pages frontend can communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For the pilot study, we allow all origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Data Schema
class ChatRequest(BaseModel):
    session_id: str
    text: str
    history: List[Dict[str, str]] =[]

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    The main interaction endpoint.
    Takes user text -> checks for craving (RAG) -> Prompts LLM -> Streams response.
    """
    # 1. Run the RAG logic
    rag_result = rag.process_query(req.text, embedder)
    
    if rag_result.triggered:
        logger.info(f"RAG Triggered! Score: {rag_result.intent_score:.3f}")
    
    # 2. Format the Llama-3 prompt WITH HISTORY!
    full_prompt = format_llama3_prompt(req.text, req.history, rag_result.context_string)
    
    # 3. Create an asynchronous streaming generator
    def response_generator():
        full_response = ""
        
        # Pull tokens from your MLX engine
        for token in llm.generate_response_stream(full_prompt, max_tokens=150):
            # Filter out Llama-3 special stop tokens so the user doesn't see them
            if "<|eot_id|>" in token or "<|start_header_id|>" in token:
                break
                
            full_response += token
            
            # Yield as Server-Sent Events (SSE) format to the frontend
            yield f"data: {json.dumps({'chunk': token})}\n\n"
            
        # Send a special [DONE] flag so the frontend knows to stop listening
        yield f"data: {json.dumps({'chunk': '[DONE]'})}\n\n"
        
        # 4. Log the interaction to our CSV file behind the scenes!
        log_interaction(
            session_id=req.session_id,
            user_message=req.text,
            rag_triggered=rag_result.triggered,
            intent_score=rag_result.intent_score,
            bot_response=full_response.strip()
        )

    # Return the stream directly to the user's browser
    return StreamingResponse(response_generator(), media_type="text/event-stream")

@app.get("/health")
def health_check():
    """Simple endpoint to verify the server is alive."""
    return {"status": "online", "hardware": "Apple Silicon (M4)"}

# --- Serve Frontend ---
# This points to your frontend folder one directory up
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/")
def serve_index():
    """Serve the main HTML interface."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# Mount the rest of the frontend folder (like script.js) so the browser can download it
app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="static")