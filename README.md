# Dietary Health Coach: A Local RAG Intervention System

This repository contains the software artifact for my Projects. It is an autonomous, privacy-preserving dietary intervention chatbot built entirely on local hardware (Apple Silicon M4).

The system combines **Motivational Interviewing (MI)** with **Retrieval-Augmented Generation (RAG)** to deliver real-time, evidence-based health facts when users experience food cravings.

---

# System Architecture

- **Hardware:** Apple Silicon (M-Series) >= 16GB Unified Memory
- **LLM Engine:** `Meta-Llama-3-8B-Instruct-4bit` (via Apple MLX)
- **Embedding Engine:** `all-MiniLM-L6-v2` (Hardware accelerated via PyTorch MPS)
- **Vector Core:** Custom NumPy implementations of Cosine Similarity and Maximal Marginal Relevance (MMR)
- **Backend:** FastAPI with Server-Sent Events (SSE) streaming
- **Telemetry:** Automated local CSV logging for N=32 pilot study

---

# 🛠 Mac Setup & Installation

## 1. Prerequisites

Ensure you have the following installed on your Mac:

- Python 3.10+
- Homebrew

---

## 2. Environment Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/YOUR-USERNAME/AI-chatbot-pragatheesh.git
cd AI-chatbot-pragatheesh

# Recommended: Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 3. Model Weight Caching

The first time you run the server, the system automatically downloads the required model weights from HuggingFace.

### Download Sizes

- **Embedder (`all-MiniLM-L6-v2`)**: ~90 MB
- **Llama-3 8B (4-bit quantized)**: ~4.5 GB

---

# 🚀 Running the System

## Step 1: Start the Local API

The backend is designed to load heavy models into RAM exactly once during the server lifespan.

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Wait for the terminal output:

```text
To use locally: Open http://localhost:8000 in your web browser.
```

---

## Step 2: Open the Cloudflare Tunnel (Optional External Access)

To expose the local AI system externally (e.g., for mobile access or pilot study participants) without port-forwarding:

### Install Cloudflare Tunnel

```bash
brew install cloudflared
```

### Start the HTTP/2 Tunnel

```bash
cloudflared tunnel --url http://localhost:8000 --protocol http2
```

Look for a generated URL similar to:

```text
https://xxxx.trycloudflare.com
```

Share this link with study participants.

---

# 📊 Research Telemetry (Data Collection)

All user interactions are logged locally for post-study statistical analysis.

## Log File Location

```text
research_data/chat_telemetry.csv
```

## Captured Fields

- Timestamp
- Session ID
- User Message
- RAG Trigger Status
- Intent Score
- Bot Response

## Privacy Note

For ethical compliance:

- Session IDs are anonymized browser-generated strings
- No Personally Identifiable Information (PII) is collected
- No data is transmitted to external API providers
- All inference runs 100% locally on the host machine

---

# 🧪 Running Unit Tests

The vector mathematics and RAG logic gates are unit-tested to ensure reproducibility and academic rigor.

```bash
pytest backend/ -v
```

---

# 📚 Research Context

This project was developed as part of a Project investigating:

- Local-first AI systems
- Privacy-preserving health interventions
- Retrieval-Augmented Generation (RAG)
- Motivational Interviewing (MI)
- Lightweight clinical AI deployment on consumer hardware

The system is designed for real-time dietary intervention research while maintaining full local inference and user privacy.

---

# 📄 License

```text
MIT License
```