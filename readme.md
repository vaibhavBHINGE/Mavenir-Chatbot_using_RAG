# 3GPP Technical Specification AI Assistant & Knowledge Retrieval System

A production-grade, zero-hallucination Retrieval-Augmented Generation (RAG) and Agentic system engineered to query, analyze, and synthesize insights from massive telecommunications standards documents (specifically the **1,510-page 3GPP TS 23.501** specification).

Built for enterprise telecom environments, this platform bridges raw technical documentation with intelligent agentic orchestration, ensuring strict adherence to source materials while providing an intuitive, high-performance user experience.

---

## 🏗️ System Architecture & File Structure

The project features a decoupled, modular architecture that cleanly separates document embedding, backend intelligence tools, API routing, and the frontend user interface:

```text
MV/
│
├── Backend/
│   ├── __init__.py
│   ├── main.py            # FastAPI entry point, CORS middleware, and API router endpoints
│   ├── agents.py          # Custom ReAct agent executor, LLM integration, and intent handling
│   ├── tools.py           # LangChain custom tools (3GPP Vector Search & Human Escalation)
│   └── rag.py             # ChromaDB vector store wrapper, document chunking, and embeddings
│
├── frontend/
│   └── app.py             # Streamlit interactive UI (Dark mode, session states, chat history)
│
├── data/
│   └── 23501-j80.docx     # Primary 3GPP Technical Specification (1,510 Pages)
│
├── sessions/              # Persistent JSON chat message histories per session ID
├── vector_db/             # Local Chroma vector database storage directory
├── test_agent.py          # Standalone verification script for agent pipeline
├── test_tools.py          # Standalone verification script for RAG vector tools
├── requirements.txt       # Project Python dependencies
└── README.md              # Project documentation
```

---

## 📂 Detailed Component Breakdown

### 1. Knowledge Base & RAG Pipeline (`Backend/rag.py`)

**Document Ingestion:** Parses the monolithic 1,510-page Word document (`23501-j80.docx`).

**Chunking & Storage:** Splits complex specifications into granular, semantically meaningful chunks with overlapping windows to preserve cross-reference context. Embeddings are stored locally in ChromaDB for sub-second similarity matching.

### 2. Custom Tools (`Backend/tools.py`)

**`search_3gpp_standards`:** An enterprise retrieval tool that takes natural language prompts, queries the Chroma vector store, and extracts the most relevant specification clauses paired with source section headers.

**`escalate_to_human`:** A guardrail tool that dynamically generates formal support tickets (e.g., TKT-10492) whenever user prompts fall outside document scope or lack matching data.

### 3. Agentic Intelligence & ReAct Engine (`Backend/agents.py`)

**Custom ReAct Loop:** Implements a robust Reasoning-Acting (ReAct) decision cycle designed to bypass framework wrapper bugs.

**Hugging Face Inference:** Integrates with Mistral-7B-Instruct-v0.3 using the `huggingface_hub` client, handling prompt formatting via native `[INST]` tags and managing multi-turn session history via `FileChatMessageHistory`.

### 4. API Backend (`Backend/main.py`)

**FastAPI Server:** Exposes asynchronous REST endpoints (`/chat`, `/health`) handling CORS middleware, payload validation, and robust exception management.

### 5. Interactive Frontend (`frontend/app.py`)

**Streamlit Interface:** Styled with a clean enterprise dark-mode theme, active knowledge base status indicators, session reset controls, and real-time streaming chat responses.

---

## 🛡️ How We Eliminated Hallucinations (Anti-Hallucination Strategy)

Large Language Models are inherently prone to hallucinations—a critical failure point in telecommunications engineering where a single incorrect protocol parameter or architecture reference point can break network design. To guarantee 100% factual accuracy, this system enforces a Multi-Layered Guardrail Architecture:

### Strict Context Binding (Zero-Guessing System Prompt)

The prompt structure explicitly forbids the model from drawing upon its internal parametric memory for technical questions, restricting it entirely to the retrieved context chunks:

> "You are a professional 3GPP Telecom Expert. Read the technical specification context below and write a clear, polished, structured explanation answering the user's question. Do not output raw chunks or metadata tags; synthesize the explanation naturally."

### Mandatory Tool Gatekeeping

The agent checks user intent, automatically routing conversational queries to polite dialogues and restricting technical inquiries strictly to vector search output.

### Deterministic Local Fallback Mode

If external cloud inference APIs experience network drops, routing mismatches, or rate limits, the system seamlessly triggers a deterministic local fallback mechanism. This strips raw metadata headers and safely surfaces exact specification sections without generating unverified synthetic text.

### Automated Out-of-Scope Escalation

Non-telecom queries (such as general trivia, recipes, or weather) are intercepted by safety guards and routed directly to the `escalate_to_human` ticketing tool, preserving strict domain boundaries.

---

## 🚀 Installation & Local Setup

### 1. Clone & Navigate to Project Root

```bash
cd C:\Users\vbhin\OneDrive\Desktop\MV
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root directory and add your Hugging Face API token:

```env
HUGGINGFACEHUB_API_TOKEN=hf_your_actual_token_here
```

---

## 💡 Running the Application

### Step 1: Start the FastAPI Backend

Run Uvicorn specifying single-worker execution to ensure optimal process stability:

```bash
uvicorn Backend.main:app --reload --workers 1
```

### Step 2: Launch the Streamlit Frontend

Open a separate terminal window, navigate to the project root, and launch the UI:

```bash
streamlit run frontend/app.py
```

Open your browser at http://localhost:8501 to interact with your 3GPP AI Assistant!
