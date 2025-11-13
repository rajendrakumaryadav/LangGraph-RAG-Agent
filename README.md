# LangGraph RAG Agent 🚀

> An advanced Retrieval-Augmented Generation (RAG) system built with LangChain, LangGraph, FastAPI, and Streamlit

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [System Workflow](#system-workflow)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Core Components](#core-components)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## 🎯 Overview

**LangGraph RAG Agent** is a production-ready, enterprise-grade Retrieval-Augmented Generation system that combines cutting-edge AI technologies to enable intelligent document understanding and question-answering capabilities. The system leverages:

- **LangChain** for building LLM applications with composable components
- **LangGraph** for creating stateful, multi-step workflows
- **Qdrant** as a high-performance vector database
- **Ollama** for running local LLMs (Generator and Embeddings)
- **Groq** for high-speed critic model inference
- **FastAPI** for scalable REST API endpoints
- **Streamlit** for an intuitive web interface

### What is RAG?

Retrieval-Augmented Generation (RAG) is an AI framework that enhances large language models by retrieving relevant information from external knowledge bases before generating responses. This approach:

- ✅ Reduces hallucinations by grounding responses in factual data
- ✅ Enables LLMs to access up-to-date, domain-specific information
- ✅ Provides source attribution for generated answers
- ✅ Allows working with private documents and proprietary data

### Why This Project?

This implementation goes beyond basic RAG systems by incorporating:

1. **Multi-Stage Retrieval**: Uses advanced retrieval techniques including:
   - Maximum Marginal Relevance (MMR) for diverse results
   - Multi-Query Retrieval for comprehensive coverage
   - Contextual Compression to reduce noise

2. **Quality Assurance**: Implements a critique node that fact-checks generated answers against source documents

3. **Production-Ready Design**: Includes both API and UI interfaces with proper state management

4. **Flexible Architecture**: Easy to swap models, retrievers, and vector stores

---

## ✨ Key Features

### Document Management
- 📄 **Multi-Format Support**: Ingest PDF, DOCX, and TXT files
- 🔄 **Intelligent Chunking**: Uses RecursiveCharacterTextSplitter with configurable chunk size (1000) and overlap (200)
- 🗂️ **Batch Processing**: Efficient document ingestion with batch size optimization (32 documents per batch)
- 📚 **Metadata Preservation**: Maintains source information for attribution

### Advanced Retrieval
- 🔍 **Semantic Search**: Vector-based similarity search using embeddings
- 🎯 **MMR Search**: Maximum Marginal Relevance for diverse, non-redundant results
- 🔄 **Multi-Query**: Automatically generates multiple query variations for better recall
- 📉 **Contextual Compression**: Filters retrieved documents to only relevant content
- ⚡ **Configurable Parameters**: Fetch k=20, return k=8 for optimal balance

### Answer Generation
- 🤖 **LLM-Powered**: Uses Ollama models (default: gemma3:1b) for fast local inference
- 🔒 **Context-Aware**: Generates answers strictly based on retrieved documents
- 🎨 **Customizable Prompts**: Easy to modify system prompts for specific use cases
- ❌ **Honest Responses**: Explicitly states when information is not available

### Quality Control
- ✅ **Fact-Checking**: Critic model (Groq-powered) validates generated answers
- 🔄 **Auto-Revision**: Automatically revises answers that don't meet quality standards
- 📝 **Source Attribution**: Every answer includes source document references
- 🎯 **Accuracy Focus**: Reduces hallucinations and improves reliability

### Deployment Options
- 🌐 **REST API**: FastAPI with async support and automatic OpenAPI documentation
- 💬 **Web UI**: Streamlit chat interface with document upload capabilities
- 🐳 **Docker Support**: Complete containerized deployment with docker-compose
- 📊 **Production-Ready**: Includes proper error handling, logging, and state management

---

## 🏗️ Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│  ┌──────────────────────┐    ┌──────────────────────────┐      │
│  │   Streamlit UI       │    │   FastAPI REST API       │      │
│  │  (Port: 8501)        │    │   (Port: 8000)           │      │
│  └──────────────────────┘    └──────────────────────────┘      │
└────────────────────────┬──────────────────┬────────────────────┘
                         │                  │
                         ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              LangGraph RAG Workflow                       │  │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────────┐       │  │
│  │  │ Retrieve │───→│ Generate │───→│   Critique   │       │  │
│  │  │  Node    │    │   Node   │    │     Node     │       │  │
│  │  └──────────┘    └──────────┘    └──────────────┘       │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Retrievers  │  │   Models     │  │   Ingest     │        │
│  │  (MMR, MQ,   │  │  (Generator, │  │  (Docling,   │        │
│  │  Compress)   │  │  Critic,     │  │  Splitter)   │        │
│  └──────────────┘  │  Embeddings) │  └──────────────┘        │
│                    └──────────────┘                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Infrastructure Layer                         │
│  ┌──────────────────────┐    ┌──────────────────────────┐      │
│  │   Qdrant Vector DB   │    │   Ollama LLM Server      │      │
│  │   (Port: 6333/6334)  │    │   (Port: 11434)          │      │
│  │   - Vector Storage   │    │   - Generator Model      │      │
│  │   - Similarity Search│    │   - Embedding Model      │      │
│  └──────────────────────┘    └──────────────────────────┘      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐      │
│  │   Groq API (Cloud)                                    │      │
│  │   - Critic Model (openai/gpt-oss-20b)                │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Step 1: RETRIEVE                                        │
│ - Convert query to embedding vector                     │
│ - Multi-Query: Generate 3-5 query variations           │
│ - MMR Search: Find top 20 similar chunks (fetch_k=20)  │
│ - Return 8 most diverse results (k=8)                   │
│ Output: List[Document] with metadata                    │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: COMPRESS                                        │
│ - Extract only relevant portions from each document     │
│ - Remove redundant context                              │
│ - Preserve essential information                        │
│ Output: Compressed List[Document]                       │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: GENERATE                                        │
│ - Format documents as context                           │
│ - Apply RAG prompt template                             │
│ - Generate answer using LLM (Ollama)                    │
│ - Temperature: 0 for consistency                        │
│ Output: Generated answer string                         │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ Step 4: CRITIQUE                                        │
│ - Fact-check answer against source documents           │
│ - Critic model evaluates accuracy (Groq)                │
│ - Decision: "accept" or "revise"                        │
│ - If revise: Return corrected answer                    │
│ - Add source attribution                                │
│ Output: Final answer with sources                       │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
           Final Response
```

---

## 🔄 System Workflow

### 1. Document Ingestion Workflow

```python
# Process: apps/streamlit-app.py → core/ingest.py

Upload Files (PDF/DOCX/TXT)
    │
    ├─→ DoclingLoader.load()
    │   └─→ Extracts text with structure preservation
    │
    ├─→ RecursiveCharacterTextSplitter
    │   ├─ chunk_size: 1000 characters
    │   ├─ chunk_overlap: 200 characters
    │   └─→ Creates overlapping chunks for context continuity
    │
    ├─→ OllamaEmbeddings.embed_documents()
    │   └─→ Convert chunks to 1024-dim vectors (bge-m3)
    │
    └─→ QdrantVectorStore.add_documents()
        └─→ Store vectors with metadata in Qdrant collection
```

**Key Parameters:**
- Chunk Size: 1000 (balances context and precision)
- Overlap: 200 (maintains continuity between chunks)
- Batch Size: 32 (optimizes ingestion performance)

### 2. Query Processing Workflow

```python
# Process: core/rag.py (LangGraph State Machine)

User Question
    │
    ▼
[GraphState: question]
    │
    ├─→ NODE: retrieve_documents()
    │   ├─ Embed query → vector
    │   ├─ MultiQueryRetriever: Generate query variations
    │   ├─ MMR Search: fetch_k=20, return k=8
    │   └─ ContextualCompression: Filter relevant content
    │
    ├─→ [GraphState: question, documents]
    │
    ├─→ NODE: generate_answer()
    │   ├─ Format documents as context
    │   ├─ Apply RAG prompt template
    │   ├─ LLM.invoke() → ChatOllama (gemma3:1b)
    │   └─ Generate answer
    │
    ├─→ [GraphState: question, documents, generation]
    │
    ├─→ NODE: critique_answer()
    │   ├─ Format critique prompt
    │   ├─ Critic LLM.invoke() → ChatGroq (gpt-oss-20b)
    │   ├─ Parse JSON response: {"decision": "accept|revise", "revision": "..."}
    │   ├─ If revise: Use revision as final answer
    │   └─ Append source attribution
    │
    └─→ [GraphState: question, documents, generation, final_answer]
```

---

## 📦 Prerequisites

### System Requirements
- **OS**: Linux, macOS, or Windows (with WSL2)
- **RAM**: Minimum 8GB (16GB recommended for optimal performance)
- **Storage**: 10GB free space (for models and vector data)
- **Python**: 3.12 or higher

### Required Services
- **Docker & Docker Compose** (for containerized deployment)
- **Ollama** (for local LLM inference)
- **Qdrant** (vector database)

### API Keys
- **Groq API Key** (for critic model): Get from [Groq Console](https://console.groq.com/)

---

## 🚀 Installation

### Option 1: Docker Compose (Recommended)

This is the fastest way to get started with all services running:

```bash
# 1. Clone the repository
git clone https://github.com/rajendrakumaryadav/LangGraph-RAG-Agent.git
cd LangGraph-RAG-Agent

# 2. Create .env file with your configuration
cat > .env << EOF
# Models
GEN_MODEL=gemma3:1b
EMBED_MODEL=bge-m3

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=twodocs

# Groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_CRITIC_MODEL=openai/gpt-oss-20b
EOF

# 3. Start all services (Ollama + Qdrant)
docker-compose up -d

# 4. Pull required Ollama models
docker exec -it ollama ollama pull gemma3:1b
docker exec -it ollama ollama pull bge-m3

# 5. Install Python dependencies
pip install uv  # Fast Python package installer
uv sync         # Installs from pyproject.toml

# Verify services are running
curl http://localhost:11434/api/version  # Ollama
curl http://localhost:6333/collections   # Qdrant
```

### Option 2: Local Installation

If you prefer to run services locally without Docker:

```bash
# 1. Clone repository
git clone https://github.com/rajendrakumaryadav/LangGraph-RAG-Agent.git
cd LangGraph-RAG-Agent

# 2. Install Python dependencies
pip install uv
uv sync

# 3. Install and start Ollama
# macOS/Linux:
curl -fsSL https://ollama.com/install.sh | sh
ollama serve

# Pull required models
ollama pull gemma3:1b
ollama pull bge-m3

# 4. Install and start Qdrant
# Using Docker:
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant

# Or install locally: https://qdrant.tech/documentation/guides/installation/

# 5. Configure environment
cp .env.example .env
# Edit .env with your settings
```

### Option 3: Development Setup

For active development with hot reload:

```bash
# 1. Clone and navigate
git clone https://github.com/rajendrakumaryadav/LangGraph-RAG-Agent.git
cd LangGraph-RAG-Agent

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install in editable mode
pip install -e .

# 4. Start services
docker-compose up -d

# 5. Configure environment
cp .env.example .env
nano .env  # Edit configuration
```

---

## ⚙️ Configuration

### Environment Variables

The system is configured via the `.env` file. Here's a detailed explanation of each setting:

```bash
# ============================================
# LLM Configuration
# ============================================

# Generator Model: Used for answer generation
# Options: Any Ollama model (gemma3:1b, llama3.2, mistral, etc.)
# Recommendation: gemma3:1b for speed, llama3.2:3b for quality
GEN_MODEL=gemma3:1b

# Embedding Model: Converts text to vectors
# Options: bge-m3, nomic-embed-text, mxbai-embed-large
# Recommendation: bge-m3 (multilingual, high quality)
EMBED_MODEL=bge-m3

# Critic Model: Fact-checks generated answers
# Must be a Groq model for fast inference
GROQ_CRITIC_MODEL=openai/gpt-oss-20b
GROQ_API_KEY=your_groq_api_key_here

# ============================================
# Vector Database Configuration
# ============================================

# Qdrant URL: Connection string to Qdrant server
# Local: http://localhost:6333
# Docker: http://qdrant:6333 (if app also in Docker)
QDRANT_URL=http://localhost:6333

# Qdrant API Key: Leave empty for local development
QDRANT_API_KEY=

# Collection Name: Where vectors are stored
# Use different collections for different document sets
QDRANT_COLLECTION=twodocs
```

### Model Selection Guide

| Use Case | Generator Model | Embedding Model | Notes |
|----------|----------------|-----------------|-------|
| **Development** | gemma3:1b | bge-m3 | Fast, low RAM |
| **Production** | llama3.2:3b | bge-m3 | Better quality |
| **Multilingual** | gemma3:1b | bge-m3 | bge-m3 supports 100+ languages |
| **High Quality** | llama3.2:8b | nomic-embed-text | Requires 16GB RAM |
| **Code Documents** | codellama:13b | bge-m3 | Specialized for code |

### Advanced Configuration

Modify `core/settings.py` for advanced tuning:

```python
# Retrieval parameters
RETRIEVAL_K = 8           # Number of documents to return
RETRIEVAL_FETCH_K = 20    # Number of documents to fetch before MMR
MMR_LAMBDA = 0.5          # Diversity parameter (0=max diversity, 1=max relevance)

# Chunking parameters
CHUNK_SIZE = 1000         # Characters per chunk
CHUNK_OVERLAP = 200       # Overlap between chunks

# Batch processing
BATCH_SIZE = 32           # Documents per ingestion batch
```

---

## 💻 Usage

### 1. Streamlit Web UI (Recommended for Most Users)

The Streamlit interface provides an intuitive chat experience:

```bash
# Start the Streamlit app
streamlit run apps/streamlit-app.py

# Access at: http://localhost:8501
```

**Features:**
- 📤 **Drag-and-drop document upload** (sidebar)
- 💬 **Chat interface** for asking questions
- 📜 **Conversation history** maintained during session
- 🔗 **Source attribution** for every answer
- ⚡ **Real-time processing** with progress indicators

**Usage Steps:**
1. Click "Browse files" in the sidebar
2. Upload PDF, DOCX, or TXT files
3. Click "Ingest Documents"
4. Wait for "✅ Ingestion complete!" message
5. Type your question in the chat input
6. View answer with source references

### 2. FastAPI REST API (For Programmatic Access)

The REST API is ideal for integrations and automation:

```bash
# Start the FastAPI server
uvicorn apps.fastapi-apps:app --reload --host 0.0.0.0 --port 8000

# Access API docs at: http://localhost:8000/docs
```

**Endpoints:**

#### `GET /`
Health check endpoint.

```bash
curl http://localhost:8000/
```

Response:
```json
{
  "message": "Advanced RAG API is running. Use the /ask endpoint to post questions."
}
```

#### `POST /ask`
Submit a question and receive an answer.

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the documents?"}'
```

Response:
```json
{
  "answer": "The main topic discusses...\n\n**Sources:**\n- document1.pdf\n- document2.docx"
}
```

**Python Client Example:**

```python
import requests

API_URL = "http://localhost:8000"

def ask_question(question: str) -> str:
    response = requests.post(
        f"{API_URL}/ask",
        json={"question": question}
    )
    response.raise_for_status()
    return response.json()["answer"]

# Usage
answer = ask_question("What are the key findings?")
print(answer)
```

**JavaScript Client Example:**

```javascript
async function askQuestion(question) {
  const response = await fetch('http://localhost:8000/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });
  const data = await response.json();
  return data.answer;
}

// Usage
const answer = await askQuestion('What are the key findings?');
console.log(answer);
```

### 3. Docker Compose (Full Stack Deployment)

Run all services together in containers:

```bash
# Start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Stop and remove volumes (clears all data)
docker-compose down -v
```

**Services Running:**
- Ollama: http://localhost:11434
- Qdrant: http://localhost:6333 (HTTP), http://localhost:6334 (gRPC)

### 4. Programmatic Usage (Python)

Use the RAG system directly in your Python code:

```python
import sys
sys.path.append('/path/to/LangGraph-RAG-Agent')

from core.ingest import ingest_paths
from core.rag import create_rag_workflow

# 1. Ingest documents
document_paths = [
    "/path/to/document1.pdf",
    "/path/to/document2.docx"
]
ingest_paths(document_paths)

# 2. Create RAG workflow
rag_app = create_rag_workflow()

# 3. Ask questions
result = rag_app.invoke({"question": "What is the summary?"})
answer = result.get("final_answer")
print(answer)

# 4. Access intermediate steps
documents = result.get("documents")  # Retrieved documents
generation = result.get("generation")  # Initial answer before critique
```

---

## 🧩 Core Components

### 1. `core/ingest.py` - Document Ingestion

**Purpose**: Load, process, and store documents in the vector database.

**Key Functions:**

```python
def ingest_paths(paths: List[str]) -> None:
    """
    Ingests documents from file paths into Qdrant.
    
    Process:
    1. Load documents using DoclingLoader
    2. Split into chunks (1000 chars, 200 overlap)
    3. Generate embeddings
    4. Store in Qdrant vector database
    
    Args:
        paths: List of file paths (PDF, DOCX, TXT)
    """
```

**Features:**
- ✅ Robust file parsing with Docling
- ✅ Error handling per document (continues on failure)
- ✅ Batch processing for efficiency
- ✅ Progress logging

**Usage Example:**
```python
from core.ingest import ingest_paths

# Ingest single document
ingest_paths(["/path/to/document.pdf"])

# Ingest multiple documents
ingest_paths([
    "/data/report1.pdf",
    "/data/report2.docx",
    "/data/notes.txt"
])
```

### 2. `core/rag.py` - RAG Workflow Orchestration

**Purpose**: Implements the LangGraph state machine for RAG pipeline.

**State Definition:**
```python
class GraphState(TypedDict):
    question: str           # User's question
    documents: List[Document]  # Retrieved documents
    generation: str         # Initial LLM answer
    final_answer: str       # Critiqued and revised answer
```

**Nodes:**

#### `retrieve_documents(state: GraphState) -> GraphState`
- Embeds the question
- Retrieves top-k relevant documents
- Uses MMR, Multi-Query, and Compression

#### `generate_answer(state: GraphState) -> GraphState`
- Formats documents as context
- Applies RAG prompt template
- Generates answer with LLM

#### `critique_answer(state: GraphState) -> GraphState`
- Fact-checks the generated answer
- Requests revision if needed
- Adds source attribution

**Workflow Creation:**
```python
from core.rag import create_rag_workflow

app = create_rag_workflow()
result = app.invoke({"question": "What is...?"})
```

### 3. `core/models.py` - LLM and Embedding Models

**Purpose**: Centralized model management for easy swapping.

**Functions:**

```python
def get_embedding_model() -> OllamaEmbeddings:
    """Returns the embedding model for vectorization."""
    
def get_generator_model() -> ChatOllama:
    """Returns the generator LLM for answer creation."""
    
def get_critic_model() -> ChatGroq:
    """Returns the critic LLM for fact-checking."""
```

**Customization:**
```python
# Modify core/models.py to use different models

# Option 1: Use OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def get_generator_model():
    return ChatOpenAI(model="gpt-4", temperature=0)

# Option 2: Use Anthropic
from langchain_anthropic import ChatAnthropic

def get_generator_model():
    return ChatAnthropic(model="claude-3-sonnet-20240229")
```

### 4. `core/retriever.py` - Advanced Retrieval

**Purpose**: Implements sophisticated retrieval strategies.

**Retrieval Pipeline:**

```python
def create_retriever(vector_store) -> ContextualCompressionRetriever:
    """
    Creates a multi-stage retriever:
    
    1. Base MMR Retriever
       - search_type: "mmr"
       - k: 8 (final results)
       - fetch_k: 20 (candidates)
       
    2. Multi-Query Retriever
       - Generates 3-5 query variations
       - Retrieves for each variation
       - Combines and deduplicates results
       
    3. Contextual Compression
       - Extracts only relevant portions
       - Reduces noise
       - Improves generation quality
    """
```

**Retrieval Strategies Explained:**

- **MMR (Maximum Marginal Relevance)**: Balances relevance and diversity
  - Prevents redundant results
  - Lambda=0.5: 50% relevance, 50% diversity

- **Multi-Query**: Generates multiple perspectives
  - "How does X work?" → "Explain X", "X mechanism", "X functionality"
  - Increases recall

- **Contextual Compression**: Filters documents
  - Removes irrelevant sentences
  - Keeps only parts that answer the question

### 5. `core/stores.py` - Vector Store Management

**Purpose**: Manages Qdrant vector database operations.

**Functions:**

```python
def get_qdrant_client() -> qdrant_client.QdrantClient:
    """Returns authenticated Qdrant client."""

def get_vector_store(embeddings) -> QdrantVectorStore:
    """
    Returns LangChain Qdrant vector store.
    
    Features:
    - Auto-creates collection if missing
    - Detects embedding dimensions automatically
    - Uses COSINE distance metric
    """
```

**Collection Management:**
```python
from core.stores import get_qdrant_client

client = get_qdrant_client()

# List collections
collections = client.get_collections()

# Delete collection (clears all documents)
client.delete_collection("collection_name")

# Get collection info
info = client.get_collection("collection_name")
print(f"Vectors: {info.vectors_count}")
```

### 6. `core/settings.py` - Configuration Management

**Purpose**: Loads and validates environment variables.

**Settings:**
```python
GROQ_API_KEY: str              # Groq API key for critic
QDRANT_URL: str                # Qdrant connection URL
QDRANT_COLLECTION_NAME: str    # Collection name
EMBEDDING_MODEL: str           # Ollama embedding model
GENERATOR_MODEL: str           # Ollama generator model
CRITIC_MODEL: str              # Groq critic model
```

### 7. `apps/streamlit-app.py` - Web UI

**Purpose**: Interactive Streamlit interface for end-users.

**Features:**
- 📤 Document upload with drag-and-drop
- 💬 Chat interface with message history
- 📊 Progress indicators during processing
- 🎨 Markdown rendering for formatted answers
- 🔗 Source attribution display

**State Management:**
```python
if "rag_app" not in st.session_state:
    st.session_state.rag_app = create_rag_workflow()

if "messages" not in st.session_state:
    st.session_state.messages = []
```

### 8. `apps/fastapi-apps.py` - REST API

**Purpose**: Production-grade REST API with FastAPI.

**Features:**
- ⚡ Async support for concurrent requests
- 📖 Auto-generated OpenAPI docs
- 🛡️ Request validation with Pydantic
- 🔄 Lifecycle management (startup/shutdown)
- ❌ Proper error handling

**Lifecycle:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load RAG workflow
    state["rag_app"] = create_rag_workflow()
    yield
    # Shutdown: Clean up resources
    state.clear()
```

---

## 📚 API Reference

### FastAPI Endpoints

#### Health Check
```http
GET /
```

**Response:**
```json
{
  "message": "Advanced RAG API is running. Use the /ask endpoint to post questions."
}
```

#### Ask Question
```http
POST /ask
Content-Type: application/json

{
  "question": "string"
}
```

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| question | string | Yes | The question to ask |

**Response:**
```json
{
  "answer": "string"
}
```

**Status Codes:**
- `200`: Success
- `500`: Internal server error
- `503`: RAG workflow not available

**Example with cURL:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main findings in the research?"
  }'
```

**Example with Python:**
```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/ask",
        json={"question": "What are the key points?"}
    )
    data = response.json()
    print(data["answer"])
```

### Streamlit Interface

The Streamlit app doesn't have a formal API but provides:

**Session State Variables:**
- `st.session_state.rag_app`: The LangGraph workflow instance
- `st.session_state.messages`: List of chat messages

**File Upload:**
- Supported formats: PDF, DOCX, TXT
- Multiple files: Yes
- Max size: Unlimited (constrained by RAM)

---

## 🚀 Deployment

### Production Deployment Checklist

- [ ] Set strong API keys in `.env`
- [ ] Use production-grade Qdrant (persistent storage)
- [ ] Configure reverse proxy (Nginx/Traefik)
- [ ] Enable HTTPS with SSL certificates
- [ ] Set up monitoring and logging
- [ ] Configure CORS for API
- [ ] Implement rate limiting
- [ ] Set up health checks
- [ ] Configure auto-restart on failure
- [ ] Implement backup strategy for vector DB

### Docker Compose Production

```yaml
# docker-compose.prod.yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    restart: always
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        limits:
          memory: 8G

  qdrant:
    image: qdrant/qdrant:latest
    restart: always
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT__STORAGE__STORAGE_PATH: /qdrant/storage
      QDRANT__SERVICE__ENABLE_TLS: "true"

  fastapi:
    build: .
    restart: always
    ports:
      - "8000:8000"
    environment:
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - ollama
      - qdrant
    command: uvicorn apps.fastapi-apps:app --host 0.0.0.0 --port 8000

volumes:
  ollama_data:
  qdrant_data:
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: fastapi
        image: your-registry/rag-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: QDRANT_URL
          value: "http://qdrant-service:6333"
        - name: GROQ_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: groq-api-key
```

### Cloud Platforms

#### AWS Deployment
- **Compute**: ECS/Fargate for containers
- **Vector DB**: Qdrant Cloud or self-hosted on EC2
- **Storage**: S3 for document storage
- **API Gateway**: API Gateway + Lambda (serverless)

#### GCP Deployment
- **Compute**: Cloud Run for containers
- **Vector DB**: Qdrant Cloud or GKE
- **Storage**: Cloud Storage for documents
- **Load Balancer**: Cloud Load Balancing

#### Azure Deployment
- **Compute**: Container Apps
- **Vector DB**: Qdrant Cloud or AKS
- **Storage**: Blob Storage for documents
- **API Gateway**: API Management

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Ollama Connection Error

**Error:**
```
requests.exceptions.ConnectionError: HTTPConnectionPool(host='localhost', port=11434)
```

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# If not running, start it
docker-compose up -d ollama
# OR
ollama serve

# Pull required models
ollama pull gemma3:1b
ollama pull bge-m3
```

#### 2. Qdrant Connection Error

**Error:**
```
qdrant_client.exceptions.UnexpectedResponse: Unexpected Response: 404
```

**Solution:**
```bash
# Check if Qdrant is running
curl http://localhost:6333/collections

# Start Qdrant
docker-compose up -d qdrant

# Verify collection exists
curl http://localhost:6333/collections/twodocs
```

#### 3. Groq API Key Error

**Error:**
```
ValueError: GROQ_API_KEY environment variable not set
```

**Solution:**
```bash
# Set API key in .env file
echo "GROQ_API_KEY=your_actual_key_here" >> .env

# Or export temporarily
export GROQ_API_KEY="your_actual_key_here"
```

#### 4. Out of Memory Error

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Use smaller models
# In .env:
GEN_MODEL=gemma3:1b  # Instead of llama3.2:8b
EMBED_MODEL=bge-m3   # Instead of larger models

# Reduce batch size in core/ingest.py:
vector_store.add_documents(chunks, batch_size=16)  # Default: 32
```

#### 5. Slow Retrieval Performance

**Symptoms**: Queries take > 5 seconds

**Solution:**
```python
# In core/retriever.py, simplify retrieval:

def create_retriever(vector_store):
    # Remove compression for speed
    base_retriever = vector_store.as_retriever(
        search_type="similarity",  # Instead of "mmr"
        search_kwargs={'k': 4}     # Reduce k
    )
    return base_retriever
```

#### 6. Document Ingestion Fails

**Error:**
```
Failed to load document.pdf: ...
```

**Solution:**
```bash
# Check file permissions
chmod +r document.pdf

# Verify file is not corrupted
file document.pdf

# Try with a simpler file format first
# Convert to TXT and test
```

#### 7. Empty or Irrelevant Answers

**Symptoms**: Answers are "I don't know" or off-topic

**Solution:**
```python
# 1. Check if documents were ingested
from core.stores import get_qdrant_client

client = get_qdrant_client()
info = client.get_collection("twodocs")
print(f"Documents in DB: {info.vectors_count}")

# 2. Test retrieval directly
from core.retriever import create_retriever
from core.stores import get_vector_store
from core.models import get_embedding_model

embeddings = get_embedding_model()
vector_store = get_vector_store(embeddings)
retriever = create_retriever(vector_store)

docs = retriever.invoke("your question")
print(f"Retrieved {len(docs)} documents")
for doc in docs:
    print(doc.page_content[:200])
```

### Debug Mode

Enable detailed logging:

```python
# Add to apps/streamlit-app.py or apps/fastapi-apps.py

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# In RAG workflow
logger.debug(f"Retrieved documents: {len(documents)}")
logger.debug(f"Generated answer: {generation[:100]}")
```

### Performance Tuning

| Issue | Solution |
|-------|----------|
| Slow ingestion | Increase `batch_size` to 64 |
| High memory usage | Reduce `fetch_k` to 10 |
| Slow queries | Disable contextual compression |
| Poor answer quality | Increase `k` to 10-12 |
| Redundant results | Use MMR with lambda=0.3 |

---

## 🎯 Best Practices

### Document Preparation
1. **Clean your documents**: Remove headers, footers, page numbers
2. **Use consistent formatting**: Helps the chunking algorithm
3. **Break large documents**: Split PDFs > 100 pages
4. **Include metadata**: Filename should be descriptive

### Query Formulation
- ✅ **Good**: "What are the key findings in the 2023 report?"
- ❌ **Bad**: "Tell me stuff"
- ✅ **Good**: "Explain the methodology used in section 3"
- ❌ **Bad**: "How"

### System Configuration
1. **Development**: Use `gemma3:1b` for fast iteration
2. **Production**: Use `llama3.2:3b` or better for quality
3. **Always**: Keep `temperature=0` for consistency
4. **Monitor**: Vector count in Qdrant regularly

### Security
- 🔒 Never commit `.env` to version control
- 🔒 Use environment-specific API keys
- 🔒 Implement authentication on FastAPI in production
- 🔒 Sanitize user inputs to prevent injection attacks
- 🔒 Use HTTPS in production deployments

### Scalability
- 📊 Use Qdrant Cloud for large datasets (> 1M vectors)
- 📊 Implement caching for frequent queries
- 📊 Use async processing for document ingestion
- 📊 Load balance multiple API instances
- 📊 Monitor and set rate limits

### Monitoring
Track these metrics in production:
- Query latency (p50, p95, p99)
- Vector count in Qdrant
- LLM token usage (cost tracking)
- Error rate and types
- Memory and CPU usage

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/LangGraph-RAG-Agent.git
cd LangGraph-RAG-Agent

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dev dependencies
pip install -e ".[dev]"

# 4. Create feature branch
git checkout -b feature/your-feature-name
```

### Code Style
- Use **Black** for formatting: `black .`
- Use **isort** for imports: `isort .`
- Follow **PEP 8** guidelines
- Add **type hints** to all functions
- Write **docstrings** for public APIs

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add support for CSV files
fix: resolve Qdrant connection timeout
docs: update installation instructions
refactor: simplify retriever creation
test: add unit tests for ingest module
```

### Pull Request Process
1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Create PR with clear description

### Areas for Contribution
- 🆕 Add support for more document formats (CSV, HTML, Markdown)
- 🎨 Improve UI/UX of Streamlit app
- 📊 Add monitoring and observability
- 🧪 Increase test coverage
- 🌐 Add internationalization (i18n)
- 📝 Improve documentation and examples
- ⚡ Performance optimizations
- 🔧 Add configuration presets

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ℹ️ License and copyright notice required

---

## 🙏 Acknowledgements

This project builds upon excellent open-source technologies:

### Core Technologies
- **[LangChain](https://github.com/langchain-ai/langchain)** - Framework for LLM applications
- **[LangGraph](https://github.com/langchain-ai/langgraph)** - Build stateful, multi-actor applications with LLMs
- **[Qdrant](https://qdrant.tech/)** - High-performance vector database
- **[Ollama](https://ollama.ai/)** - Run large language models locally
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern, fast web framework
- **[Streamlit](https://streamlit.io/)** - Fast way to build data apps

### AI Providers
- **[Groq](https://groq.com/)** - Ultra-fast LLM inference
- **[Docling](https://github.com/DS4SD/docling)** - Advanced document understanding

### Inspiration
- LangChain tutorials and examples
- RAG research papers and implementations
- Open-source RAG frameworks

---

## 📞 Support

### Getting Help
- 📖 **Documentation**: Read this README thoroughly
- 💬 **Issues**: [GitHub Issues](https://github.com/rajendrakumaryadav/LangGraph-RAG-Agent/issues)
- 📧 **Email**: rajendra.ecti@gmail.com
- 🌟 **Star this repo** if you find it useful!

### Reporting Issues
When reporting issues, please include:
1. Operating system and Python version
2. Exact error message and stack trace
3. Steps to reproduce
4. Relevant configuration (`.env` without secrets)
5. Logs from services (Ollama, Qdrant)

### Feature Requests
We'd love to hear your ideas! Please:
1. Search existing issues first
2. Describe the use case clearly
3. Explain why it would be valuable
4. Suggest implementation approach (optional)

---

## 🔮 Roadmap

### Version 0.2.0 (Next Release)
- [ ] Add support for CSV and Excel files
- [ ] Implement user authentication
- [ ] Add conversation memory (multi-turn)
- [ ] Web scraping capability
- [ ] Export chat history

### Version 0.3.0 (Future)
- [ ] Multi-language support in UI
- [ ] Advanced analytics dashboard
- [ ] Document comparison feature
- [ ] Hybrid search (keyword + semantic)
- [ ] Custom prompt templates UI

### Version 1.0.0 (Stable)
- [ ] Production-hardened deployment
- [ ] Comprehensive test suite
- [ ] Monitoring and alerting
- [ ] Multi-tenancy support
- [ ] Enterprise features

---

## 📊 Project Stats

![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

---

**Built with ❤️ by [Rajendra Kumar Yadav](https://github.com/rajendrakumaryadav)**

If you find this project helpful, please consider giving it a ⭐ star on GitHub!

