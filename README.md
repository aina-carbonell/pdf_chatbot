# DocuChat — Document-Grounded AI Assistant

A self-hosted chatbot that answers questions **exclusively** from your uploaded documents, with source citations, multi-user auth, and persistent storage.

---

## Architecture

```
docuchat/
├── backend/
│   ├── app.py                  # Flask REST API
│   ├── document_processor.py   # PDF/DOCX/TXT text extraction & chunking
│   ├── vector_store.py         # TF-IDF / semantic search index
│   ├── llm_client.py           # Claude / OpenAI / Ollama wrapper
│   └── requirements.txt
├── frontend/
│   └── index.html              # Single-file SPA (HTML + CSS + JS)
├── .env.example
└── README.md
```

**Data flow:**
1. User uploads a document → backend extracts text → chunks it → stores in per-user vector index
2. User asks a question → relevant chunks retrieved → sent to LLM with strict grounding prompt → answer with citations returned

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- An API key from Anthropic, OpenAI, or a local Ollama install

### 2. Backend Setup

```bash
cd docuchat/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: semantic search (much better retrieval)
pip install sentence-transformers numpy
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API key
nano .env
```

### 4. Run the Backend

```bash
cd backend

# Development
python app.py

# Production (gunicorn)
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

The API runs on `http://localhost:5000`

### 5. Open the Frontend

Simply open `frontend/index.html` in your browser. It talks to the backend at `/api`.

**Option A: Direct file open (for local dev)**
```bash
open frontend/index.html
```
> Note: For file:// protocol, you may need to disable CORS or serve via a local server (see below).

**Option B: Serve via Python (recommended)**
```bash
# From project root
python -m http.server 3000 --directory frontend
# Open http://localhost:3000
```

**Option C: Have Flask serve the frontend**

Add to `app.py`:
```python
from flask import send_from_directory

@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend', 'index.html')
```

---

## Configuration

### LLM Providers

| Provider | `LLM_PROVIDER` | Key Env Var | Default Model |
|----------|---------------|-------------|---------------|
| Anthropic Claude | `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-4-5-20251001` |
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Ollama (local) | `ollama` | — | `llama3.2` |

Switch by editing `.env`:
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o
```

### Semantic Search (Optional but Recommended)

Install and the system automatically switches from keyword to semantic search:
```bash
pip install sentence-transformers numpy
```
Model `all-MiniLM-L6-v2` (~80MB) is downloaded on first use. Improves retrieval quality significantly for long or complex documents.

---

## API Reference

### Auth
| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| POST | `/api/auth/register` | `{username, password}` | Create account |
| POST | `/api/auth/login` | `{username, password}` | Sign in (session cookie) |
| POST | `/api/auth/logout` | — | Sign out |
| GET | `/api/auth/me` | — | Check session |

### Documents
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/documents` | List user's documents |
| POST | `/api/documents/upload` | Upload file (multipart) |
| GET | `/api/documents/:id/status` | Poll processing status |
| DELETE | `/api/documents/:id` | Remove document |

### Chat
| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| GET | `/api/conversations` | — | List conversations |
| POST | `/api/conversations` | — | New conversation |
| GET | `/api/conversations/:id/messages` | — | Message history |
| POST | `/api/conversations/:id/chat` | `{question, document_ids?}` | Ask question |

---

## Supported File Formats

| Format | Library | Notes |
|--------|---------|-------|
| PDF | pdfplumber | Text-based PDFs; scanned PDFs need OCR (install pytesseract) |
| DOCX/DOC | python-docx | Full text + structure extraction |
| TXT | built-in | Plain text, any encoding |
| Markdown | built-in | .md files |

**Add OCR for scanned PDFs:**
```bash
pip install pytesseract pillow
brew install tesseract  # macOS
# apt-get install tesseract-ocr  # Linux
```

---

## Anti-Hallucination Design

The system is designed to prevent the LLM from making up answers:

1. **Strict system prompt**: The LLM is explicitly instructed to only answer from provided context
2. **Context injection**: Only the most relevant chunks are passed — not the full document
3. **Mandatory citations**: Every factual claim must cite `[Source: filename, page N]`
4. **Fallback message**: If no relevant chunks found, the LLM returns a structured "not found" response
5. **No internet access**: The LLM cannot search the web

---

## Production Deployment

### Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY backend/ .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### Persistent Storage

By default, data is stored locally in the backend directory:
- `docuchat.db` — SQLite database (users, docs metadata, conversations)
- `uploads/` — Raw uploaded files (per user)
- `vector_store/` — Search indices (per user)

For production, consider:
- **Database**: PostgreSQL (`pip install flask-sqlalchemy psycopg2`)
- **File storage**: S3 / GCS (`pip install boto3`)
- **Vector store**: ChromaDB or Qdrant for better scalability

### Upgrade to ChromaDB (optional)

```bash
pip install chromadb
```

Replace `VectorStore` import in `app.py` with a ChromaDB-backed version for production-grade similarity search.

---

## Test Cases

### 1. Student — Study Notes
- Upload lecture PDFs
- Ask: *"Summarize the key concepts of chapter 3"*
- Ask: *"What are the differences between X and Y according to my notes?"*
- Ask: *"Generate 5 exam questions based on this material"*

### 2. Enterprise — Internal Docs
- Upload company policies, SOPs
- Ask: *"What is the vacation policy?"*
- Ask: *"What are the steps to onboard a new vendor?"*

### 3. Research — Multi-Document Analysis
- Upload multiple papers/reports
- Ask: *"Compare how each document addresses topic X"*
- Ask: *"What conclusions do all documents agree on?"*

---

## Troubleshooting

**Q: LLM returns "Error contacting LLM"**
→ Check your API key in `.env`. Verify the key has credits.

**Q: Document stuck in "processing"**
→ Check backend logs (`python app.py`). Ensure pdfplumber/python-docx are installed.

**Q: Answers are generic, not from my docs**
→ The document may not have processed correctly. Check `chunk_count` in the document list. If 0, the file may be image-based — use OCR.

**Q: CORS error in browser**
→ Make sure you're accessing the frontend from `http://localhost:3000` (not `file://`). Or add your origin to the CORS config in `app.py`.