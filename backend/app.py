"""
DocuChat Backend - Flask API
Processes documents and handles chat via LLM (Claude API)
"""

import os
import json
import uuid
import hashlib
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from functools import wraps

from flask import Flask, request, jsonify, session, g
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_client import LLMClient

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = BASE_DIR / "docuchat.db"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt", "md"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App Setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production-" + str(uuid.uuid4()))
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

CORS(app, supports_credentials=True, origins=["http://localhost:3000", "http://127.0.0.1:3000", "null"])

UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(str(DB_PATH), detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    db = sqlite3.connect(str(DB_PATH))
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            original_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER,
            chunk_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'processing',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );
    """)
    db.commit()
    db.close()
    logger.info("Database initialized")

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated

def get_user_upload_dir(user_id):
    d = UPLOAD_DIR / user_id
    d.mkdir(exist_ok=True)
    return d

def get_user_vector_dir(user_id):
    d = VECTOR_STORE_DIR / user_id
    d.mkdir(exist_ok=True)
    return d

# ─── Auth Routes ──────────────────────────────────────────────────────────────

@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    if len(username) < 3 or len(username) > 50:
        return jsonify({"error": "Username must be 3–50 characters"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    db = get_db()
    existing = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        return jsonify({"error": "Username already taken"}), 409

    user_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO users (id, username, password_hash) VALUES (?, ?, ?)",
        (user_id, username, generate_password_hash(password))
    )
    db.commit()

    session["user_id"] = user_id
    session["username"] = username
    return jsonify({"user_id": user_id, "username": username}), 201

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    return jsonify({"user_id": user["id"], "username": user["username"]})

@app.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})

@app.route("/api/auth/me", methods=["GET"])
def me():
    if "user_id" not in session:
        return jsonify({"authenticated": False}), 200
    return jsonify({
        "authenticated": True,
        "user_id": session["user_id"],
        "username": session["username"]
    })

# ─── Document Routes ───────────────────────────────────────────────────────────

@app.route("/api/documents", methods=["GET"])
@require_auth
def list_documents():
    user_id = session["user_id"]
    db = get_db()
    docs = db.execute(
        "SELECT id, original_name, file_type, file_size, chunk_count, status, created_at "
        "FROM documents WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    ).fetchall()
    return jsonify([dict(d) for d in docs])

@app.route("/api/documents/upload", methods=["POST"])
@require_auth
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    user_id = session["user_id"]
    upload_dir = get_user_upload_dir(user_id)
    vector_dir = get_user_vector_dir(user_id)

    original_name = secure_filename(file.filename)
    doc_id = str(uuid.uuid4())
    file_ext = original_name.rsplit(".", 1)[1].lower()
    stored_name = f"{doc_id}.{file_ext}"
    file_path = upload_dir / stored_name

    file.save(str(file_path))
    file_size = file_path.stat().st_size

    db = get_db()
    db.execute(
        "INSERT INTO documents (id, user_id, filename, original_name, file_type, file_size, status) "
        "VALUES (?, ?, ?, ?, ?, ?, 'processing')",
        (doc_id, user_id, stored_name, original_name, file_ext, file_size)
    )
    db.commit()

    # Process asynchronously in thread
    import threading
    t = threading.Thread(target=process_document_async, args=(doc_id, user_id, str(file_path), vector_dir))
    t.daemon = True
    t.start()

    return jsonify({
        "document_id": doc_id,
        "original_name": original_name,
        "status": "processing"
    }), 202

def process_document_async(doc_id, user_id, file_path, vector_dir):
    """Extract text, chunk, embed, and store in vector index."""
    try:
        processor = DocumentProcessor()
        chunks = processor.process(file_path)

        store = VectorStore(str(vector_dir))
        store.add_document(doc_id, chunks)

        db = sqlite3.connect(str(DB_PATH))
        db.execute(
            "UPDATE documents SET status = 'ready', chunk_count = ? WHERE id = ?",
            (len(chunks), doc_id)
        )
        db.commit()
        db.close()
        logger.info(f"Document {doc_id} processed: {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {e}")
        db = sqlite3.connect(str(DB_PATH))
        db.execute("UPDATE documents SET status = 'error' WHERE id = ?", (doc_id,))
        db.commit()
        db.close()

@app.route("/api/documents/<doc_id>", methods=["DELETE"])
@require_auth
def delete_document(doc_id):
    user_id = session["user_id"]
    db = get_db()
    doc = db.execute(
        "SELECT * FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id)
    ).fetchone()
    if not doc:
        return jsonify({"error": "Document not found"}), 404

    # Remove file
    upload_dir = get_user_upload_dir(user_id)
    file_path = upload_dir / doc["filename"]
    if file_path.exists():
        file_path.unlink()

    # Remove from vector store
    vector_dir = get_user_vector_dir(user_id)
    store = VectorStore(str(vector_dir))
    store.remove_document(doc_id)

    db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    db.commit()

    return jsonify({"ok": True})

@app.route("/api/documents/<doc_id>/status", methods=["GET"])
@require_auth
def document_status(doc_id):
    user_id = session["user_id"]
    db = get_db()
    doc = db.execute(
        "SELECT id, status, chunk_count FROM documents WHERE id = ? AND user_id = ?",
        (doc_id, user_id)
    ).fetchone()
    if not doc:
        return jsonify({"error": "Not found"}), 404
    return jsonify(dict(doc))

# ─── Chat Routes ───────────────────────────────────────────────────────────────

@app.route("/api/conversations", methods=["GET"])
@require_auth
def list_conversations():
    user_id = session["user_id"]
    db = get_db()
    convs = db.execute(
        "SELECT id, title, created_at FROM conversations WHERE user_id = ? ORDER BY created_at DESC LIMIT 50",
        (user_id,)
    ).fetchall()
    return jsonify([dict(c) for c in convs])

@app.route("/api/conversations", methods=["POST"])
@require_auth
def create_conversation():
    user_id = session["user_id"]
    conv_id = str(uuid.uuid4())
    db = get_db()
    db.execute(
        "INSERT INTO conversations (id, user_id, title) VALUES (?, ?, ?)",
        (conv_id, user_id, "New conversation")
    )
    db.commit()
    return jsonify({"conversation_id": conv_id}), 201

@app.route("/api/conversations/<conv_id>/messages", methods=["GET"])
@require_auth
def get_messages(conv_id):
    user_id = session["user_id"]
    db = get_db()
    conv = db.execute(
        "SELECT id FROM conversations WHERE id = ? AND user_id = ?", (conv_id, user_id)
    ).fetchone()
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404

    msgs = db.execute(
        "SELECT id, role, content, sources, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at",
        (conv_id,)
    ).fetchall()
    result = []
    for m in msgs:
        d = dict(m)
        d["sources"] = json.loads(d["sources"]) if d["sources"] else []
        result.append(d)
    return jsonify(result)

@app.route("/api/conversations/<conv_id>/chat", methods=["POST"])
@require_auth
def chat(conv_id):
    user_id = session["user_id"]
    data = request.get_json()
    question = (data.get("question") or "").strip()
    doc_ids = data.get("document_ids") or []  # Optional filter by specific docs

    if not question:
        return jsonify({"error": "Question is required"}), 400

    db = get_db()
    conv = db.execute(
        "SELECT id FROM conversations WHERE id = ? AND user_id = ?", (conv_id, user_id)
    ).fetchone()
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404

    # Get conversation history (last 10 messages)
    history = db.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at DESC LIMIT 10",
        (conv_id,)
    ).fetchall()
    history = list(reversed([dict(h) for h in history]))

    # Get user's ready documents
    if doc_ids:
        placeholders = ",".join("?" * len(doc_ids))
        docs = db.execute(
            f"SELECT id, original_name FROM documents WHERE user_id = ? AND id IN ({placeholders}) AND status = 'ready'",
            [user_id] + doc_ids
        ).fetchall()
    else:
        docs = db.execute(
            "SELECT id, original_name FROM documents WHERE user_id = ? AND status = 'ready'",
            (user_id,)
        ).fetchall()

    if not docs:
        return jsonify({
            "answer": "No documents available. Please upload and process documents first before asking questions.",
            "sources": []
        })

    doc_map = {d["id"]: d["original_name"] for d in docs}

    # Retrieve relevant chunks
    vector_dir = get_user_vector_dir(user_id)
    store = VectorStore(str(vector_dir))
    chunks = store.search(question, doc_ids=list(doc_map.keys()), top_k=8)

    if not chunks:
        answer = "I couldn't find relevant information in your documents to answer this question."
        sources = []
    else:
        # Build context with source attribution
        context_parts = []
        for i, chunk in enumerate(chunks):
            doc_name = doc_map.get(chunk["doc_id"], "Unknown document")
            page_info = f", page {chunk['page']}" if chunk.get("page") else ""
            context_parts.append(
                f"[SOURCE {i+1}: {doc_name}{page_info}]\n{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Call LLM
        llm = LLMClient()
        answer, sources = llm.answer(question, context, history, chunks, doc_map)

    # Save messages
    msg_user_id = str(uuid.uuid4())
    msg_bot_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, 'user', ?)",
        (msg_user_id, conv_id, question)
    )
    db.execute(
        "INSERT INTO messages (id, conversation_id, role, content, sources) VALUES (?, ?, 'assistant', ?, ?)",
        (msg_bot_id, conv_id, answer, json.dumps(sources))
    )

    # Update conversation title from first user message
    first_msg = db.execute(
        "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ?", (conv_id,)
    ).fetchone()
    if first_msg["cnt"] <= 2:
        title = question[:60] + ("..." if len(question) > 60 else "")
        db.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))

    db.commit()

    return jsonify({
        "answer": answer,
        "sources": sources,
        "message_id": msg_bot_id
    })

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)