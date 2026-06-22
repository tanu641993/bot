import hashlib
import os
import re
import tempfile
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd

from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS  
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Configuration ──────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
MAX_TOKENS    = 1024
TOP_K         = 6

# Look at the left margin - it must have ZERO spaces before it!
app = FastAPI(title="Gemini RAG Data API", version="1.0")


# ── Check Environment Variables ───────────────────────────────────────────────
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("Missing GOOGLE_API_KEY environment variable.")

# ── In-Memory Global State (Simulating Session State / Cache) ─────────────────
GLOBAL_STATE = {
    "file_hash": None,
    "filename": None,
    "retriever": None,
    "prompt_cache": {},
}

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.0,
    max_output_tokens=MAX_TOKENS,
)

# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    user: str
    assistant: str

class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[ChatMessage]] = []

class QueryResponse(BaseModel):
    answer: str
    source_chunks: List[str]

# ── Helpers ────────────────────────────────────────────────────────────────────
def compute_hash(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def prompt_hash(p: str) -> str:
    return hashlib.md5(p.encode()).hexdigest()

def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def llm_invoke(prompt: str) -> str:
    p_hash = prompt_hash(prompt)
    if p_hash in GLOBAL_STATE["prompt_cache"]:
        return GLOBAL_STATE["prompt_cache"][p_hash]
    
    raw = llm.invoke(prompt)
    result = strip_think_tags(str(raw.content))
    GLOBAL_STATE["prompt_cache"][p_hash] = result
    return result

def clean_temp_file(path: str):
    """Safely removes temporary uploaded files."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

# ── Vector Engine Processing ──────────────────────────────────────────────────
def process_document(file_path: str, ext: str):
    """Parses document chunks and builds the FAISS index."""
    if ext == ".csv":
        docs = CSVLoader(file_path=file_path).load()
    elif ext == ".pdf":
        docs = PyPDFLoader(file_path=file_path).load()
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vs = FAISS.from_documents(chunks, embeddings)
    GLOBAL_STATE["retriever"] = vs.as_retriever(search_kwargs={"k": TOP_K})

# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home_interface():
    """Renders a fully interactive HTML Chat UI directly on your root homepage."""
    current_file = GLOBAL_STATE["filename"] or "No document indexed yet"
    return f"""
    <!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gemini RAG Assistant</title>
    <style>
        body {
                {
                font-family: 'Segoe UI', system-ui, sans-serif;
                background-color: #f3f4f6;
                margin: 0;
                padding: 20px;
                display: flex;
                justify-content: center;
            }
        }

        .app-card {
                {
                width: 100%;
                max-width: 650px;
                background: white;
                border-radius: 16px;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
                padding: 30px;
                box-sizing: border-box;
            }
        }

        h2 {
                {
                margin-top: 0;
                color: #1e3a8a;
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 24px;
            }
        }

        .file-box {
                {
                background: #f8fafc;
                border: 2px dashed #cbd5e1;
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                margin-bottom: 20px;
                transition: border 0.2s;
            }
        }

        .file-box:hover {
                {
                border-color: #3b82f6;
            }
        }

        .status-badge {
                {
                display: inline-block;
                background: #e0f2fe;
                color: #0369a1;
                padding: 4px 10px;
                border-radius: 20px;
                font-size: 13px;
                font-weight: 600;
                margin-top: 10px;
            }
        }

        input[type="file"] {
                {
                display: none;
            }
        }

        .upload-btn {
                {
                background: #3b82f6;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                display: inline-block;
                margin-top: 5px;
            }
        }

        .upload-btn:hover {
                {
                background: #2563eb;
            }
        }

        #chat-window {
                {
                height: 350px;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                background: #fafafa;
                overflow-y: auto;
                padding: 15px;
                margin-bottom: 15px;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }
        }

        .bubble {
                {
                max-width: 80%;
                padding: 10px 14px;
                border-radius: 12px;
                font-size: 15px;
                line-height: 1.4;
                word-wrap: break-word;
            }
        }

        .bubble.user {
                {
                background: #2563eb;
                color: white;
                align-self: flex-end;
                border-bottom-right-radius: 2px;
            }
        }

        .bubble.bot {
                {
                background: #e2e8f0;
                color: #1e293b;
                align-self: flex-start;
                border-bottom-left-radius: 2px;
            }
        }

        .chat-controls {
                {
                display: flex;
                gap: 10px;
            }
        }

        .chat-controls input {
                {
                flex: 1;
                padding: 12px;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                font-size: 15px;
                outline: none;
            }
        }

        .chat-controls input:focus {
                {
                border-color: #3b82f6;
                box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
            }
        }

        .chat-controls button {
                {
                background: #1e3a8a;
                color: white;
                border: none;
                padding: 0 22px;
                border-radius: 8px;
                font-weight: bold;
                cursor: pointer;
            }
        }

        .chat-controls button:hover {
                {
                background: #172554;
            }
        }

        .reset-link {
                {
                display: block;
                text-align: center;
                color: #94a3b8;
                font-size: 13px;
                margin-top: 15px;
                text-decoration: none;
                cursor: pointer;
            }
        }

        .reset-link:hover {
                {
                color: #ef4444;
            }
        }
    </style>
</head>

<body>
    <div class="app-card">
        <h2>&#129504; Gemini RAG Data Assistant</h2>

        <div class="file-box">
            <label class="upload-btn" for="file-picker">Choose CSV / PDF</label>
            <input type="file" id="file-picker" accept=".csv,.pdf" onchange="handleFileUpload()">
            <div id="filename-display" style="font-size: 14px; margin-top: 8px; color: #64748b;">Ready to process file</div>
            <div><span class="status-badge" id="active-badge">{current_file}</span></div>
        </div>

        <div id="chat-window">
            <div class="bubble bot">Hello! Upload a data document above, and I'll answer your targeted queries using Gemini context retrieval.</div>
        </div>

        <div class="chat-controls">
            <input type="text" id="query-field" placeholder="Ask a question about the document..." onkeydown="if(event.key === 'Enter') sendQuery()">
            <button onclick="sendQuery()">Ask</button>
        </div>

        <span class="reset-link" onclick="resetApp()">Wipe vector state session cache</span>
    </div>

    <script>
        let localHistory = [];

        async function handleFileUpload() {
            {
                const picker = document.getElementById('file-picker');
                if (!picker.files.length) return;

                const file = picker.files[0];
                document.getElementById('filename-display').innerText = `Uploading: ${{file.name}}...`;

                const formData = new FormData();
                formData.append('file', file);

                try {
                    {
                        const response = await fetch('/upload', {
                            {
                                method: 'POST',
                                body: formData
                            }
                        });
                        const data = await response.json();
                        if (response.ok) {
                            {
                                document.getElementById('active-badge').innerText = file.name;
                                document.getElementById('filename-display').innerText = "Processing finished successfully!";
                                appendBubble("bot", `Successfully processed and vector indexed **${{file.name}}**. Let me know what you would like to find out!`);
                            }
                        } else {
                            {
                                throw new Error(data.detail || "Upload error");
                            }
                        }
                    }
                } catch (err) {
                    {
                        document.getElementById('filename-display').innerText = "Upload sequence broke down.";
                        alert(err.message);
                    }
                }
            }
        }

        async function sendQuery() {
                {
                    const field = document.getElementById('query-field');
                    const text = field.value.trim();
                    if (!text) return;

                    field.value = '';
                    appendBubble("user", text);

                    try {
                        {
                            const response = await fetch('/query', {
                                {
                                    method: 'POST',
                                    headers: {
                                        {
                                            'Content-Type': 'application/json'
                                        }
                                    },
                                    body: JSON.stringify({
                                        {
                                            question: text,
                                            chat_history: localHistory
                                        }
                                    })
                                }
                            });
                            const data = await response.json();

                            if (response.ok) {
                                {
                                    appendBubble("bot", data.answer);
                                    localHistory.push({
                                        {
                                            user: text,
                                            assistant: data.answer
                                        }
                                    });
                                }
                            }
    </script>
</body>

</html>
<html>
<p>
    """</p>

</html>