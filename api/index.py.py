import hashlib
import os
import re
import tempfile
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
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
async def home_interface():
    # If you need to pass the current file name, you can do it with a simple replace
    html = open("templates/index.html", "r", encoding="utf-8").read()
    # Replace a placeholder in the HTML if you want dynamic content
    # For example, in your HTML, put {{ current_file }} and then do:
    # html = html.replace("{{ current_file }}", GLOBAL_STATE["filename"] or "No document indexed yet")
    return HTMLResponse(content=html)