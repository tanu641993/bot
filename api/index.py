import hashlib
import os
import re
import tempfile
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
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

app = FastAPI(title="Gemini RAG Data API", version="1.0")

# ── Check Environment Variables ───────────────────────────────────────────────
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("Missing GOOGLE_API_KEY environment variable.")

# ── In-Memory Global State (Simulating Session State / Cache) ─────────────────
# Note: For production with multiple users, replace these with Redis or DB storage.
GLOBAL_STATE = {
    "file_hash": None,
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

@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Uploads file, validates extensions, and runs asynchronous vector creation."""
    contents = await file.read()
    current_hash = compute_hash(contents)
    
    # Avoid rebuilding index if it's the exact same file
    if GLOBAL_STATE["file_hash"] == current_hash and GLOBAL_STATE["retriever"] is not None:
        return {"message": "File already uploaded and indexed.", "file_hash": current_hash}
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".csv", ".pdf"]:
        raise HTTPException(status_code=400, detail="Only CSV and PDF files are supported.")

    # Save to safe temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        process_document(tmp_path, ext)
        GLOBAL_STATE["file_hash"] = current_hash
        # Cleanup file after processing
        background_tasks.add_task(clean_temp_file, tmp_path)
    except Exception as e:
        background_tasks.add_task(clean_temp_file, tmp_path)
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")

    return {"message": f"Successfully indexed {file.filename}", "file_hash": current_hash}


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Executes RAG pipeline against the active document context."""
    if not GLOBAL_STATE["retriever"]:
        raise HTTPException(status_code=400, detail="No active document. Please upload a file first.")

    # Fetch context documents
    docs = GLOBAL_STATE["retriever"].get_relevant_documents(request.question)
    if not docs:
        return QueryResponse(
            answer="I could not find relevant information in the uploaded document.", 
            source_chunks=[]
        )

    # Format retrieved document context
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{d.page_content}" for i, d in enumerate(docs)
    )
    
    # Format chat history array (limiting to the last 4 elements)
    history = "\n".join(
        f"User: {h.user}\nAssistant: {h.assistant}"
        for h in request.chat_history[-4:]
    ) or "None"

    prompt = f"""You are a helpful assistant that answers questions strictly from the provided document context.

RULES:
- Answer ONLY from the context below. Do NOT use outside knowledge.
- If the answer is not in the context, say exactly: "The document does not contain information about this."
- Be concise and direct. Quote or reference the relevant chunk when useful.
- Never make up names, numbers, or facts.

--- DOCUMENT CONTEXT START ---
{context}
--- DOCUMENT CONTEXT END ---

Conversation so far:
{history}

Question: {question}

Answer:"""

    answer = llm_invoke(prompt)
    source_chunks = [d.page_content for d in docs]
    
    return QueryResponse(answer=answer, source_chunks=source_chunks)


@app.post("/reset")
async def reset_state():
    """Wipes memory session cache."""
    GLOBAL_STATE["file_hash"] = None
    GLOBAL_STATE["retriever"] = None
    GLOBAL_STATE["prompt_cache"] = {}
    return {"message": "State reset successfully."}
