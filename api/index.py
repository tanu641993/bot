import hashlib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# LangChain & vector store
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# ── Configuration ──────────────────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_TOKENS = 1024
TOP_K = 6
PROMPT_CACHE_MAX_SIZE = 100  # limit cache to avoid unbounded growth

# ── Check Environment Variables (early) ──────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("Missing GOOGLE_API_KEY environment variable.")
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Gemini RAG Data API",
    version="1.0",
    description="Upload CSV/PDF and query with Gemini + RAG",
)

# ── CORS (optional – uncomment if needed) ────────────────────────────────────
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# ── Global State ──────────────────────────────────────────────────────────────
GLOBAL_STATE = {
    "file_hash": None,
    "filename": None,
    "retriever": None,
    "prompt_cache": {},
}

# ── Initialize Gemini LLM ─────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.0,
    max_output_tokens=MAX_TOKENS,
)

# ── Load HTML template once at startup (caching) ─────────────────────────────
TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "index.html"
try:
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        HTML_TEMPLATE = f.read()
except FileNotFoundError:
    logger.warning("templates/index.html not found – using fallback HTML.")
    HTML_TEMPLATE = "<html><body><h1>Gemini RAG</h1><p>Template missing.</p></body></html>"

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
    """Invoke Gemini with caching. Returns cleaned response."""
    p_hash = prompt_hash(prompt)
    cache = GLOBAL_STATE["prompt_cache"]
    if p_hash in cache:
        logger.debug(f"Cache hit for prompt hash {p_hash[:8]}")
        return cache[p_hash]

    raw = llm.invoke(prompt)
    result = strip_think_tags(str(raw.content))

    # Manage cache size
    if len(cache) >= PROMPT_CACHE_MAX_SIZE:
        # Remove oldest entry (first inserted)
        oldest_key = next(iter(cache))
        del cache[oldest_key]
        logger.debug(f"Cache evicted oldest key {oldest_key[:8]}")

    cache[p_hash] = result
    logger.debug(f"Cached new response for hash {p_hash[:8]}")
    return result

def process_document(file_path: str, ext: str):
    """
    Parses the document, splits into chunks, builds FAISS index.
    Raises HTTPException on failure.
    """
    try:
        if ext == ".csv":
            docs = CSVLoader(file_path=file_path).load()
        elif ext == ".pdf":
            docs = PyPDFLoader(file_path=file_path).load()
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        GLOBAL_STATE["retriever"] = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        GLOBAL_STATE["file_hash"] = compute_hash(open(file_path, "rb").read())
        GLOBAL_STATE["filename"] = Path(file_path).name
        logger.info(f"Successfully indexed {len(chunks)} chunks from {GLOBAL_STATE['filename']}")

    except Exception as e:
        logger.exception("Document processing failed")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home_interface():
    """
    Serve the main chat interface.
    Dynamically inject the current filename into the template.
    """
    current_file = GLOBAL_STATE["filename"] or "No document indexed yet"
    html = HTML_TEMPLATE.replace("{{ current_file }}", current_file)
    return HTMLResponse(content=html)

@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a CSV or PDF file, process it in the background.
    Returns immediately with a status, while indexing runs asynchronously.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in (".csv", ".pdf"):
        raise HTTPException(status_code=400, detail="Only CSV or PDF files are allowed.")

    # Save uploaded file to a temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Process in background to keep response fast
        background_tasks.add_task(process_document, tmp_path, ext)

        # Also schedule cleanup of the temp file after processing
        # We can't rely on BackgroundTasks to run after, so we'll use a try/finally in process_document
        # But we can also schedule deletion after a delay, or rely on OS temp cleanup.
        # To be safe, we delete the temp file after processing (see process_document)
        # However, process_document doesn't delete it now. We'll add cleanup.

        # Actually, we need to delete it after processing. Let's modify process_document to accept cleanup.
        # But for simplicity, we'll just use a separate background task for cleanup, but that's tricky.
        # Better: In the background task, we process then delete.
        # Let's redefine process_document to handle its own cleanup.
        # I'll show an alternative below.

        return {
            "message": "File uploaded. Indexing started in the background.",
            "filename": file.filename,
        }
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

# We need a version of process_document that deletes the temp file.
# Let's redefine it here (override the previous one) but keep the core logic.
# I'll rename it to process_and_cleanup and use it in background_tasks.

def process_and_cleanup(file_path: str, ext: str):
    """
    Wraps process_document and ensures the temporary file is removed.
    """
    try:
        process_document(file_path, ext)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        # re-raise to be caught by the background task? We'll just log.
    finally:
        try:
            os.unlink(file_path)
            logger.debug(f"Deleted temporary file {file_path}")
        except Exception as e:
            logger.warning(f"Could not delete temp file {file_path}: {e}")

# Now update the upload endpoint to use this new function
@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    ext = Path(file.filename).suffix.lower()
    if ext not in (".csv", ".pdf"):
        raise HTTPException(status_code=400, detail="Only CSV or PDF files are allowed.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        background_tasks.add_task(process_and_cleanup, tmp_path, ext)
        return {
            "message": "File uploaded. Indexing started in the background.",
            "filename": file.filename,
        }
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Accept a question and optional chat history.
    Retrieves relevant chunks and generates an answer using Gemini.
    """
    if GLOBAL_STATE["retriever"] is None:
        raise HTTPException(
            status_code=400,
            detail="No document has been indexed yet. Please upload a file first."
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Retrieve relevant documents
        docs = GLOBAL_STATE["retriever"].invoke(request.question)
        if not docs:
            raise HTTPException(status_code=404, detail="No relevant documents found.")

        # Prepare context
        context = "\n\n".join([doc.page_content for doc in docs])
        source_chunks = [doc.page_content[:200] + "..." for doc in docs]  # preview

        # Build prompt (you can customize this)
        prompt = f"""
        You are a helpful assistant. Answer the user's question using only the context provided.
        If the answer is not in the context, say "I don't have that information."

        Context:
        {context}

        Question: {request.question}

        Answer:
        """

        # Invoke LLM with caching
        answer = llm_invoke(prompt)

        return QueryResponse(answer=answer, source_chunks=source_chunks)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@app.post("/reset")
async def reset_state():
    """
    Reset the in‑memory state (clear vector index and cache).
    """
    GLOBAL_STATE["retriever"] = None
    GLOBAL_STATE["file_hash"] = None
    GLOBAL_STATE["filename"] = None
    GLOBAL_STATE["prompt_cache"].clear()
    logger.info("State reset")
    return {"message": "State reset successfully."}

# ── Health Check (optional) ──────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "file_indexed": GLOBAL_STATE["filename"] is not None}