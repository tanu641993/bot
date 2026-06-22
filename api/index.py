import os
os.environ["GOOGLE_API_VERSION"] = "v1"
os.environ["GOOGLE_API_ENDPOINT"] = "https://generativelanguage.googleapis.com/v1/"

import hashlib
import logging
import re
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use the new google.genai SDK (recommended)
from google import genai
from google.genai import types

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# ── Configuration ──────────────────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_TOKENS = 1024
TOP_K = 6
PROMPT_CACHE_MAX_SIZE = 100

# ── Environment ──────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("Missing GOOGLE_API_KEY")
    raise RuntimeError("Missing GOOGLE_API_KEY")

# ── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Gemini RAG Data API", version="1.0")

# ── Global State ──────────────────────────────────────────────────────────────
GLOBAL_STATE = {
    "file_hash": None,
    "filename": None,
    "retriever": None,
    "prompt_cache": {},
    "all_chunks": [],
}

# ── Custom Gemini Embeddings (using google.genai) ─────────────────────────────
class GeminiEmbeddings:
    def __init__(self, model="gemini-embedding-2"):
        self.client = genai.Client(
            api_key=GOOGLE_API_KEY,
            http_options={'api_version': 'v1'}
        )
        self.model = model

    def embed_documents(self, texts):
        result = []
        for text in texts:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(task_type="retrieval_document"),
            )
            result.append(response.embeddings[0].values)
        return result

    def embed_query(self, text):
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type="retrieval_query"),
        )
        return response.embeddings[0].values

# ── Custom Gemini Chat (using google.genai) ─────────────────────────────────
class ChatGemini:
    def __init__(self, model="gemini-1.5-flash", temperature=0.0, max_output_tokens=1024):
        self.client = genai.Client(
            api_key=GOOGLE_API_KEY,
            http_options={'api_version': 'v1'}   # force v1
        )
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def invoke(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
        )
        return response.text

# ── LLM instance ─────────────────────────────────────────────────────────────
llm = ChatGemini(
    model="gemini-2.5-flash",   # pick one from the list
    temperature=0.0,
    max_output_tokens=MAX_TOKENS,
)

# ── HTML Template ─────────────────────────────────────────────────────────────
TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "index.html"
try:
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        HTML_TEMPLATE = f.read()
except FileNotFoundError:
    HTML_TEMPLATE = "<html><body><h1>Gemini RAG</h1><p>Template missing.</p></body></html>"

# ── Schemas ──────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    user: str
    assistant: str

class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[ChatMessage]] = []

class QueryResponse(BaseModel):
    answer: str
    source_chunks: List[str]

# ── Helpers ──────────────────────────────────────────────────────────────────
def compute_hash(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def prompt_hash(p: str) -> str:
    return hashlib.md5(p.encode()).hexdigest()

def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def llm_invoke(prompt: str) -> str:
    p_hash = prompt_hash(prompt)
    cache = GLOBAL_STATE["prompt_cache"]
    if p_hash in cache:
        return cache[p_hash]
    raw = llm.invoke(prompt)
    result = strip_think_tags(raw)
    if len(cache) >= PROMPT_CACHE_MAX_SIZE:
        oldest_key = next(iter(cache))
        del cache[oldest_key]
    cache[p_hash] = result
    return result

def process_document(file_path: str, ext: str):
    logger.info(f"Processing {file_path} with extension {ext}")
    try:
        if ext == ".csv":
            docs = CSVLoader(file_path=file_path).load()
        elif ext == ".pdf":
            docs = PyPDFLoader(file_path=file_path).load()
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")

        if not docs:
            raise HTTPException(status_code=400, detail="No content extracted from file")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} chunks")

        GLOBAL_STATE["all_chunks"] = [chunk.page_content for chunk in chunks]

        embeddings = GeminiEmbeddings(model="gemini-embedding-2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        GLOBAL_STATE["retriever"] = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        GLOBAL_STATE["file_hash"] = compute_hash(open(file_path, "rb").read())
        GLOBAL_STATE["filename"] = Path(file_path).name
        logger.info(f"Indexed {GLOBAL_STATE['filename']} successfully")
    except Exception as e:
        logger.exception("process_document failed")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home_interface():
    current_file = GLOBAL_STATE["filename"] or "No document indexed yet"
    html = HTML_TEMPLATE.replace("{{ current_file }}", current_file)
    return HTMLResponse(content=html)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    ext = Path(file.filename).suffix.lower()
    if ext not in (".csv", ".pdf"):
        raise HTTPException(status_code=400, detail="Only CSV or PDF allowed")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        process_document(tmp_path, ext)
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

    return {"message": "File uploaded and indexed successfully.", "filename": file.filename}

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if GLOBAL_STATE["retriever"] is None:
        raise HTTPException(status_code=400, detail="No document indexed yet.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        docs = GLOBAL_STATE["retriever"].invoke(request.question)
        if not docs:
            raise HTTPException(status_code=404, detail="No relevant documents found.")
        context = "\n\n".join([doc.page_content for doc in docs])
        source_chunks = [doc.page_content[:200] + "..." for doc in docs]
        prompt = f"Answer using only the context. If not present, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {request.question}\n\nAnswer:"
        answer = llm_invoke(prompt)
        return QueryResponse(answer=answer, source_chunks=source_chunks)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@app.get("/summary")
async def get_summary():
    if not GLOBAL_STATE["all_chunks"]:
        raise HTTPException(status_code=400, detail="No document has been indexed yet.")

    full_text = "\n\n".join(GLOBAL_STATE["all_chunks"])
    if len(full_text) < 12000:
        prompt = f"Summarize the following document concisely:\n\n{full_text}\n\nSummary:"
        summary = llm_invoke(prompt)
        return {"summary": summary}

    # Map-reduce for longer documents
    segment_size = 3000
    overlap = 200
    segments = []
    start = 0
    while start < len(full_text):
        end = start + segment_size
        segments.append(full_text[start:end])
        start = end - overlap

    segment_summaries = []
    for i, seg in enumerate(segments):
        prompt = f"Summarize this part (part {i+1} of {len(segments)}):\n\n{seg}\n\nSummary:"
        seg_summary = llm_invoke(prompt)
        segment_summaries.append(seg_summary)

    combined = "\n\n".join(segment_summaries)
    final_prompt = f"Combine these summaries into one overall summary:\n\n{combined}\n\nOverall summary:"
    final_summary = llm_invoke(final_prompt)
    return {"summary": final_summary}

@app.post("/reset")
async def reset_state():
    GLOBAL_STATE["retriever"] = None
    GLOBAL_STATE["file_hash"] = None
    GLOBAL_STATE["filename"] = None
    GLOBAL_STATE["all_chunks"] = []
    GLOBAL_STATE["prompt_cache"].clear()
    return {"message": "State reset successfully."}

@app.get("/health")
async def health():
    return {"status": "ok", "file_indexed": GLOBAL_STATE["filename"] is not None}

@app.get("/list-generative-models")
async def list_generative_models():
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY, http_options={'api_version': 'v1'})
        models = client.models.list()
        # Just return all model names, let the user pick
        model_names = [m.name for m in models]
        return {"all_models": model_names}
    except Exception as e:
        return {"error": str(e)}