import base64
import hashlib
import json
import re
import tempfile
import os

import pandas as pd
import streamlit as st
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS  # Swapped Chroma for cloud-safe FAISS in-memory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI # Swapped Ollama for Cloud API wrappers
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Configuration ──────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
MAX_TOKENS    = 1024
TOP_K         = 6

st.set_page_config(page_title="DeepSeek-R1 Data Chatbot", page_icon="🧠", layout="wide")
st.title("📊 CSV / Excel / PDF Chatbot with DeepSeek Cloud")
st.write("Upload a file, then ask questions. Type **show dashboard** to build an exportable HTML dashboard.")

# Fallback checking for Environment Variables on Vercel
if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("DEEPSEEK_API_KEY"):
    st.warning("Please configure your API keys in Vercel's Environment Variables dashboard.")

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("chat_history", []),
    ("file_hash",    None),
    ("retriever",    None),
    ("prompt_cache", {}),
    ("pdf_path",     None),
    ("df",           None),
    ("pdf_text",     None),
    ("file_ext",     None),
    ("dashboard_html", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if "llm" not in st.session_state:
    # Uses standard OpenAI or DeepSeek cloud endpoints securely via environment vars
    st.session_state.llm = ChatOpenAI(
        model="gpt-4o-mini", # change to "deepseek-chat" or "deepseek-reasoner" if using DeepSeek API
        temperature=0.0,
        max_tokens=MAX_TOKENS,
    )

# ── Helpers ────────────────────────────────────────────────────────────────────
def compute_hash(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def prompt_hash(p: str) -> str:
    return hashlib.md5(p.encode()).hexdigest()

def get_cache(p):
    return st.session_state.prompt_cache.get(prompt_hash(p))

def set_cache(p, v):
    st.session_state.prompt_cache[prompt_hash(p)] = v

def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def llm_invoke(prompt: str) -> str:
    cached = get_cache(prompt)
    if cached:
        return cached
    # Cloud ChatOpenAI uses .invoke() and returns an AIMessage object
    raw = st.session_state.llm.invoke(prompt)
    result = strip_think_tags(str(raw.content))
    set_cache(prompt, result)
    return result

# ── RAG builders (Optimized for Serverless Memory) ─────────────────────────────
@st.cache_resource
def build_retriever_csv(file_path: str):
    docs   = CSVLoader(file_path=file_path).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)
    # Using cloud embeddings to fix memory/CPU constraints on serverless host
    vs = FAISS.from_documents(chunks, OpenAIEmbeddings())
    return vs.as_retriever(search_kwargs={"k": TOP_K})

@st.cache_resource
def build_retriever_pdf(file_path: str):
    docs   = PyPDFLoader(file_path).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)
    vs = FAISS.from_documents(chunks, OpenAIEmbeddings())
    return vs.as_retriever(search_kwargs={"k": TOP_K})

# ── Core RAG query ─────────────────────────────────────────────────────────────
def run_rag_query(question: str) -> tuple[str, list]:
    docs = st.session_state.retriever.get_relevant_documents(question)
    if not docs:
        return "I could not find relevant information in the uploaded document.", []

    context = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{d.page_content}" for i, d in enumerate(docs)
    )
    history = "\n".join(
        f"User: {h['user']}\nAssistant: {h['assistant']}"
        for h in st.session_state.chat_history[-4:]
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
    return answer, docs

# ── Pandas fast-path ───────────────────────────────────────────────────────────
def try_pandas_answer(question: str, df: pd.DataFrame):
    q = question.lower()
    if "how many rows" in q or "row count" in q or "total records" in q:
        return f"The dataset has **{len(df):,} rows**."
    if "how many columns" in q or "column count" in q:
        return f"The dataset has **{len(df.columns)} columns**: {', '.join(df.columns)}."
    if "missing" in q or "null" in q:
        m = df.isnull().sum()
        m = m[m > 0]
        if m.empty:
            return "There are **no missing values** in the dataset."
        return "Missing values per column:\n" + "\n".join(f"- {c}: {v}" for c, v in m.items())
    if "duplicate" in q:
        return f"There are **{int(df.duplicated().sum())} duplicate rows**."
    for word in ["average", "mean"]:
        if word in q:
            for col in df.select_dtypes(include="number").columns:
                if col.lower() in q:
                    return f"The average of **{col}** is **{df[col].mean():.4f}**."
    for word, fn in [("maximum","max"),("minimum","min"),("max","max"),("min","min")]:
        if word in q:
            for col in df.select_dtypes(include="number").columns:
                if col.lower() in q:
                    return f"The {fn} of **{col}** is **{getattr(df[col], fn)()}**."
    if "unique" in q or "distinct" in q:
        for col in df.columns:
            if col.lower() in q:
                return f"**{col}** has **{df[col].nunique()} unique values**."
    return None

# [Rest of your Dashboard builders / Streamlit UI Logic remaining the same...]
