import base64
import hashlib
import json
import re
import tempfile
import os

import pandas as pd
import streamlit as st
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS  
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI # Swapped to Gemini
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Configuration ──────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
MAX_TOKENS    = 1024
TOP_K         = 6

st.set_page_config(page_title="Gemini RAG Data Chatbot", page_icon="🧠", layout="wide")
st.title("📊 CSV / Excel / PDF Chatbot with Google Gemini")
st.write("Upload a file, then ask questions. Type **show dashboard** to build an exportable HTML dashboard.")

# Fallback checking for Gemini Environment Variable on Vercel
if not os.environ.get("GOOGLE_API_KEY"):
    st.warning("Please configure your GOOGLE_API_KEY in Vercel's Environment Variables dashboard.")

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
    # Uses Gemini 1.5 Flash (fast, smart, and includes a generous free tier)
    st.session_state.llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0,
        max_output_tokens=MAX_TOKENS,
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
    raw = st.session_state.llm.invoke(prompt)
    result = strip_think_tags(str(raw.content))
    set_cache(prompt, result)
    return result

# ── RAG builders (Using Free Gemini Embeddings) ──────────────────────────────────
@st.cache_resource
def build_retriever_csv(file_path: str):
    docs   = CSVLoader(file_path=file_path).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)
    # Native Gemini text embeddings
    vs = FAISS.from_documents(chunks, GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"))
    return vs.as_retriever(search_kwargs={"k": TOP_K})

@st.cache_resource
def build_retriever_pdf(file_path: str):
    docs   = PyPDFLoader(file_path).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)
    vs = FAISS.from_documents(chunks, GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"))
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

# [Rest of your Pandas logic and dashboard generation remains identical]
