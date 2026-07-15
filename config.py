"""
Central configuration for the RAG chatbot.
Loads secrets from .env and exposes tunable knobs in one place
so you can log/experiment with them for your eval report.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
RAW_DATA_DIR = "data/raw"          # drop your PDFs / docx / md / txt here
PROCESSED_DATA_DIR = "data/processed"  # chunked + embedded output cache

# --- Collections (think of these as separate "knowledge sources") ---
COLLECTIONS = {
    "personal_docs": {
        "path": os.path.join(RAW_DATA_DIR, "personal_docs"),
        "description": "Your resume, notes, personal PDFs/docs",
    },
    "domain_corpus": {
        "path": os.path.join(RAW_DATA_DIR, "domain_corpus"),
        "description": "arXiv papers on LLMs / RAG / Transformers",
    },
    "conversation_memory": {
        "path": None,  # populated dynamically at runtime, not from files
        "description": "Summarized long-term conversation memory",
    },
}

# --- Chunking ---
CHUNK_SIZE_TOKENS = 400        # try 256 vs 400 vs 512, log recall@k for each
CHUNK_OVERLAP_TOKENS = 60
CHUNK_STRATEGY = "recursive"   # "recursive" | "semantic" | "fixed"

# --- Embeddings ---
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"   # open-source, 1024-dim
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 32

# --- Vector store (Qdrant) ---
# USE_LOCAL_QDRANT=True runs Qdrant embedded (no Docker, no server) —
# recommended for 8GB RAM laptops and for Colab/Jupyter, where you
# either can't run Docker or don't want the extra RAM overhead.
USE_LOCAL_QDRANT = os.getenv("USE_LOCAL_QDRANT", "true").lower() == "true"
QDRANT_LOCAL_PATH = "data/qdrant_local"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)  # only needed for Qdrant Cloud

# --- Hybrid search ---
DENSE_WEIGHT = 0.6   # weight for vector similarity score
SPARSE_WEIGHT = 0.4  # weight for BM25 score

# --- Reranking ---
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANK_TOP_K = 20     # how many candidates to pull before reranking
FINAL_TOP_K = 5        # how many chunks actually go into the LLM context

# --- LLM ---
LLM_PROVIDER = "groq"          # "anthropic" | "openai" | "groq" (groq is free)
LLM_MODEL = "llama-3.1-8b-instant"  # good free Groq model; swap to "claude-sonnet-5" + LLM_PROVIDER="anthropic" if you fund API credits later
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Memory ---
MEMORY_BUFFER_TURNS = 6         # last N raw turns kept verbatim
MEMORY_SUMMARIZE_AFTER = 10      # summarize + store to vector memory after N turns
