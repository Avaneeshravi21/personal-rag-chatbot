"""
FastAPI backend for the RAG chatbot.

Why FastAPI instead of just keeping the CLI script: a terminal script
only YOU can use, on YOUR machine, one message at a time. An API turns
the same logic into something a web frontend (or mobile app, or another
program) can call over HTTP -- this is the actual difference between
"a script I wrote" and "an application I built," which matters for how
this reads on a resume and in an interview.

Run with:
    uvicorn api.main:app --reload --port 8000

Then visit http://localhost:8000/docs for an automatic interactive API
explorer (FastAPI generates this for free from the code below -- a nice
thing to demo live in an interview).
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.pipeline import RAGPipeline

app = FastAPI(title="Personal RAG Chatbot API")

# CORS lets a frontend running on a different port/domain (e.g. Streamlit
# on port 8501) actually call this API from a browser. Without this,
# browsers block the request by default as a security measure.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for a personal project this is fine; a real
                            # product would restrict this to its own domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded ONCE when the server starts, not per-request -- loading the
# embedding model and reranker takes real time, so we don't want to pay
# that cost on every single API call.
pipeline: RAGPipeline | None = None


@app.on_event("startup")
def load_pipeline():
    global pipeline
    pipeline = RAGPipeline()


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    standalone_query: str
    sub_queries: list[str]


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = pipeline.ask(request.session_id, request.message)
    return ChatResponse(**result)


@app.get("/health")
def health():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}
