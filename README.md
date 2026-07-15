---
title: Personal RAG Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.36.0"
app_file: app.py
pinned: false
---
🔗 **[Try the live demo here](https://your-actual-streamlit-url.streamlit.app)**

# Personal RAG Chatbot — Week 1: Ingestion & Retrieval

A multi-source RAG assistant over (1) your personal documents, (2) a domain
corpus of LLM/RAG/Transformer papers, and (3) long-term conversation memory.

## Week 1 scope
Ingestion pipeline: load → chunk → embed → store, plus a script to sanity-check
retrieval before moving on to hybrid search + reranking (Week 2).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# fill in ANTHROPIC_API_KEY / OPENAI_API_KEY (needed later, in Week 3)
```

### Vector store: embedded mode (default, no Docker needed)
By default (`USE_LOCAL_QDRANT=true` in config.py), Qdrant runs embedded —
in-process, storing data at `data/qdrant_local/`. No Docker, no server,
lighter on RAM. This is the recommended mode for 8GB RAM machines and
for running in Colab/Jupyter.

### Optional: run Qdrant as a Docker server instead
Only do this if you want the Qdrant web dashboard, or plan to scale beyond
a laptop later. Set `USE_LOCAL_QDRANT=False` in your `.env`, then:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Add your data
```
data/raw/personal_docs/     <- drop your resume, notes, PDFs, docx here
data/raw/domain_corpus/     <- drop ~15-20 arXiv papers on LLMs/RAG/Transformers
```
Good starter papers for domain_corpus: "Attention Is All You Need",
"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
"REALM", "RETRO", "Dense Passage Retrieval", "Lost in the Middle".

### Ingest
```bash
python scripts/ingest.py --collection personal_docs
python scripts/ingest.py --collection domain_corpus --strategy semantic
```

### Sanity-check retrieval
```bash
python scripts/test_retrieval.py --collection domain_corpus \
  --query "What is retrieval augmented generation?"
```

## Project structure
```
config.py                  # all tunable knobs in one place
ingestion/
  loaders.py                # PDF / docx / md / txt -> Document
  chunker.py                 # recursive & semantic chunking strategies
embeddings/
  embedder.py                 # bge-large-en-v1.5 wrapper
vectorstore/
  qdrant_store.py              # Qdrant create/upsert/search
scripts/
  ingest.py                     # main pipeline entrypoint
  test_retrieval.py              # manual retrieval sanity check
data/
  raw/                            # your source files (gitignored)
  processed/                       # cache (gitignored)
```

## Week 2 — Hybrid Search, Reranking, and Evaluation

### Test hybrid search + reranking on a single query
```bash
python scripts/test_hybrid.py --collection domain_corpus --query "What is retrieval augmented generation?"
```
This shows results at each stage: hybrid search (dense + BM25 fused via
Reciprocal Rank Fusion) first, then the same candidates re-scored by a
cross-encoder reranker. Compare the ordering — reranking should push the
most relevant chunks higher.

### Run the eval set (get real recall@k / MRR numbers)
```bash
python eval/run_eval.py --collection domain_corpus --k 5
```
This runs all eval questions through three configurations — dense-only,
hybrid, and hybrid+reranked — and prints recall@5 and MRR for each, so you
can see the measurable improvement from each stage. Run it again with
`--k 1` and `--k 3` to see how much reranking specifically helps get the
right answer into the very top spot.

Add more questions to `eval/eval_set.json` over time — 20-30 questions is
a good target for a resume-credible eval set. This is the actual evidence
behind a resume line like "achieved X% recall@5" — write down the numbers
you get here.

## Project structure (updated)
```
retrieval/
  bm25_search.py              # BM25 keyword search over Qdrant's stored payload text
  hybrid_search.py             # fuses dense + BM25 via Reciprocal Rank Fusion
  reranker.py                   # cross-encoder reranking (bge-reranker-base)
eval/
  eval_set.json                  # test questions with known correct source documents
  run_eval.py                     # computes recall@k and MRR across all 3 configs
scripts/
  test_hybrid.py                  # manual single-query test of hybrid + rerank
```

## Design notes — Week 2
- **Why Reciprocal Rank Fusion instead of weighted score averaging**: dense
  cosine similarity (0-1) and BM25 scores (unbounded, corpus-dependent) live
  on different scales. RRF only uses *rank position* from each list, so it
  sidesteps the normalization problem entirely — this is what production
  search systems (Elasticsearch, Weaviate, Azure AI Search) use by default.
- **Two-stage retrieval (retrieve wide, then rerank narrow)**: hybrid search
  pulls ~20 candidates cheaply, then the cross-encoder reranks just those 20
  with a slower-but-more-accurate model, keeping only the true top 5 for the
  LLM's context window. Running the cross-encoder over the whole corpus
  directly would be far too slow.
- **BM25 index rebuilds from Qdrant's payload** — no second document store
  needed, since chunk text already lives in the vector DB's payload field.

## Week 3 — LLM Integration and Conversational Memory

### Get an API key
This project uses Claude by default. Get a key from https://console.anthropic.com/
(or set `LLM_PROVIDER=openai` in `.env` and use an OpenAI key instead).
Add it to your `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### Run the interactive chatbot
```bash
python scripts/chat.py
```
This is the full pipeline running together: your question gets rewritten
into a standalone form (resolving "it"/"that" references using recent
history), searched across both `personal_docs` and `domain_corpus` via
hybrid search, reranked, then handed to the LLM along with grounding
instructions -- the model is told to answer ONLY from the retrieved
context and to cite sources like `[Source 1]`.

Try a multi-turn conversation to see memory in action, e.g.:
```
You: What is RETRO?
Assistant: [answer about RETRO, citing sources]

You: How does it compare to REALM?
  [rewritten query: How does RETRO compare to REALM?]
Assistant: [answer comparing both, correctly resolving "it" to RETRO]
```

## Project structure (updated)
```
llm/
  client.py                     # wraps Anthropic/OpenAI API behind one interface
  prompt_builder.py              # builds grounded, citation-instructed system prompts
  query_rewriter.py               # resolves follow-up references ("it", "that") before retrieval
memory/
  conversation_memory.py           # short-term buffer + LLM-summarized long-term memory
scripts/
  chat.py                            # the full assembled interactive chatbot
```

## Design notes — Week 3
- **Grounding + citation instructions in the system prompt** are what stop
  the model from silently answering out of its own general training
  knowledge instead of your actual documents -- without explicit
  instructions, LLMs will often answer confidently from memory even when
  you intended RAG to be the only source of truth.
- **Query rewriting happens BEFORE retrieval, not after** -- a follow-up
  question like "how does it compare?" would otherwise fail to retrieve
  anything useful, because "it" carries no embeddable meaning on its own.
- **Search both collections and let reranking decide relevance**, rather
  than building a separate classifier to route "is this a resume question
  or a papers question" -- simpler, and the reranker already does this
  job as a side effect of scoring true relevance.
- **Buffer + summarization memory** trades a bit of detail for staying
  within the model's context window indefinitely, instead of the prompt
  growing forever as a conversation continues.
- **Page-level chunking for PDFs**: each `Document` keeps its source page,
  so retrieved chunks can cite "source.pdf, page 4" instead of just a filename.
- **bge query instruction prefix**: `bge` models need a specific instruction
  string prepended to *queries* (not documents) for best retrieval quality —
  handled in `Embedder.embed_query`. Easy detail to miss, meaningful quality bump.
- **Overlap in recursive chunking**: consecutive chunks share a token tail so
  facts that straddle a chunk boundary aren't lost entirely from either chunk.
- **Payload-stored text**: chunk text lives in Qdrant's payload (not just the
  vector), so BM25 can be computed over the same payload at query time without
  a second document store.

## Week 4 — API, Frontend, and Deployment

### Run the backend API
```bash
uvicorn api.main:app --reload --port 8000
```
Visit http://localhost:8000/docs for FastAPI's auto-generated interactive
API explorer -- you can test the `/chat` endpoint directly from the
browser, no frontend needed. This is worth demoing live in an interview.

### Run the frontend (in a SEPARATE terminal, backend must already be running)
```bash
streamlit run frontend/app.py
```
Opens a chat UI in your browser at http://localhost:8501, talking to the
FastAPI backend over HTTP.

### Test with multiple "users" at once
Open the Streamlit URL in two different browser tabs -- each gets its own
`session_id` and separate conversation memory, proving the API correctly
serves multiple sessions rather than sharing one global conversation
(the bug we'd have if we'd kept using the CLI script's single global
memory object for a real web app).

### Deployment
The `Dockerfile` packages the FastAPI backend (with your already-ingested
`data/qdrant_local` vector store baked in) into a container. You do NOT
need Docker installed locally to deploy -- platforms like HuggingFace
Spaces or Render build the Dockerfile remotely on their own servers.

**HuggingFace Spaces (free tier, simplest option):**
1. Create a new Space at https://huggingface.co/new-space, select "Docker" as the Space type
2. Push this repo to the Space's git remote
3. Add `ANTHROPIC_API_KEY` / `GROQ_API_KEY` as a Space secret (Settings -> Repository secrets), not committed in `.env`
4. The Space builds the Dockerfile automatically and gives you a public URL

**Render / Railway (also have free tiers):** connect your GitHub repo,
they detect the Dockerfile automatically, add your API key as an
environment variable in their dashboard.

## Project structure (updated — Week 4)
```
core/
  pipeline.py                 # the full RAG pipeline as a reusable class (used by both CLI and API)
api/
  main.py                      # FastAPI backend exposing /chat and /health endpoints
frontend/
  app.py                         # Streamlit chat UI, calls the API over HTTP
Dockerfile                         # containerizes the backend for deployment
```

## Design notes — Week 4
- **Why extract `core/pipeline.py` instead of importing straight from
  `scripts/chat.py`**: the CLI script used one global `ConversationMemory`
  object for the whole session, which is fine for one person typing in a
  terminal, but breaks immediately for a real web app serving multiple
  users/tabs at once -- each needs its own separate memory, keyed by
  `session_id`. Pulling the logic into a class made this possible without
  duplicating the retrieval/generation code in two places.
- **Models load once at server startup (`@app.on_event("startup")`), not
  per-request** -- loading the embedding model, BM25 indexes, and reranker
  takes real time; doing that on every single API call would make each
  response painfully slow.
- **CORS middleware** is what allows a browser-based frontend (Streamlit,
  running on a different port) to actually call this API -- browsers
  block cross-origin requests by default as a security measure.
- **Docker doesn't require Docker installed on your dev machine** -- it's
  only needed by whatever platform ultimately deploys the container.
  Local development the whole project stayed Docker-free the entire time.

  🔗 **[Try the live demo here](https://your-actual-streamlit-url.streamlit.app)**
