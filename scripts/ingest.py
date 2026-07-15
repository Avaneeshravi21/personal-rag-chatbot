"""
Week 1 pipeline entrypoint.

Usage:
    python scripts/ingest.py --collection personal_docs
    python scripts/ingest.py --collection domain_corpus --strategy semantic

This loads every file in the collection's raw folder, chunks it,
embeds it, and upserts it into Qdrant. Run it once per collection
after you drop files into data/raw/<collection_name>/.
"""
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ingestion.loaders import load_directory
from ingestion.chunker import chunk_document
from embeddings.embedder import Embedder
from vectorstore.qdrant_store import VectorStore


def run(collection_name: str, strategy: str = None):
    if collection_name not in config.COLLECTIONS:
        raise ValueError(f"Unknown collection '{collection_name}'. Options: {list(config.COLLECTIONS)}")

    coll_cfg = config.COLLECTIONS[collection_name]
    strategy = strategy or config.CHUNK_STRATEGY

    print(f"\n=== Ingesting '{collection_name}' ({coll_cfg['description']}) ===")
    print(f"Source dir: {coll_cfg['path']}")
    print(f"Chunk strategy: {strategy}\n")

    # 1. Load
    docs = load_directory(coll_cfg["path"])
    if not docs:
        print("[warn] no documents found. Add files to the path above and re-run.")
        return
    print(f"\nLoaded {len(docs)} raw documents (pages/files).")

    # 2. Chunk
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(
            doc,
            strategy=strategy,
            chunk_size=config.CHUNK_SIZE_TOKENS,
            overlap=config.CHUNK_OVERLAP_TOKENS,
        )
        all_chunks.extend(chunks)
    print(f"Produced {len(all_chunks)} chunks.")

    # 3. Embed
    print(f"\nEmbedding with {config.EMBEDDING_MODEL} ...")
    embedder = Embedder(config.EMBEDDING_MODEL)
    texts = [c.text for c in all_chunks]
    vectors = embedder.embed_documents(texts, batch_size=config.EMBEDDING_BATCH_SIZE)

    # 4. Store
    print(f"\nStoring into Qdrant collection '{collection_name}' ...")
    if config.USE_LOCAL_QDRANT:
        store = VectorStore(local_path=config.QDRANT_LOCAL_PATH)
    else:
        store = VectorStore(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    store.create_collection(collection_name, dim=config.EMBEDDING_DIM)
    store.upsert_chunks(collection_name, all_chunks, vectors)

    total = store.count(collection_name)
    print(f"\nDone. Collection '{collection_name}' now has {total} points total.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, choices=list(config.COLLECTIONS.keys()))
    parser.add_argument("--strategy", choices=["recursive", "semantic"], default=None)
    args = parser.parse_args()
    run(args.collection, args.strategy)
