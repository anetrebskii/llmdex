#!/usr/bin/env python3
"""Query the indexed workspace documents."""

import sys
from pathlib import Path

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

STORAGE_DIR = Path(__file__).resolve().parent / "storage"


def query(question: str, top_k: int = 5):
    # Load embedding model
    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    Settings.embed_model = embed_model
    Settings.llm = None

    # Load index
    storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    index = load_index_from_storage(storage_context)

    # Query - retriever only (no LLM synthesis)
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(question)

    print(f"\nQuery: {question}")
    print(f"Top {top_k} results:\n")

    for i, node in enumerate(results, 1):
        source = node.metadata.get("file_path", "unknown")
        # Make path relative to workspace
        try:
            source = str(Path(source).relative_to(Path(__file__).resolve().parent.parent))
        except ValueError:
            pass
        score = node.score
        text_preview = node.text[:200].replace("\n", " ")
        print(f"{i}. [{score:.3f}] {source}")
        print(f"   {text_preview}...")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query.py 'your search question' [top_k]")
        sys.exit(1)

    question = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    query(question, top_k)
