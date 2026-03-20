#!/usr/bin/env python3
"""Index all markdown documents in the workspace using LlamaIndex with local embeddings."""

import os
import sys
import time
from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.node_parser import MarkdownNodeParser, CodeSplitter, SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


WORKSPACE = Path(__file__).resolve().parent.parent
INDEX_DIR = Path(__file__).resolve().parent
STORAGE_DIR = INDEX_DIR / "storage"

# Directories to skip
SKIP_DIRS = {
    "node_modules",
    ".git",
    ".llm-index",
    ".claude",
    ".cursor",
    ".vscode",
    ".next",
    ".nuxt",
    ".turbo",
    ".cache",
    ".parcel-cache",
    ".svelte-kit",
    "dist",
    "build",
    "out",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "egg-info",
    ".eggs",
    "coverage",
    ".nyc_output",
    "rnd",
}


def collect_markdown_files(root: Path) -> list[str]:
    """Collect all markdown files, skipping ignored directories."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip ignored directories in-place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if f.endswith((".md", ".ts", ".json")):
                files.append(os.path.join(dirpath, f))
    return files


def build_index():
    start = time.time()

    # Local embeddings - no API key needed
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    Settings.embed_model = embed_model
    Settings.llm = None  # No LLM needed for indexing

    # Collect files
    md_files = collect_markdown_files(WORKSPACE)
    print(f"Found {len(md_files)} markdown files")

    for f in md_files:
        rel = os.path.relpath(f, WORKSPACE)
        print(f"  {rel}")

    if not md_files:
        print("No files found!")
        sys.exit(1)

    # Split files by type
    md_paths = [f for f in md_files if f.endswith(".md")]
    ts_paths = [f for f in md_files if f.endswith(".ts")]
    json_paths = [f for f in md_files if f.endswith(".json")]

    all_nodes = []
    fallback_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    # Parse markdown files with markdown-aware parser
    if md_paths:
        print(f"Loading {len(md_paths)} markdown files...")
        md_docs = SimpleDirectoryReader(input_files=md_paths).load_data()
        md_parser = MarkdownNodeParser()
        md_nodes = md_parser.get_nodes_from_documents(md_docs)
        print(f"  -> {len(md_nodes)} markdown nodes")
        all_nodes.extend(md_nodes)

    # Parse TypeScript files with code-aware splitter, fallback to sentence splitter
    if ts_paths:
        print(f"Loading {len(ts_paths)} TypeScript files...")
        ts_docs = SimpleDirectoryReader(input_files=ts_paths).load_data()
        code_parser = CodeSplitter(language="typescript", chunk_lines=40, chunk_lines_overlap=10)
        skipped = 0
        for doc in ts_docs:
            try:
                nodes = code_parser.get_nodes_from_documents([doc])
                all_nodes.extend(nodes)
            except ValueError:
                nodes = fallback_parser.get_nodes_from_documents([doc])
                all_nodes.extend(nodes)
                skipped += 1
        print(f"  -> {len([n for n in all_nodes])} total nodes ({skipped} files used fallback parser)")

    # Parse JSON files with sentence splitter
    if json_paths:
        print(f"Loading {len(json_paths)} JSON files...")
        json_docs = SimpleDirectoryReader(input_files=json_paths).load_data()
        json_nodes = fallback_parser.get_nodes_from_documents(json_docs)
        print(f"  -> {len(json_nodes)} JSON nodes")
        all_nodes.extend(json_nodes)

    nodes = all_nodes
    print(f"Total: {len(nodes)} nodes")

    # Build index
    print("Building vector index...")
    index = VectorStoreIndex(nodes, embed_model=embed_model)

    # Persist
    print(f"Saving index to {STORAGE_DIR}...")
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))

    elapsed = time.time() - start
    print(f"\nDone! Indexed {len(md_files)} files ({len(nodes)} nodes) in {elapsed:.1f}s")
    print(f"Index saved to: {STORAGE_DIR}")


if __name__ == "__main__":
    build_index()
