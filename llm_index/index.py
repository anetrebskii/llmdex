#!/usr/bin/env python3
"""Index project files using LlamaIndex with local embeddings."""

import argparse
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

EXTENSIONS = (".md", ".ts", ".json")


def collect_files(root: Path) -> list[str]:
    """Collect all indexable files, skipping ignored directories."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if f.endswith(EXTENSIONS):
                files.append(os.path.join(dirpath, f))
    return files


def storage_dir(workspace: Path) -> Path:
    return workspace / ".llm-index" / "storage"


def build_index(workspace: Path):
    start = time.time()

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    Settings.embed_model = embed_model
    Settings.llm = None

    all_files = collect_files(workspace)
    print(f"Found {len(all_files)} files")

    for f in all_files:
        rel = os.path.relpath(f, workspace)
        print(f"  {rel}")

    if not all_files:
        print("No files found!")
        sys.exit(1)

    md_paths = [f for f in all_files if f.endswith(".md")]
    ts_paths = [f for f in all_files if f.endswith(".ts")]
    json_paths = [f for f in all_files if f.endswith(".json")]

    all_nodes = []
    fallback_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    if md_paths:
        print(f"Parsing {len(md_paths)} markdown files...")
        md_docs = SimpleDirectoryReader(input_files=md_paths).load_data()
        md_nodes = MarkdownNodeParser().get_nodes_from_documents(md_docs)
        print(f"  -> {len(md_nodes)} nodes")
        all_nodes.extend(md_nodes)

    if ts_paths:
        print(f"Parsing {len(ts_paths)} TypeScript files...")
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
        if skipped:
            print(f"  -> {skipped} files used fallback parser")

    if json_paths:
        print(f"Parsing {len(json_paths)} JSON files...")
        json_docs = SimpleDirectoryReader(input_files=json_paths).load_data()
        json_nodes = fallback_parser.get_nodes_from_documents(json_docs)
        print(f"  -> {len(json_nodes)} nodes")
        all_nodes.extend(json_nodes)

    print(f"Total: {len(all_nodes)} nodes")

    print("Building vector index...")
    index = VectorStoreIndex(all_nodes, embed_model=embed_model)

    out = storage_dir(workspace)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving index to {out}...")
    index.storage_context.persist(persist_dir=str(out))

    elapsed = time.time() - start
    print(f"\nDone! Indexed {len(all_files)} files ({len(all_nodes)} nodes) in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Index project files for semantic search")
    parser.add_argument("directory", nargs="?", default=".", help="Project directory to index (default: current dir)")
    args = parser.parse_args()

    workspace = Path(args.directory).resolve()
    if not workspace.is_dir():
        print(f"Error: {workspace} is not a directory")
        sys.exit(1)

    print(f"Indexing: {workspace}")
    build_index(workspace)


if __name__ == "__main__":
    main()
