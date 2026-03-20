#!/usr/bin/env python3
"""Core indexing logic, shared by CLI and server."""

import os
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

# Extension -> parser type
PARSER_MAP = {
    ".md": "markdown",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".json": "text",
    ".py": "python",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".css": "text",
    ".html": "text",
    ".yaml": "text",
    ".yml": "text",
    ".toml": "text",
    ".txt": "text",
}

CODE_LANGUAGES = {"typescript", "javascript", "python", "rust", "go", "java"}


def collect_files(root: Path, extensions: tuple[str, ...]) -> list[str]:
    """Collect files matching extensions, skipping ignored directories."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if f.endswith(extensions):
                files.append(os.path.join(dirpath, f))
    return files


DATA_DIR = Path.home() / ".llmdex" / "indexes"


def storage_dir(workspace: Path) -> Path:
    """Central storage: ~/.llmdex/indexes/<hash>-<name>/"""
    import hashlib
    key = hashlib.sha256(str(workspace).encode()).hexdigest()[:12]
    name = workspace.name
    return DATA_DIR / f"{key}-{name}"


def _log(msg):
    print(msg, flush=True)


def parse_files(file_paths: list[str], embed_model, log=_log) -> list:
    """Parse files into nodes using appropriate parsers. Returns list of nodes."""
    # Group files by parser type
    groups: dict[str, list[str]] = {}
    for f in file_paths:
        ext = os.path.splitext(f)[1].lower()
        parser_type = PARSER_MAP.get(ext, "text")
        groups.setdefault(parser_type, []).append(f)

    all_nodes = []
    fallback_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    for parser_type, paths in groups.items():
        log(f"Parsing {len(paths)} {parser_type} files...")
        docs = SimpleDirectoryReader(input_files=paths).load_data()

        if parser_type == "markdown":
            nodes = MarkdownNodeParser().get_nodes_from_documents(docs)
            all_nodes.extend(nodes)

        elif parser_type in CODE_LANGUAGES:
            code_parser = CodeSplitter(language=parser_type, chunk_lines=40, chunk_lines_overlap=10)
            skipped = 0
            for doc in docs:
                try:
                    nodes = code_parser.get_nodes_from_documents([doc])
                    all_nodes.extend(nodes)
                except ValueError:
                    nodes = fallback_parser.get_nodes_from_documents([doc])
                    all_nodes.extend(nodes)
                    skipped += 1
            if skipped:
                log(f"  -> {skipped} files used fallback parser")

        else:  # text
            nodes = fallback_parser.get_nodes_from_documents(docs)
            all_nodes.extend(nodes)

        log(f"  -> {len(nodes) if parser_type != 'markdown' else len(all_nodes)} nodes")

    return all_nodes


def build_index(workspace: Path, extensions: tuple[str, ...] | None = None,
                embed_model=None, log=_log) -> dict:
    """Build vector index for a workspace. Returns stats dict."""
    start = time.time()

    if extensions is None:
        extensions = (".md", ".ts", ".json")

    if embed_model is None:
        log("Loading embedding model (all-MiniLM-L6-v2)...")
        embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
        Settings.embed_model = embed_model
        Settings.llm = None

    all_files = collect_files(workspace, extensions)
    log(f"Found {len(all_files)} files")

    for f in all_files:
        log(f"  {f}")

    if not all_files:
        return {"files": 0, "nodes": 0, "elapsed": 0, "error": "No files found"}

    all_nodes = parse_files(all_files, embed_model, log=log)
    log(f"Total: {len(all_nodes)} nodes")

    log("Building vector index...")
    index = VectorStoreIndex(all_nodes, embed_model=embed_model)

    out = storage_dir(workspace)
    out.mkdir(parents=True, exist_ok=True)
    log(f"Saving index to {out}...")
    index.storage_context.persist(persist_dir=str(out))

    elapsed = time.time() - start
    log(f"\nDone! Indexed {len(all_files)} files ({len(all_nodes)} nodes) in {elapsed:.1f}s")

    # Register this folder
    from llm_index.registry import register
    register(str(workspace), list(extensions))

    return {
        "files": len(all_files),
        "nodes": len(all_nodes),
        "elapsed": round(elapsed, 1),
        "directory": str(workspace),
        "extensions": list(extensions),
    }
