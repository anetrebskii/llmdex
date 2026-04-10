#!/usr/bin/env python3
"""Core indexing logic, shared by CLI and server."""

import json
import logging
import os
import subprocess
import time
from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
    load_index_from_storage,
)
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SentenceSplitter,
)
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

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


def _git_ignored_files(root: Path, files: list[str]) -> set[str]:
    """Return subset of files that are git-ignored. Uses git check-ignore."""
    if not (root / ".git").exists():
        return set()
    try:
        result = subprocess.run(
            ["git", "check-ignore", "--stdin", "-z"],
            input="\0".join(files),
            capture_output=True,
            text=True,
            cwd=root,
            timeout=30,
        )
        if result.returncode not in (0, 1):  # 1 = none ignored
            return set()
        return set(result.stdout.strip("\0").split("\0")) if result.stdout else set()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return set()


def collect_files(root: Path, extensions: tuple[str, ...]) -> tuple[list[str], int]:
    """Collect files matching extensions, skipping ignored directories and gitignored files.
    Returns (files, gitignored_count)."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if f.endswith(extensions):
                files.append(os.path.join(dirpath, f))

    ignored = _git_ignored_files(root, files)
    if ignored:
        files = [f for f in files if f not in ignored]
    return files, len(ignored)


DATA_DIR = Path.home() / ".llmdex" / "indexes"


def storage_dir(workspace: Path) -> Path:
    """Central storage: ~/.llmdex/indexes/<hash>-<name>/"""
    import hashlib

    key = hashlib.sha256(str(workspace).encode()).hexdigest()[:12]
    name = workspace.name
    return DATA_DIR / f"{key}-{name}"


def _log(msg):
    print(msg, flush=True)


def _manifest_path(store: Path) -> Path:
    return store / "file_manifest.json"


def _build_manifest(files: list[str]) -> dict[str, dict]:
    """Build {filepath: {mtime, size}} manifest from file list."""
    manifest = {}
    for f in files:
        try:
            st = os.stat(f)
            manifest[f] = {"mtime": st.st_mtime, "size": st.st_size}
        except OSError:
            pass
    return manifest


def _load_manifest(store: Path) -> dict[str, dict]:
    mp = _manifest_path(store)
    if not mp.exists():
        return {}
    try:
        return json.loads(mp.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_manifest(store: Path, manifest: dict[str, dict]):
    _manifest_path(store).write_text(json.dumps(manifest))


def _diff_files(
    current_files: list[str], old_manifest: dict[str, dict]
) -> tuple[list[str], list[str], list[str]]:
    """Compare current files against old manifest.
    Returns (new_files, changed_files, deleted_files)."""
    current_set = set(current_files)
    old_set = set(old_manifest.keys())

    new = [f for f in current_files if f not in old_set]
    deleted = [f for f in old_set if f not in current_set]
    changed = []
    for f in current_files:
        if f in old_set:
            try:
                st = os.stat(f)
                old = old_manifest[f]
                if st.st_mtime != old["mtime"] or st.st_size != old["size"]:
                    changed.append(f)
            except OSError:
                changed.append(f)
    return new, changed, deleted


def _enrich_node_text(node, root: Path | None = None) -> None:
    """Prepend file/language context to node text for better embeddings."""
    file_path = node.metadata.get("file_path", "")
    ext = os.path.splitext(file_path)[1].lower()
    language = PARSER_MAP.get(ext, "text")

    # Use relative path if root is provided
    if root and file_path:
        try:
            file_path = str(Path(file_path).relative_to(root))
        except ValueError:
            pass

    parts = [f"file: {file_path}"]

    if language == "markdown":
        header = node.metadata.get("header_path")
        if header:
            parts.append(f"section: {header}")
    elif node.metadata.get("symbol"):
        parts.append(f"symbol: {node.metadata['symbol']}")
    else:
        parts.append(f"language: {language}")

    node.text = "# " + " | ".join(parts) + "\n" + node.text


def parse_files(file_paths: list[str], embed_model, log=_log, root: Path | None = None) -> list:
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
            from llm_index.ast_chunker import chunk_file

            skipped = 0
            for fp in paths:
                chunks = chunk_file(fp, parser_type)
                if not chunks:
                    # Fallback: load via SimpleDirectoryReader + sentence splitter
                    doc = SimpleDirectoryReader(input_files=[fp]).load_data()
                    nodes = fallback_parser.get_nodes_from_documents(doc)
                    all_nodes.extend(nodes)
                    skipped += 1
                    continue
                for chunk in chunks:
                    node = TextNode(
                        text=chunk.text,
                        metadata={
                            "file_path": fp,
                            "file_name": os.path.basename(fp),
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "symbol": chunk.symbol,
                        },
                    )
                    node.excluded_embed_metadata_keys = ["file_path", "file_name", "start_line", "end_line", "symbol"]
                    node.excluded_llm_metadata_keys = ["file_path", "file_name", "start_line", "end_line", "symbol"]
                    all_nodes.append(node)
            if skipped:
                log(f"  -> {skipped} files used fallback parser")

        else:  # text
            nodes = fallback_parser.get_nodes_from_documents(docs)
            all_nodes.extend(nodes)

        log(f"  -> {len(all_nodes)} total nodes")

    # Compute line numbers for each node from character offsets
    # Group nodes by file to avoid re-reading the same file
    nodes_by_file: dict[str, list] = {}
    for node in all_nodes:
        fp = node.metadata.get("file_path")
        if fp and (node.start_char_idx is not None or node.end_char_idx is not None):
            nodes_by_file.setdefault(fp, []).append(node)

    for fp, nodes in nodes_by_file.items():
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError:
            continue

        # Build cumulative newline positions once per file
        newlines = [-1]  # sentinel: "line 1 starts after index -1"
        for i, ch in enumerate(content):
            if ch == "\n":
                newlines.append(i)

        def _char_to_line(idx: int) -> int:
            # Binary search for the line number
            lo, hi = 0, len(newlines) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if newlines[mid] < idx:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return lo  # 1-based line number

        for node in nodes:
            if node.start_char_idx is not None:
                node.metadata["start_line"] = _char_to_line(node.start_char_idx)
            if node.end_char_idx is not None:
                node.metadata["end_line"] = _char_to_line(node.end_char_idx)

    # Enrich node text with file/language context for better embeddings
    for node in all_nodes:
        _enrich_node_text(node, root)

    return all_nodes


def _load_embed_model(verbose: bool, log=_log):
    """Load embedding model, suppressing noisy logs unless verbose."""
    log(f"Loading embedding model ({EMBED_MODEL_NAME})...")
    if not verbose:
        for name in ("httpx", "sentence_transformers", "llama_index"):
            logging.getLogger(name).setLevel(logging.WARNING)
        _orig_stdout = os.dup(1)
        _orig_stderr = os.dup(2)
        _devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull, 1)
        os.dup2(_devnull, 2)
    try:
        embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
        Settings.embed_model = embed_model
        Settings.llm = None
    finally:
        if not verbose:
            os.dup2(_orig_stdout, 1)
            os.dup2(_orig_stderr, 2)
            os.close(_devnull)
            os.close(_orig_stdout)
            os.close(_orig_stderr)
    return embed_model


def build_index(
    workspace: Path,
    extensions: tuple[str, ...] | None = None,
    embed_model=None,
    log=_log,
    verbose: bool = False,
) -> dict:
    """Build vector index for a workspace. Returns stats dict."""
    start = time.time()

    if extensions is None:
        extensions = (".md", ".ts", ".json")

    all_files, gitignored = collect_files(workspace, extensions)

    # Show per-extension counts
    ext_counts: dict[str, int] = {}
    for f in all_files:
        ext = os.path.splitext(f)[1].lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    summary = ", ".join(f"{count} {ext}" for ext, count in sorted(ext_counts.items()))
    log(f"Found {len(all_files)} files ({summary})")
    if gitignored:
        log(f"Excluded {gitignored} file(s) via .gitignore")

    if verbose:
        for f in all_files:
            log(f"  {f}")

    if not all_files:
        return {"files": 0, "nodes": 0, "elapsed": 0, "error": "No files found"}

    out = storage_dir(workspace)
    old_manifest = _load_manifest(out)
    new_files, changed_files, deleted_files = _diff_files(all_files, old_manifest)
    files_to_parse = new_files + changed_files

    # Check if we can do an incremental update
    has_existing_index = old_manifest and (out / "docstore.json").exists()
    if has_existing_index and not files_to_parse and not deleted_files:
        elapsed = time.time() - start
        log(f"\nNo changes detected. Index is up to date ({elapsed:.1f}s)")
        return {
            "files": len(all_files),
            "nodes": 0,
            "elapsed": round(elapsed, 1),
            "directory": str(workspace),
            "extensions": list(extensions),
            "skipped": True,
        }

    if embed_model is None:
        embed_model = _load_embed_model(verbose, log)

    if has_existing_index and (len(files_to_parse) + len(deleted_files)) < len(all_files):
        # Incremental update
        log(f"Incremental update: {len(new_files)} new, {len(changed_files)} changed, {len(deleted_files)} deleted")
        storage_context = StorageContext.from_defaults(persist_dir=str(out))
        index = load_index_from_storage(storage_context, embed_model=embed_model)

        # Remove nodes for changed and deleted files
        for f in changed_files + deleted_files:
            try:
                index.delete_ref_doc(f, delete_from_docstore=True)
            except (KeyError, ValueError):
                pass

        # Parse and insert new/changed files
        if files_to_parse:
            new_nodes = parse_files(files_to_parse, embed_model, log=log, root=workspace)
            log(f"Inserting {len(new_nodes)} nodes...")
            index.insert_nodes(new_nodes)
        else:
            new_nodes = []

        log(f"Saving index to {out}...")
        index.storage_context.persist(persist_dir=str(out))

        _save_manifest(out, _build_manifest(all_files))
        elapsed = time.time() - start
        log(
            f"\nDone! Updated {len(files_to_parse)} files ({len(new_nodes)} nodes) in {elapsed:.1f}s"
        )
    else:
        # Full rebuild
        if old_manifest:
            log("Full rebuild...")
        all_nodes = parse_files(all_files, embed_model, log=log, root=workspace)
        log(f"Total: {len(all_nodes)} nodes")

        log("Building vector index...")
        index = VectorStoreIndex(all_nodes, embed_model=embed_model)

        out.mkdir(parents=True, exist_ok=True)
        log(f"Saving index to {out}...")
        index.storage_context.persist(persist_dir=str(out))

        _save_manifest(out, _build_manifest(all_files))
        elapsed = time.time() - start
        log(
            f"\nDone! Indexed {len(all_files)} files ({len(all_nodes)} nodes) in {elapsed:.1f}s"
        )
        new_nodes = all_nodes

    # Register this folder
    from llm_index.registry import register

    register(str(workspace), list(extensions))

    return {
        "files": len(all_files),
        "nodes": len(new_nodes),
        "elapsed": round(elapsed, 1),
        "directory": str(workspace),
        "extensions": list(extensions),
    }
