#!/usr/bin/env python3
"""Core indexing logic, shared by CLI and server."""

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from llm_index.registry import EMBED_MODEL_NAME, get_entry, storage_dir

# llama_index and torch are imported where they are used: `llmdex query` and a reindex with no changes never need them.

# CPU by default: on MPS the model's Metal allocation (~1 GB) stays resident for the server's lifetime.
EMBED_DEVICE = os.environ.get("LLMDEX_EMBED_DEVICE", "cpu")

# One file per index, replaced atomically: chunk text, file and lines as JSON, and a float32 row per chunk.
INDEX_FILE = "index.npz"
# What the llama_index JSON format left behind before index.npz.
LEGACY_FILES = ("docstore.json", "default__vector_store.json", "index_store.json", "graph_store.json", "image__vector_store.json", "embed_cache.json")


def make_hf_embedding(model_name=None, device=EMBED_DEVICE, batch_size=10):
    """Build the HuggingFaceEmbedding. e5 models need query/passage prefixes to perform well."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    model_name = model_name or EMBED_MODEL_NAME
    kwargs = {"model_name": model_name, "device": device, "embed_batch_size": batch_size}
    if "e5" in model_name.lower():
        kwargs["query_instruction"] = "query: "
        kwargs["text_instruction"] = "passage: "
    try:
        # A cached model loads in 0.8 s instead of 8 s, because the hub is not asked for updates; the hub is used only when the model is missing.
        return HuggingFaceEmbedding(**kwargs, local_files_only=True)
    except OSError:
        return HuggingFaceEmbedding(**kwargs)

SKIP_DIRS = {
    "node_modules",
    ".git",
    ".notula",
    ".obsidian",
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
    ".cs": "csharp",
    ".dart": "dart",
    ".css": "text",
    ".html": "text",
    ".yaml": "text",
    ".yml": "text",
    ".toml": "text",
    ".txt": "text",
}

CODE_LANGUAGES = {"typescript", "javascript", "python", "rust", "go", "java", "csharp", "dart"}

# Inline base64 payloads (images in saved HTML) are unsearchable, and one file of them made 40k chunks.
DATA_URI = re.compile(r"(data:[\w/+.-]+;base64,)[A-Za-z0-9+/=]{100,}")
# HTML is indexed as the text a reader sees: scripts, styles, inline SVG, comments and tags go.
HTML_MARKUP = re.compile(r"<(script|style|svg)\b.*?</\1\s*>|<!--.*?-->|<[^>]*>", re.S | re.I)


def _clean_text(text: str, path: str) -> str:
    """Drop content nobody searches for. Newlines stay, so line numbers still match the file."""
    text = DATA_URI.sub(r"\1", text)
    if path.lower().endswith(".html"):
        text = HTML_MARKUP.sub(lambda m: "\n" * m.group().count("\n"), text)
        text = re.sub(r" ?\n ?", "\n", re.sub(r"[ \t]+", " ", text))
    return text

# Every extension we know how to parse. Indexing always uses this full set.
DEFAULT_EXTENSIONS = tuple(PARSER_MAP)


def _git_ignored_files(root: Path, files: list[str]) -> set[str]:
    """Return subset of files that are git-ignored by the repository holding root. Uses git check-ignore."""
    top = next((p for p in (root, *root.parents) if (p / ".git").exists()), None)
    if top is None:
        return set()
    try:
        result = subprocess.run(
            ["git", "check-ignore", "--stdin", "-z"],
            input="\0".join(files),
            capture_output=True,
            text=True,
            cwd=top,
            timeout=30,
        )
        if result.returncode not in (0, 1):  # 1 = none ignored
            return set()
        return set(result.stdout.strip("\0").split("\0")) if result.stdout else set()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return set()


def collect_files(
    root: Path, extensions: tuple[str, ...], root_only: bool = False
) -> tuple[list[str], int]:
    """Collect files matching extensions, skipping ignored directories and gitignored files.
    If root_only, only files directly under root (no subdirectories).
    Returns (files, gitignored_count)."""
    files = []
    if root_only:
        try:
            for name in os.listdir(root):
                fp = os.path.join(root, name)
                if os.path.isfile(fp) and name.endswith(extensions):
                    files.append(fp)
        except OSError:
            pass
    else:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for f in filenames:
                if f.endswith(extensions):
                    files.append(os.path.join(dirpath, f))

    ignored = _git_ignored_files(root, files)
    if ignored:
        files = [f for f in files if f not in ignored]
    return files, len(ignored)


def _log(msg):
    print(msg, flush=True)


def load_index(store: Path) -> tuple[dict, np.ndarray]:
    """Chunks as row-aligned lists (files, start_lines, end_lines, texts) and their float32 vectors."""
    with np.load(store / INDEX_FILE) as data:
        return json.loads(data["chunks"].tobytes()), data["vectors"]


def _save_index(store: Path, chunks: dict, vectors: np.ndarray):
    store.mkdir(parents=True, exist_ok=True)
    tmp = store / (INDEX_FILE + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, vectors=vectors, chunks=np.frombuffer(json.dumps(chunks, ensure_ascii=False).encode(), dtype=np.uint8))
    os.replace(tmp, store / INDEX_FILE)


def _legacy_vectors(store: Path) -> dict[str, np.ndarray]:
    """{text_md5: vector} from an index in the llama_index JSON format, so converting it embeds nothing twice."""
    vectors = json.loads((store / "default__vector_store.json").read_text(encoding="utf-8"))["embedding_dict"]
    docs = json.loads((store / "docstore.json").read_text(encoding="utf-8"))["docstore/data"]
    return {
        _text_hash(d["__data__"]["text"]): np.asarray(vectors[nid], dtype=np.float32)
        for nid, d in docs.items()
        if nid in vectors
    }


def _text_hash(text: str) -> str:
    # Key by model too: embeddings are model-specific, so a model switch must
    # miss the cache and recompute rather than reuse another model's vectors.
    return hashlib.md5((EMBED_MODEL_NAME + "\0" + text).encode()).hexdigest()


def _embed(nodes: list, embed_model, reuse: dict[str, np.ndarray], verbose: bool, log=_log) -> np.ndarray:
    """float32 vectors for nodes: reused where a chunk's text is unchanged, computed for the rest."""
    if not nodes:
        return np.zeros((0, 0), dtype=np.float32)
    hashes = [_text_hash(n.text) for n in nodes]
    missing = [i for i, h in enumerate(hashes) if h not in reuse]
    log(f"Embeddings: {len(nodes) - len(missing)} reused, {len(missing)} to compute")
    rows = [reuse.get(h) for h in hashes]
    if missing:
        embed_model = embed_model or _load_embed_model(verbose, log, len(missing))
        for start in range(0, len(missing), 1024):
            batch = missing[start:start + 1024]
            for i, vec in zip(batch, embed_model.get_text_embedding_batch([nodes[i].text for i in batch])):
                rows[i] = np.asarray(vec, dtype=np.float32)
    return np.stack(rows)


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
        return json.loads(mp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_manifest(store: Path, manifest: dict[str, dict]):
    _manifest_path(store).write_text(json.dumps(manifest), encoding="utf-8")


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
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
    from llama_index.core.schema import TextNode
    from llama_index.core.utils import get_tokenizer

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
        for doc in docs:
            doc.set_content(_clean_text(doc.text, doc.metadata["file_path"]))

        if parser_type == "markdown":
            # The model reads only a chunk's first 512 tokens, so a longer section is split like plain text.
            tokenize = get_tokenizer()
            for section in MarkdownNodeParser().get_nodes_from_documents(docs):
                if len(tokenize(section.text)) <= fallback_parser.chunk_size:
                    all_nodes.append(section)
                    continue
                for part in fallback_parser.get_nodes_from_documents([section]):
                    if section.start_char_idx is None or part.start_char_idx is None:
                        part.start_char_idx = part.end_char_idx = None
                    else:
                        part.start_char_idx += section.start_char_idx
                        part.end_char_idx += section.start_char_idx
                    all_nodes.append(part)

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

    all_nodes = [n for n in all_nodes if n.text.strip()]

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
                content = _clean_text(fh.read(), fp)
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


_indexing_models: dict[str, object] = {}


def _load_embed_model(verbose: bool, log=_log, count: int = 0):
    """Load embedding model once per process and device, suppressing noisy logs unless verbose."""
    import torch

    # MPS embeds ~6x faster but peaks ~2 GB higher (3.1 vs 1.2 GB for 116 chunks), which pays off only for a large batch.
    use_mps = torch.backends.mps.is_available() and (count >= 1000 or "mps" in _indexing_models)
    device = os.environ.get("LLMDEX_EMBED_DEVICE") or ("mps" if use_mps else "cpu")
    if device in _indexing_models:
        return _indexing_models[device]
    log(f"Loading embedding model ({EMBED_MODEL_NAME}, {device})...")
    if not verbose:
        for name in ("httpx", "sentence_transformers", "llama_index"):
            logging.getLogger(name).setLevel(logging.WARNING)
        _orig_stdout = os.dup(1)
        _orig_stderr = os.dup(2)
        _devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull, 1)
        os.dup2(_devnull, 2)
    try:
        _indexing_models[device] = make_hf_embedding(device=device, batch_size=64)
    finally:
        if not verbose:
            os.dup2(_orig_stdout, 1)
            os.dup2(_orig_stderr, 2)
            os.close(_devnull)
            os.close(_orig_stdout)
            os.close(_orig_stderr)
    return _indexing_models[device]


def build_index(
    workspace: Path,
    embed_model=None,
    log=_log,
    verbose: bool = False,
    force: bool = False,
    split: bool = False,
    parent_tags: list[str] | None = None,
) -> dict:
    """Build vector index for a workspace. Returns stats dict.

    If split=True, also indexes each immediate subfolder as a separate child index
    tagged with folder:<name>. parent_tags is inherited by children."""
    if split:
        return _build_index_split(
            workspace, embed_model, log, verbose, force, parent_tags
        )

    return _build_index_single(
        workspace, embed_model, log, verbose, force, root_only=False
    )


def _build_index_split(
    workspace: Path,
    embed_model,
    log,
    verbose: bool,
    force: bool,
    parent_tags: list[str] | None,
) -> dict:
    from llm_index.registry import register, set_tags, get_entry, drop_indexes

    entry = get_entry(str(workspace)) or {}
    skip = set(entry.get("skip", []))

    log(f"=== Root (root-level files only): {workspace} ===")
    root_result = _build_index_single(
        workspace, embed_model, log, verbose, force, root_only=True
    )
    if root_result.get("error"):
        shutil.rmtree(storage_dir(workspace), ignore_errors=True)
        root_result = {"files": 0, "nodes": 0, "elapsed": 0, "directory": str(workspace)}

    # Discover immediate subfolders
    subfolders: list[Path] = []
    try:
        for name in sorted(os.listdir(workspace)):
            sub = workspace / name
            if sub.is_dir() and name not in SKIP_DIRS and not name.startswith(".") and name not in skip:
                subfolders.append(sub)
    except OSError:
        pass

    children: list[str] = []
    for sub in subfolders:
        log(f"\n=== Subfolder: {sub} ===")
        if str(sub) not in entry.get("children", []) and get_entry(str(sub)) is not None:
            log(f"  note: {sub} was already indexed independently -- overwriting")

        result = _build_index_single(
            sub, embed_model, log, verbose, force, root_only=False
        )
        if result.get("error") or result.get("files", 0) == 0:
            log("  skipped (no matching files)")
            continue

        child_tags = list(parent_tags or []) + [f"folder:{sub.name}"]
        set_tags(str(sub), child_tags)
        children.append(str(sub))

    stale = [c for c in entry.get("children", []) if c not in children]
    if stale:
        log(f"\nRemoving subfolder indexes no longer used: {', '.join(Path(c).name for c in stale)}")
        drop_indexes(stale)

    # Update parent registry entry with children list and split flag
    register(str(workspace), children=children, split=True)
    if parent_tags:
        set_tags(str(workspace), parent_tags)

    return {**root_result, "children": children, "split": True}


def _build_index_single(
    workspace: Path,
    embed_model,
    log,
    verbose: bool,
    force: bool,
    root_only: bool,
) -> dict:
    """Build a single vector index (no splitting). Returns stats dict."""
    start = time.time()

    all_files, gitignored = collect_files(workspace, DEFAULT_EXTENSIONS, root_only=root_only)

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

    out = storage_dir(workspace)
    if not all_files:
        for name in LEGACY_FILES:
            (out / name).unlink(missing_ok=True)
        return {"files": 0, "nodes": 0, "elapsed": 0, "error": "No files found"}

    same_model = (get_entry(str(workspace)) or {}).get("embed_model") == EMBED_MODEL_NAME
    old_manifest = _load_manifest(out) if same_model and not force and (out / INDEX_FILE).exists() else {}
    new_files, changed_files, deleted_files = _diff_files(all_files, old_manifest)

    if old_manifest and not new_files and not changed_files and not deleted_files:
        elapsed = time.time() - start
        log(f"\nNo changes detected. Index is up to date ({elapsed:.1f}s)")
        return {
            "files": len(all_files),
            "nodes": 0,
            "elapsed": round(elapsed, 1),
            "directory": str(workspace),
            "skipped": True,
        }

    # The existing index is the embedding cache: chunks whose text did not change keep their vectors.
    chunks = {"files": [], "start_lines": [], "end_lines": [], "texts": []}
    vectors = np.zeros((0, 0), dtype=np.float32)
    reuse: dict[str, np.ndarray] = {}
    loaded = False
    if same_model:
        try:
            if (out / INDEX_FILE).exists():
                chunks, vectors = load_index(out)
                loaded = True
            elif (out / "default__vector_store.json").exists():
                log("Converting the index from the llama_index JSON format...")
                reuse = _legacy_vectors(out)
        except Exception as e:
            # A killed run of an older version can leave truncated JSON, so rebuild instead of crashing.
            log(f"  existing index unreadable ({type(e).__name__}); rebuilding from scratch")
    if not loaded:
        new_files, changed_files, deleted_files = all_files, [], []
    files_to_parse = new_files + changed_files

    if old_manifest and loaded:
        log(f"Incremental update: {len(new_files)} new, {len(changed_files)} changed, {len(deleted_files)} deleted")
    elif loaded or reuse:
        log("Full rebuild...")

    # Rows of files that are gone or re-parsed leave the index; their vectors stay available for reuse.
    current, reparsed = set(all_files), set(files_to_parse)
    keep = np.array([f in current and f not in reparsed for f in chunks["files"]], dtype=bool)
    for i in np.flatnonzero(~keep):
        reuse[_text_hash(chunks["texts"][i])] = vectors[i]

    new_nodes = parse_files(files_to_parse, embed_model, log=log, root=workspace) if files_to_parse else []
    new_vectors = _embed(new_nodes, embed_model, reuse, verbose, log)
    reuse = {}

    kept = np.flatnonzero(keep)
    chunks = {
        "files": [chunks["files"][i] for i in kept] + [n.metadata["file_path"] for n in new_nodes],
        "start_lines": [chunks["start_lines"][i] for i in kept] + [n.metadata.get("start_line") for n in new_nodes],
        "end_lines": [chunks["end_lines"][i] for i in kept] + [n.metadata.get("end_line") for n in new_nodes],
        "texts": [chunks["texts"][i] for i in kept] + [n.text for n in new_nodes],
    }
    parts = [v for v in (vectors[kept], new_vectors) if len(v)]
    vectors = np.concatenate(parts) if parts else np.zeros((0, 0), dtype=np.float32)

    log(f"Saving index to {out}...")
    _save_index(out, chunks, vectors)
    _save_manifest(out, _build_manifest(all_files))
    for name in LEGACY_FILES:
        (out / name).unlink(missing_ok=True)
    elapsed = time.time() - start
    if old_manifest and loaded:
        log(f"\nDone! Updated {len(files_to_parse)} files ({len(new_nodes)} nodes) in {elapsed:.1f}s")
    else:
        log(f"\nDone! Indexed {len(all_files)} files ({len(new_nodes)} nodes) in {elapsed:.1f}s")

    # Register this folder
    from llm_index.registry import register

    register(str(workspace))

    return {
        "files": len(all_files),
        "nodes": len(new_nodes),
        "elapsed": round(elapsed, 1),
        "directory": str(workspace),
    }
