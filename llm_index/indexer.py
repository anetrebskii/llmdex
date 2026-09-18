#!/usr/bin/env python3
"""Core indexing logic, shared by CLI and server."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from bisect import bisect_left, bisect_right
from fnmatch import fnmatch
from pathlib import Path

import numpy as np

from llm_index.registry import EMBED_MODEL_NAME, get_entry, storage_dir

# The model is loaded where it is used: `llmdex query` and a reindex with no changes never need it.

# One file per index, replaced atomically: chunk text, file and lines as JSON, and a float32 row per chunk.
INDEX_FILE = "index.npz"
# What the llama_index JSON format left behind before index.npz.
LEGACY_FILES = ("docstore.json", "default__vector_store.json", "index_store.json", "graph_store.json", "image__vector_store.json", "embed_cache.json")

# The model reads 512 tokens; the "passage: " prefix and the "# file: ... | section: ..." line take the rest.
CHUNK_TOKENS = 450
OVERLAP_TOKENS = 50
# Where a chunk may end, widest first: paragraphs, sentences, clauses, words. Tokens are the last resort.
BREAKS = (
    re.compile(r"\n[ \t]*\n"),
    re.compile(r"(?<=[.!?\u3002\uff01\uff1f])\s+"),
    re.compile(r"(?<=[,;:])\s+|\n"),
    re.compile(r"\s+"),
)
HEADER = re.compile(r"^(#+)[^\S\r\n](.*)")


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
    """Return subset of files that git would ignore. Uses git check-ignore, against the repository holding
    root, or against an empty scratch repository when there is none -- a folder that is not a checkout has
    its .gitignore read all the same."""
    if not files:
        return set()
    top = next((p for p in (root, *root.parents) if (p / ".git").exists()), None)
    scratch = None
    prefix = ["git"]
    if top is None:
        # The work tree starts at the highest folder in an unbroken chain of .gitignore files above root,
        # so indexing one subfolder of a folder tree still obeys the .gitignore at its top.
        base = root
        while base.parent != base and (base.parent / ".gitignore").exists():
            base = base.parent
        scratch = tempfile.mkdtemp()
        if subprocess.run(["git", "init", "--bare", "-q", scratch], capture_output=True).returncode != 0:
            shutil.rmtree(scratch, ignore_errors=True)
            return set()
        prefix = ["git", f"--git-dir={scratch}", f"--work-tree={base}"]
    try:
        result = subprocess.run(
            prefix + ["check-ignore", "--stdin", "-z"],
            input="\0".join(files),
            capture_output=True,
            text=True,
            cwd=top or root,
            timeout=30,
        )
        if result.returncode not in (0, 1):  # 1 = none ignored
            return set()
        return set(result.stdout.strip("\0").split("\0")) if result.stdout else set()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return set()
    finally:
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)


def _excluded(rel_path: str, patterns: tuple[str, ...]) -> bool:
    """Match --exclude patterns as .gitignore does its simple cases: a pattern holding a slash against the
    path from the index root, a pattern without one against any single name along it."""
    for pat in patterns:
        pat = pat.rstrip("/")
        if "/" in pat:
            if fnmatch(rel_path, pat) or fnmatch(rel_path, pat + "/*"):
                return True
        elif any(fnmatch(name, pat) for name in rel_path.split("/")):
            return True
    return False


def collect_files(
    root: Path, extensions: tuple[str, ...], root_only: bool = False, exclude: tuple[str, ...] = ()
) -> tuple[list[str], int]:
    """Collect files matching extensions, skipping ignored directories, excluded paths and gitignored files.
    If root_only, only files directly under root (no subdirectories).
    Returns (files, gitignored_count)."""
    files = []
    if root_only:
        try:
            for name in os.listdir(root):
                fp = os.path.join(root, name)
                if os.path.isfile(fp) and name.endswith(extensions) and not _excluded(name, exclude):
                    files.append(fp)
        except OSError:
            pass
    else:
        for dirpath, dirnames, filenames in os.walk(root):
            if "pyvenv.cfg" in filenames:  # a virtual environment, whatever it is called
                dirnames[:] = []
                continue
            rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
            rel_dir = "" if rel_dir == "." else rel_dir + "/"
            dirnames[:] = [
                d for d in dirnames
                if d not in SKIP_DIRS and not _excluded(rel_dir + d, exclude)
            ]
            for f in filenames:
                if f.endswith(extensions) and not _excluded(rel_dir + f, exclude):
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


def _embed(nodes: list[dict], embed_model, reuse: dict[str, np.ndarray], log=_log) -> np.ndarray:
    """float32 vectors for nodes: reused where a chunk's text is unchanged, computed for the rest."""
    if not nodes:
        return np.zeros((0, 0), dtype=np.float32)
    hashes = [_text_hash(n["text"]) for n in nodes]
    missing = [i for i, h in enumerate(hashes) if h not in reuse]
    log(f"Embeddings: {len(nodes) - len(missing)} reused, {len(missing)} to compute")
    rows = [reuse.get(h) for h in hashes]
    for start in range(0, len(missing), 1024):
        batch = missing[start:start + 1024]
        for i, vec in zip(batch, embed_model.embed_texts([nodes[i]["text"] for i in batch])):
            rows[i] = vec
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


def _pieces(text: str, start: int, end: int, starts: list[int], level: int = 0) -> list[tuple[int, int]]:
    """Consecutive spans covering text[start:end], each at most CHUNK_TOKENS, cut at the widest break that makes them fit."""
    tokens = bisect_left(starts, end) - bisect_left(starts, start)
    if tokens <= CHUNK_TOKENS:
        return [(start, end)]
    if level == len(BREAKS):
        first = bisect_left(starts, start)
        bounds = [start] + [starts[i] for i in range(first + CHUNK_TOKENS, first + tokens, CHUNK_TOKENS)] + [end]
        return list(zip(bounds, bounds[1:]))
    bounds = [start] + [m.end() for m in BREAKS[level].finditer(text, start, end) if start < m.end() < end] + [end]
    return [p for a, b in zip(bounds, bounds[1:]) for p in _pieces(text, a, b, starts, level + 1)]


def _split(text: str, start: int, end: int, starts: list[int]) -> list[tuple[int, int]]:
    """Chunks of text[start:end] of up to CHUNK_TOKENS, each beginning with the last OVERLAP_TOKENS or less of the one before."""
    size = lambda a, b: bisect_left(starts, b) - bisect_left(starts, a)  # noqa: E731
    chunks, current = [], []
    for a, b in _pieces(text, start, end, starts):
        if current and size(current[0][0], b) > CHUNK_TOKENS:
            chunks.append((current[0][0], current[-1][1]))
            tail = current[-1][1]
            current = [p for p in current if size(p[0], tail) <= OVERLAP_TOKENS]
            while current and size(current[0][0], b) > CHUNK_TOKENS:
                current.pop(0)
        current.append((a, b))
    if current:
        chunks.append((current[0][0], current[-1][1]))
    return chunks


def _markdown_sections(text: str) -> list[tuple[int, int, str]]:
    """(start, end, header path) per section: a new one at every header outside a code fence, the path naming the headers above it."""
    sections, stack, start, pos, fence = [], [], 0, 0, False
    for line in text.splitlines(keepends=True):
        match = None if fence else HEADER.match(line)
        if line.lstrip().startswith("```"):
            fence = not fence
        elif match:
            if text[start:pos].strip():
                sections.append((start, pos, "/" + "".join(f"{h}/" for _, h in stack[:-1])))
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2).strip()))
            start = pos
        pos += len(line)
    if text[start:pos].strip():
        sections.append((start, pos, "/" + "".join(f"{h}/" for _, h in stack[:-1])))
    return sections


def parse_files(file_paths: list[str], embed_model, log=_log, root: Path | None = None) -> list[dict]:
    """Chunks of the files as {"file", "start_line", "end_line", "text"}, the text headed by the file and its section or symbol."""
    from llm_index.ast_chunker import chunk_file

    nodes = []
    fallbacks = 0
    for fp in file_paths:
        language = PARSER_MAP.get(os.path.splitext(fp)[1].lower(), "text")
        rel = fp
        if root:
            try:
                rel = str(Path(fp).relative_to(root))
            except ValueError:
                pass

        if language in CODE_LANGUAGES:
            chunks = chunk_file(fp, language)
            for chunk in chunks:
                if chunk.text.strip():
                    context = f"symbol: {chunk.symbol}" if chunk.symbol else f"language: {language}"
                    nodes.append({"file": fp, "start_line": chunk.start_line, "end_line": chunk.end_line, "text": f"# file: {rel} | {context}\n{chunk.text}"})
            if chunks:
                continue
            fallbacks += 1

        try:
            with open(fp, encoding="utf-8", errors="replace") as f:
                text = _clean_text(f.read(), fp)
        except OSError:
            continue
        starts = embed_model.token_starts(text)
        newlines = [i for i, ch in enumerate(text) if ch == "\n"]
        if language == "markdown":
            spans = [(a, b, f"section: {path}") for s, e, path in _markdown_sections(text) for a, b in _split(text, s, e, starts)]
        else:
            spans = [(a, b, f"language: {language}") for a, b in _split(text, 0, len(text), starts)]
        for a, b, context in spans:
            body = text[a:b]
            a += len(body) - len(body.lstrip())
            b -= len(body) - len(body.rstrip())
            if a >= b:
                continue
            nodes.append({"file": fp, "start_line": bisect_right(newlines, a - 1) + 1, "end_line": bisect_right(newlines, b - 2) + 1, "text": f"# file: {rel} | {context}\n{text[a:b]}"})

    if fallbacks:
        log(f"  -> {fallbacks} code files split as text")
    log(f"  -> {len(nodes)} chunks from {len(file_paths)} files")
    return nodes


def build_index(
    workspace: Path,
    embed_model=None,
    log=_log,
    verbose: bool = False,
    force: bool = False,
    split: bool = False,
    parent_tags: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict:
    """Build vector index for a workspace. Returns stats dict.

    If split=True, also indexes each immediate subfolder as a separate child index
    tagged with folder:<name>. parent_tags is inherited by children.
    exclude is a list of patterns to leave out; None keeps the ones from the last index."""
    if exclude is None:
        exclude = (get_entry(str(workspace)) or {}).get("exclude", [])
    if split:
        return _build_index_split(
            workspace, embed_model, log, verbose, force, parent_tags, exclude
        )

    return _build_index_single(
        workspace, embed_model, log, verbose, force, root_only=False, exclude=exclude
    )


def _build_index_split(
    workspace: Path,
    embed_model,
    log,
    verbose: bool,
    force: bool,
    parent_tags: list[str] | None,
    exclude: list[str],
) -> dict:
    from llm_index.registry import register, set_tags, get_entry, drop_indexes

    entry = get_entry(str(workspace)) or {}
    skip = set(entry.get("skip", []))

    log(f"=== Root (root-level files only): {workspace} ===")
    root_result = _build_index_single(
        workspace, embed_model, log, verbose, force, root_only=True, exclude=exclude
    )
    if root_result.get("error"):
        shutil.rmtree(storage_dir(workspace), ignore_errors=True)
        root_result = {"files": 0, "nodes": 0, "elapsed": 0, "directory": str(workspace)}

    # Discover immediate subfolders
    subfolders: list[Path] = []
    try:
        for name in sorted(os.listdir(workspace)):
            sub = workspace / name
            if (
                sub.is_dir()
                and name not in SKIP_DIRS
                and not name.startswith(".")
                and name not in skip
                and not _excluded(name, tuple(exclude))
            ):
                subfolders.append(sub)
    except OSError:
        pass

    children: list[str] = []
    for sub in subfolders:
        log(f"\n=== Subfolder: {sub} ===")
        if str(sub) not in entry.get("children", []) and get_entry(str(sub)) is not None:
            log(f"  note: {sub} was already indexed independently -- overwriting")

        result = _build_index_single(
            sub, embed_model, log, verbose, force, root_only=False, exclude=exclude
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
    register(str(workspace), children=children, split=True, exclude=exclude)
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
    exclude: list[str] | None = None,
) -> dict:
    """Build a single vector index (no splitting). Returns stats dict."""
    start = time.time()

    all_files, gitignored = collect_files(
        workspace, DEFAULT_EXTENSIONS, root_only=root_only, exclude=tuple(exclude or ())
    )

    # Show per-extension counts
    ext_counts: dict[str, int] = {}
    for f in all_files:
        ext = os.path.splitext(f)[1].lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    summary = ", ".join(f"{count} {ext}" for ext, count in sorted(ext_counts.items()))
    log(f"Found {len(all_files)} files ({summary})")
    if gitignored:
        log(f"Excluded {gitignored} file(s) via .gitignore")
    if exclude:
        log(f"Excluding: {', '.join(exclude)}")

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

    if files_to_parse and embed_model is None:
        from llm_index.model import shared_embedder

        embed_model = shared_embedder(log)
    new_nodes = parse_files(files_to_parse, embed_model, log=log, root=workspace) if files_to_parse else []
    new_vectors = _embed(new_nodes, embed_model, reuse, log)
    reuse = {}

    kept = np.flatnonzero(keep)
    chunks = {
        "files": [chunks["files"][i] for i in kept] + [n["file"] for n in new_nodes],
        "start_lines": [chunks["start_lines"][i] for i in kept] + [n["start_line"] for n in new_nodes],
        "end_lines": [chunks["end_lines"][i] for i in kept] + [n["end_line"] for n in new_nodes],
        "texts": [chunks["texts"][i] for i in kept] + [n["text"] for n in new_nodes],
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

    register(str(workspace), exclude=exclude)

    return {
        "files": len(all_files),
        "nodes": len(new_nodes),
        "elapsed": round(elapsed, 1),
        "directory": str(workspace),
    }
