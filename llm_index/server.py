#!/usr/bin/env python3
"""Persistent query server with auto-shutdown after inactivity."""

import argparse
import json
import os
import re
import signal
import sys
import threading
import time
import urllib.request
from collections import OrderedDict
from http.server import HTTPServer, BaseHTTPRequestHandler
from importlib.metadata import version as pkg_version, PackageNotFoundError
from pathlib import Path

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
)

from llm_index.indexer import storage_dir, EMBED_MODEL_NAME, make_hf_embedding

INACTIVITY_TIMEOUT = 1800  # 30 minutes
PID_DIR = Path.home() / ".llmdex"
DEFAULT_PORT = 7392
# Budget is measured against the index's on-disk size; in RAM it inflates roughly 2-4x once
# chunks, embeddings and BM25 tables are Python objects. Without it an `-a` query loads all
# 16 GB of registered indexes and never gives the memory back.
MAX_CACHED_MB = max(1, int(os.environ.get("LLMDEX_MAX_CACHED_MB", "1024")))


def get_version() -> str:
    """Return installed package version."""
    try:
        return pkg_version("llmdex")
    except PackageNotFoundError:
        return "dev"


def pid_file() -> Path:
    return PID_DIR / "server.pid"


def _tokenize_code(text: str) -> list[str]:
    """Tokenize text for BM25, splitting on code boundaries."""
    # Split camelCase and PascalCase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Extract alphanumeric tokens (keeps underscored identifiers)
    return re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+', text.lower())


class IndexCache:
    """Lazy-loads and caches embed model + per-workspace indexes."""

    def __init__(self):
        self.embed_model = None
        self.indexes: OrderedDict[str, object] = OrderedDict()
        self.bm25_data: OrderedDict[str, tuple] = OrderedDict()  # key -> (BM25Okapi, node_list)
        self.sizes: dict[str, int] = {}  # key -> on-disk bytes, the eviction cost proxy
        self.lock = threading.RLock()

    def get_embed_model(self):
        if self.embed_model is None:
            self.embed_model = make_hf_embedding()
            Settings.embed_model = self.embed_model
            Settings.llm = None
        return self.embed_model

    def get_index(self, workspace: Path):
        key = str(workspace)
        with self.lock:
            if key not in self.indexes:
                self.get_embed_model()
                store = storage_dir(workspace)
                if not store.exists():
                    raise FileNotFoundError(
                        f"No index at {store}. Run: llmdex index {workspace}"
                    )
                # Check for embedding model mismatch
                from llm_index.registry import get_entry
                entry = get_entry(key)
                if entry:
                    stored_model = entry.get("embed_model", "all-MiniLM-L6-v2")
                    if stored_model != EMBED_MODEL_NAME:
                        raise ValueError(
                            f"Index was built with '{stored_model}' "
                            f"but current model is '{EMBED_MODEL_NAME}'. "
                            f"Run: llmdex reindex"
                        )
                ctx = StorageContext.from_defaults(persist_dir=str(store))
                self.indexes[key] = load_index_from_storage(ctx)
                self.sizes[key] = sum(
                    f.stat().st_size for f in store.rglob("*") if f.is_file()
                )
            self.indexes.move_to_end(key)
            self._evict()
            return self.indexes[key]

    def get_bm25(self, workspace: Path):
        key = str(workspace)
        with self.lock:
            if key not in self.bm25_data:
                index = self.get_index(workspace)
                all_nodes = list(index.docstore.docs.values())
                corpus = [_tokenize_code(node.get_content()) for node in all_nodes]
                from rank_bm25 import BM25Okapi
                bm25 = BM25Okapi(corpus)
                self.bm25_data[key] = (bm25, all_nodes)
            return self.bm25_data[key]

    def _evict(self):
        budget = MAX_CACHED_MB * 1024 * 1024
        total = sum(self.sizes.values())
        while total > budget and len(self.indexes) > 1:
            evicted, _ = self.indexes.popitem(last=False)
            self.bm25_data.pop(evicted, None)
            total -= self.sizes.pop(evicted, 0)

    def invalidate(self, workspace: Path):
        key = str(workspace)
        with self.lock:
            self.indexes.pop(key, None)
            self.bm25_data.pop(key, None)
            self.sizes.pop(key, None)


cache = IndexCache()
last_activity = time.time()
activity_lock = threading.Lock()


def touch_activity():
    global last_activity
    with activity_lock:
        last_activity = time.time()


def inactivity_watchdog(timeout: int):
    while True:
        time.sleep(30)
        with activity_lock:
            idle = time.time() - last_activity
        if idle >= timeout:
            print(f"Idle for {timeout}s, shutting down.")
            pid_file().unlink(missing_ok=True)
            os.kill(os.getpid(), signal.SIGTERM)
            return


class QueryHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence default logging

    def do_POST(self):
        touch_activity()

        if self.path == "/query":
            self._handle_query()
        elif self.path == "/index":
            self._handle_index()
        elif self.path == "/reindex":
            self._handle_reindex()
        elif self.path == "/invalidate":
            self._handle_invalidate()
        elif self.path == "/remove":
            self._handle_remove()
        elif self.path == "/health":
            self._json_response(200, {"status": "ok"})
        else:
            self._json_response(404, {"error": "not found"})

    def do_GET(self):
        touch_activity()
        if self.path == "/health":
            self._json_response(200, {"status": "ok"})
        elif self.path == "/list":
            self._handle_list()
        else:
            self._json_response(404, {"error": "not found"})

    def _handle_index(self):
        body = self._read_body()
        if not body:
            return

        directory = body.get("directory")
        if not directory:
            self._json_response(400, {"error": "missing 'directory'"})
            return

        workspace = Path(directory).resolve()
        if not workspace.is_dir():
            self._json_response(400, {"error": f"'{workspace}' is not a directory"})
            return

        # Import here to avoid circular import at module level
        from llm_index.indexer import build_index

        logs = []
        try:
            result = build_index(
                workspace,
                embed_model=cache.get_embed_model(),
                log=lambda msg: logs.append(msg),
            )
        except Exception as e:
            self._json_response(500, {"error": str(e), "logs": logs})
            return

        # Invalidate cached index so next query loads the fresh one
        cache.invalidate(workspace)

        result["logs"] = logs
        self._json_response(200, result)

    def _query_single(
        self, workspace: Path, question: str, top_k: int, folder: str | None = None
    ) -> list[dict]:
        """Query a single workspace index, return list of result dicts."""
        try:
            index = cache.get_index(workspace)
        except (FileNotFoundError, ValueError):
            return []

        fetch_k = top_k * 3
        if folder:
            fetch_k *= 2

        # Vector retrieval
        retriever = index.as_retriever(similarity_top_k=fetch_k)
        vector_results = retriever.retrieve(question)

        # BM25 retrieval
        bm25_results = []
        try:
            bm25, all_nodes = cache.get_bm25(workspace)
            query_tokens = _tokenize_code(question)
            scores = bm25.get_scores(query_tokens)
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:fetch_k]
            bm25_results = [
                (all_nodes[i], scores[i]) for i in top_indices if scores[i] > 0
            ]
        except Exception:
            pass  # fall back to vector-only

        # Reciprocal Rank Fusion
        RRF_K = 60
        fused_scores: dict[str, float] = {}
        node_map: dict[str, object] = {}  # node_id -> node or NodeWithScore

        for rank, nws in enumerate(vector_results):
            nid = nws.node.node_id
            fused_scores[nid] = fused_scores.get(nid, 0) + 1.0 / (RRF_K + rank + 1)
            node_map[nid] = nws

        for rank, (node, _bm25_score) in enumerate(bm25_results):
            nid = node.node_id
            fused_scores[nid] = fused_scores.get(nid, 0) + 1.0 / (RRF_K + rank + 1)
            if nid not in node_map:
                node_map[nid] = node

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        items = []
        for nid, score in ranked:
            entry = node_map[nid]
            # NodeWithScore (vector) vs raw BaseNode (BM25-only)
            if hasattr(entry, "node"):
                metadata = entry.node.metadata
                text = entry.text
            else:
                metadata = entry.metadata
                text = entry.get_content()

            source = metadata.get("file_path", "unknown")
            if folder and not source.startswith(folder):
                continue

            item = {
                "score": round(score, 4),
                "source": source,
                "text": text,
            }
            start_line = metadata.get("start_line")
            end_line = metadata.get("end_line")
            if start_line is not None:
                item["start_line"] = start_line
            if end_line is not None:
                item["end_line"] = end_line
            items.append(item)
            if len(items) >= top_k:
                break
        return items

    def _handle_query(self):
        body = self._read_body()
        if not body:
            return

        directory = body.get("directory")
        question = body.get("question", "")
        top_k = body.get("top_k", 5)
        folder = body.get("folder")
        tags = body.get("tags")
        search_all = body.get("all", directory is None and tags is None)

        if not question:
            self._json_response(400, {"error": "missing 'question'"})
            return

        if tags:
            # Search only indexes matching all given tags
            from llm_index.registry import find_by_tags

            entries = find_by_tags(tags)
            if not entries:
                self._json_response(
                    404, {"error": f"No indexes found with tags: {', '.join(tags)}"}
                )
                return

            all_items = []
            for dir_path in entries:
                all_items.extend(self._query_single(Path(dir_path), question, top_k, folder))

            all_items.sort(key=lambda x: x["score"], reverse=True)
            self._json_response(200, {"results": all_items[:top_k]})
        elif search_all:
            # Search across all registered indexes
            from llm_index.registry import list_registered

            entries = list_registered()
            if not entries:
                self._json_response(
                    404, {"error": "No indexed folders. Run: llmdex index <directory>"}
                )
                return

            all_items = []
            for dir_path in entries:
                all_items.extend(self._query_single(Path(dir_path), question, top_k, folder))

            # Sort by score descending, take top_k
            all_items.sort(key=lambda x: x["score"], reverse=True)
            self._json_response(200, {"results": all_items[:top_k]})
        else:
            workspace = Path(directory).resolve()
            items = self._query_single(workspace, question, top_k, folder)
            if not items:
                from llm_index.indexer import storage_dir

                store = storage_dir(workspace)
                if not store.exists():
                    self._json_response(
                        404,
                        {
                            "error": f"No index at {store}. Run: llmdex index {workspace}"
                        },
                    )
                    return
            self._json_response(200, {"results": items})

    def _handle_invalidate(self):
        body = self._read_body()
        if not body:
            return
        workspace = Path(body.get("directory", ".")).resolve()
        cache.invalidate(workspace)
        self._json_response(200, {"status": "invalidated", "directory": str(workspace)})

    def _handle_reindex(self):
        from llm_index.registry import list_registered
        from llm_index.indexer import build_index

        entries = list_registered()
        if not entries:
            self._json_response(200, {"status": "nothing to reindex", "results": []})
            return

        results = []
        for directory, meta in entries.items():
            workspace = Path(directory)
            if not workspace.is_dir():
                results.append({"directory": directory, "error": "directory not found"})
                continue

            logs = []
            try:
                result = build_index(
                    workspace,
                    embed_model=cache.get_embed_model(),
                    log=lambda msg: logs.append(msg),
                )
            except Exception as e:
                results.append({"directory": directory, "error": str(e), "logs": logs})
                continue

            cache.invalidate(workspace)
            result["logs"] = logs
            results.append(result)

        self._json_response(200, {"status": "done", "results": results})

    def _handle_list(self):
        from llm_index.registry import list_registered
        from llm_index.indexer import storage_dir

        entries = list_registered()
        items = []
        for directory, meta in entries.items():
            store = storage_dir(Path(directory))
            items.append(
                {
                    "directory": directory,
                    "indexed_at": meta.get("indexed_at", "unknown"),
                    "tags": meta.get("tags", []),
                    "has_index": store.exists(),
                }
            )
        self._json_response(200, {"folders": items})

    def _handle_remove(self):
        body = self._read_body()
        if not body:
            return

        directory = body.get("directory")
        if not directory:
            self._json_response(400, {"error": "missing 'directory'"})
            return

        directory = str(Path(directory).resolve())

        from llm_index.registry import unregister

        cache.invalidate(Path(directory))
        if unregister(directory):
            self._json_response(200, {"status": "removed", "directory": directory})
        else:
            self._json_response(404, {"error": f"not found in registry: {directory}"})

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._json_response(400, {"error": "empty body"})
            return None
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._json_response(400, {"error": "invalid JSON"})
            return None

    def _json_response(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_server(port: int = DEFAULT_PORT, timeout: int = INACTIVITY_TIMEOUT):
    PID_DIR.mkdir(parents=True, exist_ok=True)

    try:
        server = HTTPServer(("127.0.0.1", port), QueryHandler)
    except OSError as e:
        if e.errno in (48, 98):  # 48=macOS, 98=Linux: Address already in use
            print(f"Port {port} is occupied, attempting to free it...")
            _force_free_port(port)
            server = HTTPServer(("127.0.0.1", port), QueryHandler)
        else:
            raise

    # Write PID file with version
    pf = pid_file()
    pf.write_text(f"{os.getpid()}\n{port}\n{get_version()}", encoding="utf-8")

    def cleanup(*_):
        pf.unlink(missing_ok=True)
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    # Start watchdog
    watchdog = threading.Thread(
        target=inactivity_watchdog, args=(timeout,), daemon=True
    )
    watchdog.start()

    ver = get_version()
    print(f"llmdex server v{ver} running on http://127.0.0.1:{port}")
    print(f"Auto-shutdown after {timeout}s of inactivity")
    server.serve_forever()


def _detached_spawn(cmd: list[str]):
    """Popen a background process that survives the launching process's exit.

    On Windows, harnesses like Claude Code run commands inside a Job Object with
    kill-on-close; a plainly spawned child is killed when the launching `llmdex`
    process exits. Detach it and break it out of the job (falling back to a plain
    detached spawn if the job forbids breakaway). `start_new_session` is a no-op
    on Windows, so it only helps on POSIX.
    """
    import subprocess

    common = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if sys.platform == "win32":
        detached = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            return subprocess.Popen(
                cmd,
                creationflags=detached | subprocess.CREATE_BREAKAWAY_FROM_JOB,
                **common,
            )
        except OSError:
            return subprocess.Popen(cmd, creationflags=detached, **common)
    return subprocess.Popen(cmd, start_new_session=True, **common)


def get_running_server() -> tuple[int, int, str] | None:
    """Returns (pid, port, version) if server is running, None otherwise."""
    pf = pid_file()
    if not pf.exists():
        return None
    try:
        lines = pf.read_text(encoding="utf-8").strip().split("\n")
        pid, port = int(lines[0]), int(lines[1])
        ver = lines[2] if len(lines) > 2 else "unknown"
        os.kill(pid, 0)  # check if process exists
        return pid, port, ver
    except (OSError, ValueError, IndexError):
        pf.unlink(missing_ok=True)
        return None


def health_check(port: int, timeout: float = 3.0) -> bool:
    """Check if server is actually responding on the given port."""
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except Exception:
        return False


def stop_server(pid: int, port: int, wait: float = 5.0) -> bool:
    """Stop server gracefully (SIGTERM), then forcefully (SIGKILL) if needed.
    Returns True if server was stopped."""
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        # Process already dead
        pid_file().unlink(missing_ok=True)
        return True

    # Wait for graceful shutdown
    deadline = time.time() + wait
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
            time.sleep(0.2)
        except OSError:
            # Process is gone
            pid_file().unlink(missing_ok=True)
            return True

    # Force kill (no SIGKILL on Windows; os.kill there is already a hard TerminateProcess)
    try:
        os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        time.sleep(0.5)
    except OSError:
        pass

    pid_file().unlink(missing_ok=True)
    return True


def _force_free_port(port: int):
    """Find and kill whatever process is holding the port."""
    import subprocess

    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.stdout.strip():
            for pid_str in result.stdout.strip().split("\n"):
                pid = int(pid_str.strip())
                if pid != os.getpid():
                    print(f"Killing process {pid} occupying port {port}")
                    os.kill(pid, signal.SIGKILL)
            time.sleep(1)
    except Exception:
        pass

    pid_file().unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="llmdex persistent server")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=INACTIVITY_TIMEOUT,
        help="Inactivity timeout in seconds (default: 600)",
    )
    parser.add_argument("--stop", action="store_true", help="Stop running server")
    parser.add_argument("--restart", action="store_true", help="Restart running server")
    parser.add_argument(
        "--serve",
        action="store_true",
        help=argparse.SUPPRESS,  # internal: run in foreground
    )
    args = parser.parse_args()

    if args.serve:
        start_server(args.port, args.timeout)
        return

    if args.stop:
        running = get_running_server()
        if running:
            pid, port, ver = running
            stop_server(pid, port)
            print(f"Stopped server v{ver} (pid {pid}, port {port})")
        else:
            print("No server running")
        return

    running = get_running_server()
    if running:
        pid, port, ver = running

        current_ver = get_version()
        version_ok = ver == current_ver

        if args.restart or not version_ok:
            reason = "version mismatch" if not version_ok else "restart requested"
            print(
                f"Restarting server ({reason}: running v{ver}, installed v{current_ver})..."
            )
            stop_server(pid, port)
        else:
            healthy = health_check(port)
            status = "healthy" if healthy else "not responding"
            print(f"Server already running v{ver} (pid {pid}, port {port}, {status})")
            if not healthy:
                print("Hint: run `llmdex server --restart` to restart")
            return

    _launch_background(args.port, args.timeout)


def _launch_background(port: int, timeout: int):
    """Start server as a background process, wait for it, print status."""
    _detached_spawn(
        [
            sys.executable,
            "-m",
            "llm_index.server",
            "--serve",
            "-p",
            str(port),
            "-t",
            str(timeout),
        ]
    )

    for _ in range(60):
        time.sleep(0.5)
        running = get_running_server()
        if running:
            pid, rport, ver = running
            print(f"Server v{ver} started (pid {pid}, port {rport})")
            return

    print("Failed to start server")
    sys.exit(1)


if __name__ == "__main__":
    main()
