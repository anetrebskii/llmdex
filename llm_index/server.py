#!/usr/bin/env python3
"""Persistent query server with auto-shutdown after inactivity."""

import argparse
import json
import os
import signal
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llm_index.indexer import storage_dir

INACTIVITY_TIMEOUT = 1800  # 30 minutes
PID_DIR = Path.home() / ".llmdex"
DEFAULT_PORT = 7392


def pid_file() -> Path:
    return PID_DIR / "server.pid"


class IndexCache:
    """Lazy-loads and caches embed model + per-workspace indexes."""

    def __init__(self):
        self.embed_model = None
        self.indexes: dict[str, object] = {}
        self.lock = threading.Lock()

    def get_embed_model(self):
        if self.embed_model is None:
            self.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
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
                    raise FileNotFoundError(f"No index at {store}. Run: llmdex-index {workspace}")
                ctx = StorageContext.from_defaults(persist_dir=str(store))
                self.indexes[key] = load_index_from_storage(ctx)
            return self.indexes[key]

    def invalidate(self, workspace: Path):
        key = str(workspace)
        with self.lock:
            self.indexes.pop(key, None)


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

        extensions = body.get("extensions", [".md", ".ts", ".json"])
        extensions = tuple(ext if ext.startswith(".") else f".{ext}" for ext in extensions)

        # Import here to avoid circular import at module level
        from llm_index.indexer import build_index

        logs = []
        try:
            result = build_index(
                workspace,
                extensions=extensions,
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

    def _query_single(self, workspace: Path, question: str, top_k: int) -> list[dict]:
        """Query a single workspace index, return list of result dicts."""
        try:
            index = cache.get_index(workspace)
        except FileNotFoundError:
            return []

        retriever = index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(question)

        items = []
        for node in results:
            source = node.metadata.get("file_path", "unknown")
            items.append({
                "score": round(node.score, 4),
                "source": source,
                "text": node.text[:500],
            })
        return items

    def _handle_query(self):
        body = self._read_body()
        if not body:
            return

        directory = body.get("directory")
        question = body.get("question", "")
        top_k = body.get("top_k", 5)
        search_all = body.get("all", directory is None)

        if not question:
            self._json_response(400, {"error": "missing 'question'"})
            return

        if search_all:
            # Search across all registered indexes
            from llm_index.registry import list_registered
            entries = list_registered()
            if not entries:
                self._json_response(404, {"error": "No indexed folders. Run: llmdex-index <directory>"})
                return

            all_items = []
            for dir_path in entries:
                all_items.extend(self._query_single(Path(dir_path), question, top_k))

            # Sort by score descending, take top_k
            all_items.sort(key=lambda x: x["score"], reverse=True)
            self._json_response(200, {"results": all_items[:top_k]})
        else:
            workspace = Path(directory).resolve()
            items = self._query_single(workspace, question, top_k)
            if not items:
                from llm_index.indexer import storage_dir
                store = storage_dir(workspace)
                if not store.exists():
                    self._json_response(404, {"error": f"No index at {store}. Run: llmdex-index {workspace}"})
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

            extensions = tuple(meta.get("extensions", [".md", ".ts", ".json"]))
            logs = []
            try:
                result = build_index(
                    workspace,
                    extensions=extensions,
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
            items.append({
                "directory": directory,
                "extensions": meta.get("extensions", []),
                "indexed_at": meta.get("indexed_at", "unknown"),
                "has_index": store.exists(),
            })
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

    server = HTTPServer(("127.0.0.1", port), QueryHandler)

    # Write PID file
    pf = pid_file()
    pf.write_text(f"{os.getpid()}\n{port}")

    def cleanup(*_):
        pf.unlink(missing_ok=True)
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    # Start watchdog
    watchdog = threading.Thread(target=inactivity_watchdog, args=(timeout,), daemon=True)
    watchdog.start()

    print(f"llmdex server running on http://127.0.0.1:{port}")
    print(f"Auto-shutdown after {timeout}s of inactivity")
    server.serve_forever()


def get_running_server() -> tuple[int, int] | None:
    """Returns (pid, port) if server is running, None otherwise."""
    pf = pid_file()
    if not pf.exists():
        return None
    try:
        lines = pf.read_text().strip().split("\n")
        pid, port = int(lines[0]), int(lines[1])
        os.kill(pid, 0)  # check if process exists
        return pid, port
    except (OSError, ValueError, IndexError):
        pf.unlink(missing_ok=True)
        return None


def main():
    parser = argparse.ArgumentParser(description="llmdex persistent server")
    parser.add_argument("-p", "--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")
    parser.add_argument("-t", "--timeout", type=int, default=INACTIVITY_TIMEOUT, help="Inactivity timeout in seconds (default: 600)")
    parser.add_argument("--stop", action="store_true", help="Stop running server")
    args = parser.parse_args()

    if args.stop:
        running = get_running_server()
        if running:
            pid, port = running
            os.kill(pid, signal.SIGTERM)
            print(f"Stopped server (pid {pid}, port {port})")
        else:
            print("No server running")
        return

    running = get_running_server()
    if running:
        pid, port = running
        print(f"Server already running (pid {pid}, port {port})")
        return

    start_server(args.port, args.timeout)


if __name__ == "__main__":
    main()
