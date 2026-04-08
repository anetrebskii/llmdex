#!/usr/bin/env python3
"""Query indexed project files via the persistent server."""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

from llm_index.server import (
    get_running_server,
    get_version,
    health_check,
    stop_server,
)


def _start_new_server() -> int:
    """Launch server subprocess and wait for it to come up. Returns port."""
    subprocess.Popen(
        [sys.executable, "-m", "llm_index.server", "--serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    for _ in range(60):
        time.sleep(0.5)
        running = get_running_server()
        if running:
            _, port, _ = running
            return port

    print("Failed to start server")
    sys.exit(1)


def ensure_server() -> int:
    """Start server if not running, restart if stale or version mismatch. Returns port."""
    running = get_running_server()

    if running:
        pid, port, ver = running
        current_ver = get_version()

        # Check version mismatch — restart if updated
        if ver != current_ver:
            print(
                f"Server version mismatch (running: {ver}, installed: {current_ver}), restarting..."
            )
            stop_server(pid, port)
            return _start_new_server()

        # Verify server is actually responding
        if health_check(port):
            return port

        # PID exists but server not responding — kill and restart
        print("Server not responding, restarting...")
        stop_server(pid, port)
        return _start_new_server()

    return _start_new_server()


def _make_loc(source: str, start_line: int | None, end_line: int | None) -> str:
    loc = source
    if start_line and end_line:
        loc += f":{start_line}-{end_line}"
    elif start_line:
        loc += f":{start_line}"
    return loc


def query_server(port: int, question: str, top_k: int, directory: str | None = None, folder: str | None = None, tags: list[str] | None = None, compact: bool = False):
    payload = {"question": question, "top_k": top_k}
    if directory is not None:
        payload["directory"] = str(Path(directory).resolve())
    if folder is not None:
        payload["folder"] = folder
    if tags is not None:
        payload["tags"] = tags
    # else: server searches all registered indexes

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/query",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read())
        except Exception:
            print(f"Server error: {e}")
            sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Server error: {e}")
        sys.exit(1)

    if "error" in data:
        print(f"Error: {data['error']}")
        sys.exit(1)

    results = data["results"]

    if compact:
        # Minimal output for AI/automation: one line per result, no preview
        for item in results:
            loc = _make_loc(item["source"], item.get("start_line"), item.get("end_line"))
            print(f"[{item['score']:.3f}] {loc}")
        return

    print(f"Query: {question}")
    print(f"Top {len(results)} results:\n")

    for i, item in enumerate(results, 1):
        loc = _make_loc(item["source"], item.get("start_line"), item.get("end_line"))
        print(f"{i}. [{item['score']:.3f}] {loc}")
        # Show full chunk with line numbers
        start = item.get("start_line", 1)
        lines = item["text"].split("\n")
        for j, line in enumerate(lines):
            print(f"   {start + j}\t{line}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Search indexed project files")
    parser.add_argument("question", help="Search query")
    parser.add_argument(
        "-d",
        "--directory",
        default=".",
        help="Project directory to search (default: current dir)",
    )
    parser.add_argument(
        "-a", "--all", action="store_true", help="Search across all indexed projects"
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=5, help="Number of results (default: 5)"
    )
    parser.add_argument(
        "-f", "--folder", help="Filter results to files under this folder prefix"
    )
    args = parser.parse_args()

    port = ensure_server()
    directory = None if args.all else args.directory
    query_server(port, args.question, args.top_k, directory, args.folder)


if __name__ == "__main__":
    main()
