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

from llm_index.server import get_running_server


def ensure_server() -> int:
    """Start server if not running, return port."""
    running = get_running_server()
    if running:
        return running[1]

    # Start server in background
    subprocess.Popen(
        [sys.executable, "-m", "llm_index.server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Wait for it to come up
    for _ in range(60):
        time.sleep(0.5)
        running = get_running_server()
        if running:
            return running[1]

    print("Failed to start server")
    sys.exit(1)


def query_server(port: int, question: str, top_k: int, directory: str | None = None):
    payload = {"question": question, "top_k": top_k}
    if directory is not None:
        payload["directory"] = str(Path(directory).resolve())
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
    print(f"\nQuery: {question}")
    print(f"Top {len(results)} results:\n")

    for i, item in enumerate(results, 1):
        print(f"{i}. [{item['score']:.3f}] {item['source']}")
        preview = item["text"][:200].replace("\n", " ")
        print(f"   {preview}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="Search indexed project files")
    parser.add_argument("question", help="Search query")
    parser.add_argument("-d", "--directory", default=".", help="Project directory to search (default: current dir)")
    parser.add_argument("-a", "--all", action="store_true", help="Search across all indexed projects")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5)")
    args = parser.parse_args()

    port = ensure_server()
    directory = None if args.all else args.directory
    query_server(port, args.question, args.top_k, directory)


if __name__ == "__main__":
    main()
