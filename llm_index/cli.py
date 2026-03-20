#!/usr/bin/env python3
"""Unified CLI for llmdex."""

import argparse
import sys

# Force unbuffered output so progress is visible in real time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def cmd_index(args):
    from pathlib import Path
    from llm_index.indexer import build_index

    workspace = Path(args.directory).resolve()
    if not workspace.is_dir():
        print(f"Error: {workspace} is not a directory")
        sys.exit(1)

    extensions = tuple(
        ext if ext.startswith(".") else f".{ext}" for ext in args.extensions
    )

    print(f"Indexing: {workspace}")
    print(f"Extensions: {', '.join(extensions)}")
    result = build_index(workspace, extensions)

    if result.get("error"):
        print(result["error"])
        sys.exit(1)


def cmd_reindex(args):
    from pathlib import Path
    from llm_index.registry import list_registered
    from llm_index.indexer import build_index

    entries = list_registered()
    if not entries:
        print("No indexed folders found. Run: llmdex index <directory>")
        return

    print(f"Re-indexing {len(entries)} folder(s)...\n")
    for directory, meta in entries.items():
        workspace = Path(directory)
        if not workspace.is_dir():
            print(f"Skipping (not found): {directory}")
            continue

        extensions = tuple(meta.get("extensions", [".md", ".ts", ".json"]))
        print(f"--- {directory} ({', '.join(extensions)}) ---")
        result = build_index(workspace, extensions)
        if result.get("error"):
            print(f"  Error: {result['error']}")
        print()


def cmd_list(args):
    from pathlib import Path
    from llm_index.registry import list_registered
    from llm_index.indexer import storage_dir

    entries = list_registered()
    if not entries:
        print("No indexed folders.")
        return

    for directory, meta in entries.items():
        extensions = ", ".join(meta.get("extensions", []))
        store = storage_dir(Path(directory))
        exists = store.exists()
        status = "ok" if exists else "missing index"
        indexed_at = meta.get("indexed_at", "unknown")
        print(f"  {directory}")
        print(f"    extensions: {extensions}")
        print(f"    indexed at: {indexed_at}")
        print(f"    status: {status}")
        print()


def cmd_remove(args):
    from pathlib import Path
    from llm_index.registry import unregister

    directory = str(Path(args.directory).resolve())
    if unregister(directory):
        print(f"Removed: {directory}")
    else:
        print(f"Not found in registry: {directory}")


def cmd_query(args):
    from llm_index.query import ensure_server, query_server

    port = ensure_server()
    directory = None if args.all else args.directory
    query_server(port, args.question, args.top_k, directory)


def cmd_server(args):
    from llm_index.server import (
        get_running_server,
        get_version,
        health_check,
        stop_server,
        start_server,
        _launch_background,
    )

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


def main():
    from importlib.metadata import version as pkg_version, PackageNotFoundError

    try:
        ver = pkg_version("llmdex")
    except PackageNotFoundError:
        ver = "dev"

    parser = argparse.ArgumentParser(
        prog="llmdex",
        description="Local semantic search for your projects",
    )
    parser.add_argument("-v", "--version", action="version", version=f"llmdex {ver}")
    sub = parser.add_subparsers(dest="command")

    # llmdex index / i
    p_index = sub.add_parser("index", aliases=["idx"], help="Index a project directory")
    p_index.add_argument(
        "directory", nargs="?", default=".", help="Project directory (default: .)"
    )
    p_index.add_argument(
        "-e",
        "--extensions",
        nargs="+",
        default=[".md", ".ts", ".json"],
        help="File extensions to index (default: .md .ts .json)",
    )

    # llmdex reindex / re
    sub.add_parser("reindex", aliases=["re"], help="Re-index all registered projects")

    # llmdex list / ls
    sub.add_parser("list", aliases=["ls"], help="List all indexed projects")

    # llmdex remove / rm
    p_remove = sub.add_parser("remove", aliases=["rm"], help="Remove a project from the registry")
    p_remove.add_argument("directory", help="Project directory to remove")

    # llmdex query / q
    p_query = sub.add_parser("query", aliases=["q"], help="Search indexed project files")
    p_query.add_argument("question", help="Search query")
    p_query.add_argument(
        "-d",
        "--directory",
        default=".",
        help="Project directory to search (default: .)",
    )
    p_query.add_argument(
        "-a", "--all", action="store_true", help="Search across all indexed projects"
    )
    p_query.add_argument(
        "-k", "--top-k", type=int, default=5, help="Number of results (default: 5)"
    )

    # llmdex server / srv
    p_server = sub.add_parser("server", aliases=["s"], help="Manage the background server")
    p_server.add_argument(
        "-p", "--port", type=int, default=7392, help="Port (default: 7392)"
    )
    p_server.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=1800,
        help="Inactivity timeout in seconds (default: 1800)",
    )
    p_server.add_argument("--stop", action="store_true", help="Stop running server")
    p_server.add_argument(
        "--restart", action="store_true", help="Restart running server"
    )
    p_server.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "index": cmd_index,
        "idx": cmd_index,
        "reindex": cmd_reindex,
        "re": cmd_reindex,
        "list": cmd_list,
        "ls": cmd_list,
        "remove": cmd_remove,
        "rm": cmd_remove,
        "query": cmd_query,
        "q": cmd_query,
        "server": cmd_server,
        "s": cmd_server,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
