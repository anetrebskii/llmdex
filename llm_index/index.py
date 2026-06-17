#!/usr/bin/env python3
"""CLI for indexing project files."""

import argparse
import sys
from pathlib import Path

# Force unbuffered output so progress is visible in real time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def cmd_index(args):
    from llm_index.indexer import build_index

    workspace = Path(args.directory).resolve()
    if not workspace.is_dir():
        print(f"Error: {workspace} is not a directory")
        sys.exit(1)

    print(f"Indexing: {workspace}")
    result = build_index(workspace)

    if result.get("error"):
        print(result["error"])
        sys.exit(1)


def cmd_reindex(args):
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

        print(f"--- {directory} ---")
        result = build_index(workspace)
        if result.get("error"):
            print(f"  Error: {result['error']}")
        print()


def cmd_list(args):
    from llm_index.registry import list_registered
    from llm_index.indexer import storage_dir

    entries = list_registered()
    if not entries:
        print("No indexed folders.")
        return

    for directory, meta in entries.items():
        store = storage_dir(Path(directory))
        exists = store.exists()
        status = "ok" if exists else "missing index"
        indexed_at = meta.get("indexed_at", "unknown")
        print(f"  {directory}")
        print(f"    indexed at: {indexed_at}")
        print(f"    status: {status}")
        print()


def cmd_remove(args):
    from llm_index.registry import unregister

    directory = str(Path(args.directory).resolve())
    if unregister(directory):
        print(f"Removed: {directory}")
    else:
        print(f"Not found in registry: {directory}")


def main():
    # Manual dispatch: check if first arg is a subcommand
    commands = {"reindex", "list", "remove"}
    if len(sys.argv) > 1 and sys.argv[1] in commands:
        cmd = sys.argv[1]
        if cmd == "reindex":
            cmd_reindex(None)
        elif cmd == "list":
            cmd_list(None)
        elif cmd == "remove":
            if len(sys.argv) < 3:
                print("Usage: llmdex-index remove <directory>")
                sys.exit(1)
            args = argparse.Namespace(directory=sys.argv[2])
            cmd_remove(args)
    else:
        parser = argparse.ArgumentParser(
            description="llmdex - index project files for semantic search"
        )
        parser.add_argument(
            "directory", nargs="?", default=".", help="Project directory to index"
        )
        args = parser.parse_args()
        cmd_index(args)


if __name__ == "__main__":
    main()
