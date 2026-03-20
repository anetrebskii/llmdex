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

    extensions = tuple(ext if ext.startswith(".") else f".{ext}" for ext in args.extensions)

    print(f"Indexing: {workspace}")
    print(f"Extensions: {', '.join(extensions)}")
    result = build_index(workspace, extensions)

    if result.get("error"):
        print(result["error"])
        sys.exit(1)


def cmd_reindex(args):
    from llm_index.registry import list_registered
    from llm_index.indexer import build_index

    entries = list_registered()
    if not entries:
        print("No indexed folders found. Run: llmdex-index <directory>")
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
    from llm_index.registry import unregister

    directory = str(Path(args.directory).resolve())
    if unregister(directory):
        print(f"Removed: {directory}")
    else:
        print(f"Not found in registry: {directory}")


def main():
    parser = argparse.ArgumentParser(description="llmdex - index project files for semantic search")
    sub = parser.add_subparsers(dest="command")

    # Default: index (also works without subcommand for backwards compat)
    parser.add_argument("directory", nargs="?", default=None, help="Project directory to index")
    parser.add_argument("-e", "--extensions", nargs="+", default=[".md", ".ts", ".json"],
                        help="File extensions to index (default: .md .ts .json)")

    # llmdex-index reindex
    p_reindex = sub.add_parser("reindex", help="Re-index all previously indexed folders")

    # llmdex-index list
    p_list = sub.add_parser("list", help="List all indexed folders")

    # llmdex-index remove <directory>
    p_remove = sub.add_parser("remove", help="Remove a folder from the index")
    p_remove.add_argument("directory", help="Directory to remove")

    args = parser.parse_args()

    if args.command == "reindex":
        cmd_reindex(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "remove":
        cmd_remove(args)
    else:
        # Default: index a directory
        if args.directory is None:
            args.directory = "."
        cmd_index(args)


if __name__ == "__main__":
    main()
