#!/usr/bin/env python3
"""CLI for indexing project files."""

import argparse
import sys
from pathlib import Path

# Force unbuffered output so progress is visible in real time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from llm_index.indexer import build_index


def main():
    parser = argparse.ArgumentParser(description="Index project files for semantic search")
    parser.add_argument("directory", nargs="?", default=".", help="Project directory to index (default: current dir)")
    parser.add_argument("-e", "--extensions", nargs="+", default=[".md", ".ts", ".json"],
                        help="File extensions to index (default: .md .ts .json)")
    args = parser.parse_args()

    workspace = Path(args.directory).resolve()
    if not workspace.is_dir():
        print(f"Error: {workspace} is not a directory")
        sys.exit(1)

    # Normalize extensions
    extensions = tuple(ext if ext.startswith(".") else f".{ext}" for ext in args.extensions)

    print(f"Indexing: {workspace}")
    print(f"Extensions: {', '.join(extensions)}")
    result = build_index(workspace, extensions)

    if result.get("error"):
        print(result["error"])
        sys.exit(1)


if __name__ == "__main__":
    main()
