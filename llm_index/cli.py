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
    from llm_index.registry import set_tags

    workspace = Path(args.directory).resolve()
    if not workspace.is_dir():
        print(f"Error: {workspace} is not a directory")
        sys.exit(1)

    extensions = tuple(
        ext if ext.startswith(".") else f".{ext}" for ext in args.extensions
    )

    print(f"Indexing: {workspace}")
    print(f"Extensions: {', '.join(extensions)}")
    result = build_index(workspace, extensions, verbose=args.verbose)

    if result.get("error"):
        print(result["error"])
        sys.exit(1)

    if args.tag:
        set_tags(str(workspace), args.tag)
        print(f"Tags: {', '.join(sorted(set(args.tag)))}")


def cmd_add(args):
    from pathlib import Path
    from llm_index.registry import register, set_tags

    workspace = Path(args.directory).resolve()
    if not workspace.is_dir():
        print(f"Error: {workspace} is not a directory")
        sys.exit(1)

    extensions = [ext if ext.startswith(".") else f".{ext}" for ext in args.extensions]
    register(str(workspace), extensions)

    if args.tag:
        set_tags(str(workspace), args.tag)

    tags_str = f" (tags: {', '.join(args.tag)})" if args.tag else ""
    print(f"Registered: {workspace}{tags_str}")
    print(f"  extensions: {', '.join(extensions)}")
    print("Run `llmdex reindex` to build the index.")


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
        result = build_index(workspace, extensions, verbose=args.verbose)
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
        tags = meta.get("tags", [])
        print(f"  {directory}")
        print(f"    extensions: {extensions}")
        if tags:
            print(f"    tags: {', '.join(tags)}")
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


def _find_indexed_ancestor(path, get_entry):
    """Walk up from path to find the nearest indexed ancestor. Returns the directory string or None."""
    current = path
    while True:
        if get_entry(str(current)):
            return str(current)
        parent = current.parent
        if parent == current:
            return None
        current = parent


def cmd_query(args):
    from pathlib import Path
    from llm_index.query import ensure_server, query_server

    tags = args.tag if args.tag else None
    directory = None if (args.all or tags) else args.directory

    # Early check: if searching a specific directory, verify it (or a parent) is indexed
    if directory is not None:
        from llm_index.registry import get_entry

        resolved = Path(directory).resolve()
        indexed_dir = _find_indexed_ancestor(resolved, get_entry)
        if indexed_dir is None:
            print(f"Error: '{resolved}' is not indexed.")
            print(f"Run: llmdex index {resolved}")
            sys.exit(1)
        # Use the indexed ancestor directory for the query
        directory = indexed_dir

    # If tags specified, resolve to directories
    if tags:
        from llm_index.registry import find_by_tags

        matches = find_by_tags(tags)
        if not matches:
            print(f"Error: no indexes found with tags: {', '.join(tags)}")
            sys.exit(1)

    port = ensure_server()
    query_server(port, args.question, args.top_k, directory, args.folder, tags)


def cmd_tag(args):
    from pathlib import Path
    from llm_index.registry import get_entry, set_tags

    directory = str(Path(args.directory).resolve())
    entry = get_entry(directory)
    if entry is None:
        print(f"Error: '{directory}' is not indexed.")
        sys.exit(1)

    if not args.tags:
        # Show current tags
        current = entry.get("tags", [])
        if current:
            print(f"Tags for {directory}: {', '.join(current)}")
        else:
            print(f"No tags for {directory}")
        return

    if set_tags(directory, args.tags):
        print(f"Tags set for {directory}: {', '.join(sorted(set(args.tags)))}")
    else:
        print(f"Error: '{directory}' is not indexed.")
        sys.exit(1)


def cmd_tags(args):
    from llm_index.registry import list_all_tags

    tags = list_all_tags()
    if not tags:
        print("No tags found. Use: llmdex tag <directory> <tag1> <tag2> ...")
        return

    for tag, dirs in sorted(tags.items()):
        print(f"  {tag}")
        for d in dirs:
            print(f"    {d}")
        print()


INTEGRATE_HEADER = "## LLMDEX — Semantic Search"

INTEGRATE_GLOBAL = """\
## LLMDEX — Semantic Search

You have access to `llmdex` — a local semantic search tool that indexes project codebases.
The registry may contain many indexes: different projects, and even separate indexes within a project (e.g. docs, code, configs). Tags are the primary way to navigate them.

### When to use

- **Use llmdex** for exploratory searches: "where is authentication handled?", "how does the billing flow work?", "find code related to caching". It understands meaning, not just keywords.
- **Use Grep** for exact lookups: a specific function name, error message, or import path.
- **Use llmdex first**, then Grep/Read on the files it finds for detailed analysis.

### Commands

```bash
# Search current project (run from inside the project directory)
llmdex query "your question"

# Search only indexes with a specific tag (recommended for cross-project searches)
llmdex query -t <tag> "your question"

# Combine multiple tags (AND logic — matches indexes that have ALL tags)
llmdex query -t backend -t api "your question"

# Search across ALL indexed projects (use sparingly — prefer tags for precision)
llmdex query -a "your question"

# Discover available tags and which indexes they cover — run this first when unsure
llmdex tags

# Return more results (default: 5)
llmdex query -k 10 "your question"
```

### Important

- `llmdex query` exits with an error if the current directory is not indexed. Fall back to Grep/Glob if that happens.
- When the user asks to search "everywhere" or "across all projects", use `-a`.
- When the user refers to a domain, project, or layer (e.g. "in the backend", "in docs", "in our mobile apps"), run `llmdex tags` first to discover matching tags, then use `-t <tag>`.
- Prefer `-t <tag>` over `-a` when possible — it gives more relevant results and is faster.
"""

INTEGRATE_PROJECT = """\
## LLMDEX — Semantic Search

This project is indexed by `llmdex` — a local semantic search tool.

### When to use

- **Use llmdex** for exploratory searches: "where is authentication handled?", "how does the billing flow work?". It understands meaning, not just keywords.
- **Use Grep** for exact lookups: a specific function name, error message, or import path.
- **Use llmdex first**, then Grep/Read on the files it finds for detailed analysis.

```bash
llmdex query "your question"
llmdex query -k 10 "your question"       # more results
llmdex query -t <tag> "your question"     # search by tag across projects
llmdex tags                               # discover available tags
```

If `llmdex query` exits with an error, the index may be stale — fall back to Grep/Glob.
"""


def cmd_init(args):
    from pathlib import Path

    if args.scope == "global":
        claude_md = Path.home() / ".claude" / "CLAUDE.md"
        content = INTEGRATE_GLOBAL
        label = "global (~/.claude/CLAUDE.md)"
    else:
        claude_md = Path(".claude") / "CLAUDE.md"
        content = INTEGRATE_PROJECT
        label = f"project ({claude_md})"

    # Ensure parent directory exists
    claude_md.parent.mkdir(parents=True, exist_ok=True)

    if claude_md.exists():
        existing = claude_md.read_text()
        if INTEGRATE_HEADER in existing:
            print(f"Already integrated in {label}")
            print(f"  Header '{INTEGRATE_HEADER}' found in {claude_md}")
            return
        # Append to existing file
        separator = "\n" if existing.endswith("\n") else "\n\n"
        claude_md.write_text(existing + separator + content)
        print(f"Appended LLMDEX section to {claude_md}")
    else:
        claude_md.write_text(content)
        print(f"Created {claude_md} with LLMDEX section")

    print(f"Integration: {label} — done")


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
    p_index.add_argument(
        "-t",
        "--tag",
        action="append",
        help="Tag to assign (can be repeated, e.g. -t project:foo -t type:code)",
    )
    p_index.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        help="Show individual file paths",
    )

    # llmdex add
    p_add = sub.add_parser("add", help="Register a project without indexing (use reindex later)")
    p_add.add_argument(
        "directory", nargs="?", default=".", help="Project directory (default: .)"
    )
    p_add.add_argument(
        "-e",
        "--extensions",
        nargs="+",
        default=[".md", ".ts", ".json"],
        help="File extensions to index (default: .md .ts .json)",
    )
    p_add.add_argument(
        "-t",
        "--tag",
        action="append",
        help="Tag to assign (can be repeated, e.g. -t project:foo -t type:code)",
    )

    # llmdex reindex / re
    p_reindex = sub.add_parser("reindex", aliases=["re"], help="Re-index all registered projects")
    p_reindex.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        help="Show individual file paths",
    )

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
    p_query.add_argument(
        "-f", "--folder", help="Filter results to files under this folder prefix"
    )
    p_query.add_argument(
        "-t",
        "--tag",
        action="append",
        help="Search only indexes with this tag (can be repeated)",
    )

    # llmdex tag
    p_tag = sub.add_parser("tag", help="Set tags on an indexed project")
    p_tag.add_argument(
        "directory", nargs="?", default=".", help="Project directory (default: .)"
    )
    p_tag.add_argument("tags", nargs="*", help="Tags to set (omit to show current)")

    # llmdex tags
    sub.add_parser("tags", help="List all tags and their indexes")

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

    # llmdex init / i
    p_init = sub.add_parser(
        "init",
        aliases=["i"],
        help="Add LLMDEX instructions to Claude Code CLAUDE.md",
    )
    p_init.add_argument(
        "scope",
        nargs="?",
        default="project",
        choices=["global", "project"],
        help="Integration scope: 'global' (~/.claude/CLAUDE.md) or 'project' (.claude/CLAUDE.md, default)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "index": cmd_index,
        "idx": cmd_index,
        "add": cmd_add,
        "reindex": cmd_reindex,
        "re": cmd_reindex,
        "list": cmd_list,
        "ls": cmd_list,
        "remove": cmd_remove,
        "rm": cmd_remove,
        "query": cmd_query,
        "q": cmd_query,
        "tag": cmd_tag,
        "tags": cmd_tags,
        "server": cmd_server,
        "s": cmd_server,
        "init": cmd_init,
        "i": cmd_init,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
