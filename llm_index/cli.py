#!/usr/bin/env python3
"""Unified CLI for llmdex."""

import argparse
import sys

# Force UTF-8 + unbuffered output so progress is visible in real time and
# non-Latin content prints on Windows (where stdout defaults to cp1252).
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)


def cmd_index(args):
    from pathlib import Path
    from llm_index.indexer import build_index
    from llm_index.registry import set_tags, set_description

    workspace = Path(args.directory).resolve()
    if not workspace.is_dir():
        print(f"Error: {workspace} is not a directory")
        sys.exit(1)

    print(f"Indexing: {workspace}")
    if args.split:
        print("Mode: split (root + each immediate subfolder as separate index)")
    result = build_index(
        workspace,
        verbose=args.verbose,
        split=args.split,
        parent_tags=args.tag,
    )

    if result.get("error"):
        print(result["error"])
        sys.exit(1)

    if args.tag:
        set_tags(str(workspace), args.tag)
        print(f"Tags: {', '.join(sorted(set(args.tag)))}")

    if args.description:
        set_description(str(workspace), args.description)
        print(f"Description: {args.description}")

    if result.get("children"):
        print(f"Children indexes ({len(result['children'])}):")
        for c in result["children"]:
            print(f"  - {c}")

    # Ensure llmdex.md is present in the project's .claude/ directory
    _ensure_llmdex_md(workspace / ".claude")


def cmd_add(args):
    from pathlib import Path
    from llm_index.registry import register, set_tags

    workspace = Path(args.directory).resolve()
    if not workspace.is_dir():
        print(f"Error: {workspace} is not a directory")
        sys.exit(1)

    register(str(workspace), description=args.description)

    if args.tag:
        set_tags(str(workspace), args.tag)

    tags_str = f" (tags: {', '.join(args.tag)})" if args.tag else ""
    print(f"Registered: {workspace}{tags_str}")
    if args.description:
        print(f"  description: {args.description}")
    print("Run `llmdex reindex` to build the index.")


def cmd_reindex(args):
    from pathlib import Path
    from llm_index.registry import list_registered
    from llm_index.indexer import build_index

    entries = list_registered()
    if not entries:
        print("No indexed folders found. Run: llmdex index <directory>")
        return

    # Skip children of split parents -- parent reindex handles them.
    child_of_parent: set[str] = set()
    for meta in entries.values():
        for c in meta.get("children", []):
            child_of_parent.add(c)

    to_reindex = [(d, m) for d, m in entries.items() if d not in child_of_parent]

    print(f"Re-indexing {len(to_reindex)} folder(s)...\n")
    for directory, meta in to_reindex:
        workspace = Path(directory)
        if not workspace.is_dir():
            print(f"Skipping (not found): {directory}")
            continue

        split = bool(meta.get("split"))
        parent_tags = meta.get("tags", []) if split else None
        print(f"--- {directory}{' [split]' if split else ''} ---")
        result = build_index(
            workspace,
            verbose=args.verbose,
            force=args.force,
            split=split,
            parent_tags=parent_tags,
        )
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
        store = storage_dir(Path(directory))
        exists = store.exists()
        status = "ok" if exists else "missing index"
        indexed_at = meta.get("indexed_at", "unknown")
        tags = meta.get("tags", [])
        description = meta.get("description", "")
        print(f"  {directory}")
        if tags:
            print(f"    tags: {', '.join(tags)}")
        if description:
            print(f"    description: {description}")
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
    query_server(port, args.question, args.top_k, directory, args.folder, tags, compact=args.compact)


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


def cmd_describe(args):
    from pathlib import Path
    from llm_index.registry import get_entry, set_description

    directory = str(Path(args.directory).resolve())
    entry = get_entry(directory)
    if entry is None:
        print(f"Error: '{directory}' is not indexed.")
        sys.exit(1)

    if not args.text:
        current = entry.get("description", "")
        if current:
            print(f"Description for {directory}: {current}")
        else:
            print(f"No description for {directory}")
        return

    text = " ".join(args.text)
    if set_description(directory, text):
        print(f"Description set for {directory}: {text}")
    else:
        print(f"Error: '{directory}' is not indexed.")
        sys.exit(1)


def cmd_catalog(args):
    from llm_index.registry import list_registered

    entries = list_registered()
    if not entries:
        print("No indexed folders. Use: llmdex index <directory>")
        return

    # Skip children of split parents -- they share the parent context.
    child_of_parent: set[str] = set()
    for meta in entries.values():
        for c in meta.get("children", []):
            child_of_parent.add(c)

    for directory, meta in entries.items():
        if directory in child_of_parent:
            continue
        tags = meta.get("tags", [])
        description = meta.get("description", "")
        children = meta.get("children", [])
        print(directory)
        if tags:
            print(f"  tags: {', '.join(tags)}")
        if description:
            print(f"  description: {description}")
        if children:
            print(f"  subfolders ({len(children)}):")
            for c in children:
                cmeta = entries.get(c, {})
                ctags = [t for t in cmeta.get("tags", []) if t.startswith("folder:")]
                cdesc = cmeta.get("description", "")
                line = f"    {c}"
                if ctags:
                    line += f"  [{', '.join(ctags)}]"
                print(line)
                if cdesc:
                    print(f"      description: {cdesc}")
        print()


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



LLMDEX_MD_CONTENT = """\
---
name: llmdex-search
description: Semantic code/docs search across all indexed projects. Use this skill whenever the user asks to find, explain, or locate something ("where is X", "how does Y work", "find notes about Z", "what did we decide about W"). ALWAYS starts with tag discovery before searching -- do not assume the answer lives only in the current directory.
---

# LLMDEX -- Semantic Search Skill

`llmdex` is a local semantic search tool that indexes many projects and doc folders on this machine. Results are chunks of actual source code / markdown with file paths and line numbers.

## Core rule: check the catalog first, don't index

When the user asks you to find something, the answer is probably already indexed -- just not in the current working directory. **Never respond "not indexed, let me index it" without first checking what IS indexed.**

The default loop is:

1. **Run `llmdex catalog`** to see every indexed project with its tags AND a human-written description of what lives there. The description tells you what the index is *for* -- use it to match the user's intent to the right index.
2. **READ THE OUTPUT CAREFULLY.** Match on description first (semantic), then on tags (structural). `folder:meetings` under `/twinsai/meetings` with description "Weekly syncs and retros" means that's where meeting notes live.
3. **Pick the MOST SPECIFIC tag combination** that matches the user's intent -- not just one tag. Combine `project:*` + `folder:*` + `type:*` whenever the user's question points at a specific area.
4. **Query with `-t <tag> -t <tag>`** (AND logic narrows to the intersection).
5. **Only if that returns nothing useful**, broaden: drop a tag, try a sibling tag, and only then `-a`.
6. **Last resort**: say "nothing indexed matches". Do NOT auto-index.

`llmdex tags` still exists and just lists tags with no descriptions -- use it only if you already know what you want and just need to confirm a tag spelling. For discovery, always prefer `llmdex catalog`.

### Worked example

User: "Что сказал Дмитрий на последнем звонке? В twinsai"

Wrong: `llmdex query "..."` in cwd → fails → jump to `-a` with a broad query.

Right:
1. `llmdex catalog` -- find the twinsai entry; its description mentions meeting notes, and a subfolder is tagged `folder:meetings`.
2. Combine tags: `llmdex query -t project:twinsai -t folder:meetings "Дмитрий последний звонок"`.
3. That's the narrowest, most relevant slice -- use it first.

The general pattern: user mentions a project/domain + a content type (meetings, notes, slack, tasks, etc.) -> the catalog description + tags together point at the right slice. Read the catalog before querying.

Do NOT run `llmdex index` on a new directory just because `llmdex query` in the current folder failed. Indexing is a user-initiated setup step, not a recovery step.

## When to use this skill

Trigger whenever the user's request sounds like retrieval or recall:

- "where is X handled?" / "how does Y work?" / "find code related to Z"
- "what did we write about ...", "find my notes on ...", "was there a doc about ..."
- "which project uses X?", "do we have an example of Y anywhere?"
- Any question about past decisions, meetings, specs, or prior work.

## External sources may be indexed locally

GitHub issues/PRs, Slack threads, Confluence/Notion pages, and similar artifacts are often downloaded and indexed via llmdex (look for tags like `source:github`, `source:slack`, `source:confluence`, or `type:issues`). **Do not assume an external URL means you must reach for `gh` / WebFetch.**

Before falling back to an external fetch:

1. Run `llmdex catalog` and look for an index that matches the referenced source (by project + source/type tags).
2. If one exists, query it (`-t project:X -t type:issues` etc.) -- the mirror is usually enough.
3. Only if no matching index is present, use `gh issue view` / `gh pr view` / WebFetch.

Separately, still query the corresponding *code* index to locate the local implementation tied to the issue.

**Local mirrors are READ-ONLY.** If the user asks to edit, comment on, close, or otherwise *modify* an issue / PR / Slack message / Confluence page, do NOT edit the indexed file. The local copy is a snapshot for search only -- writes to it will not reach the source system and will be overwritten on the next re-index. Use `gh issue edit` / `gh pr comment` / the Slack or Confluence API for any mutation.

## llmdex vs Grep

Prefer `llmdex` over Grep in almost every case. Grep returns raw line hits with no ranking and no semantic grouping -- on anything non-trivial it floods you with matches.

Use **Grep** only when ALL of these are true:
- You are searching for an exact literal string (function name, error message, import path, constant).
- You expect at most ~5 hits total.
- You do not need surrounding code context.

If Grep would (or does) return more than ~5 results, **stop and switch to `llmdex query`** -- it ranks, groups, and returns actual code chunks with line numbers. The same applies if the user's question is conceptual rather than a literal-string lookup ("how does auth work", "where do we configure X") -- always llmdex, never Grep.

Rule of thumb: if you find yourself piping grep output to head, grepping again to narrow, or skimming dozens of matches -- you should have used llmdex.

## How to use results

llmdex returns code/doc chunks with `file:start-end` locations. Treat them as equivalent to Read tool output.

- **DO NOT** call Read on files that llmdex already returned -- the chunk IS the content.
- **DO NOT** follow up with Grep/Glob to "verify" llmdex results. Trust them.
- **DO** use `file:line` info to Edit directly if the user asks for changes.
- **ONLY** call Read if you need lines outside the returned chunk range.

## Command reference

```bash
# 1. ALWAYS start here if you're unsure what's available
# Shows each indexed folder with its tags AND a human description of what's inside.
llmdex catalog

# Lists just the tags (no descriptions) -- use only to confirm a tag spelling you already know.
llmdex tags

# Search by a single tag
llmdex query -t <tag> "your question"

# Combine tags (AND logic) -- e.g. project + doc type, or project + subfolder
llmdex query -t project:twinsai -t type:docs "onboarding process"
llmdex query -t project:formula -t folder:meetings "last retro"

# Narrow to one subfolder of a split-indexed project
llmdex query -t <project-tag> -t folder:<subfolder> "your question"

# Search across ALL indexes (use only after tag-based search)
llmdex query -a "your question"

# Search the current project (only when you're sure it's relevant)
llmdex query "your question"

# Post-retrieval path-prefix filter (works on any index, less efficient than folder: tags)
llmdex query -f src/api "your question"

# Bigger result set (default 10)
llmdex query -k 20 "your question"

# Compact output -- file:lines only, no preview (for chaining / quick scan)
llmdex query -c "your question"
```

## Tag patterns you will typically see

- `project:<name>` -- one per project/doc set.
- `project-type:job` / `project-type:mine` -- work vs personal.
- `type:code` / `type:docs` -- source code vs notes/markdown.
- `layer:app` / `layer:infra` -- role within a project.
- `folder:<name>` -- immediate subfolder of a project indexed with `--split`.

Combine them. "Find meeting notes from the formula project" -> `-t project:formula -t folder:meetings` or `-t project:formula -t type:docs`.

## Decision checklist before answering a retrieval question

- [ ] Did I run `llmdex catalog` (or do I already know the relevant index from this session)?
- [ ] Did I pick the most specific tag combo, not just the current directory?
- [ ] Did I try `-a` only after tag-based search returned nothing useful?
- [ ] Am I resisting the urge to run `llmdex index` just because a query was empty?

If all four are checked and there are still no results, report "nothing indexed matches" -- don't auto-index.

## Other rules

- Results use hybrid search (BM25 + vector): exact keyword matches and semantic matches both surface.
- `llmdex query` with no `-t`/`-a`/`-d` errors if the current directory is not indexed. That is a signal to pick a tag, not to index.
- User says "everywhere" / "across all projects" -- use `-a`.
- Prefer `-t <tag>` over `-a` -- more relevant, faster, less noisy.
"""

INTEGRATE_REFERENCE = "@llmdex.md\n"


def _ensure_llmdex_md(claude_dir):
    """Write llmdex.md and add @llmdex.md to CLAUDE.md if not present."""
    from pathlib import Path

    claude_dir = Path(claude_dir)
    claude_dir.mkdir(parents=True, exist_ok=True)

    llmdex_md = claude_dir / "llmdex.md"
    llmdex_md.write_text(LLMDEX_MD_CONTENT)

    claude_md = claude_dir / "CLAUDE.md"
    if claude_md.exists():
        existing = claude_md.read_text()
        if "@llmdex.md" not in existing:
            separator = "\n" if existing.endswith("\n") else "\n\n"
            claude_md.write_text(existing + separator + INTEGRATE_REFERENCE)
    else:
        claude_md.write_text(INTEGRATE_REFERENCE)


def cmd_init(args):
    from pathlib import Path

    if args.scope == "global":
        claude_dir = Path.home() / ".claude"
        label = "global (~/.claude/)"
    else:
        claude_dir = Path(".claude")
        label = f"project ({claude_dir}/)"

    _ensure_llmdex_md(claude_dir)
    print(f"Wrote {claude_dir / 'llmdex.md'}")
    print(f"Integration: {label} -- done")


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
    p_index.add_argument(
        "-D",
        "--description",
        help="Human-readable description of what's in this index (shown in `llmdex catalog`)",
    )
    p_index.add_argument(
        "--split",
        action="store_true",
        help="Also index each immediate subfolder as a separate child index (tagged folder:<name>)",
    )

    # llmdex add
    p_add = sub.add_parser("add", help="Register a project without indexing (use reindex later)")
    p_add.add_argument(
        "directory", nargs="?", default=".", help="Project directory (default: .)"
    )
    p_add.add_argument(
        "-t",
        "--tag",
        action="append",
        help="Tag to assign (can be repeated, e.g. -t project:foo -t type:code)",
    )
    p_add.add_argument(
        "-D",
        "--description",
        help="Human-readable description of what's in this index (shown in `llmdex catalog`)",
    )

    # llmdex reindex / re
    p_reindex = sub.add_parser("reindex", aliases=["re"], help="Re-index all registered projects")
    p_reindex.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        help="Show individual file paths",
    )
    p_reindex.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force full rebuild, ignoring change detection",
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
        "-k", "--top-k", type=int, default=10, help="Number of results (default: 10)"
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
    p_query.add_argument(
        "-c",
        "--compact",
        action="store_true",
        help="Compact output: file:lines only, no preview (for AI/automation)",
    )

    # llmdex tag
    p_tag = sub.add_parser("tag", help="Set tags on an indexed project")
    p_tag.add_argument(
        "directory", nargs="?", default=".", help="Project directory (default: .)"
    )
    p_tag.add_argument("tags", nargs="*", help="Tags to set (omit to show current)")

    # llmdex tags
    sub.add_parser("tags", help="List all tags and their indexes")

    # llmdex describe
    p_describe = sub.add_parser(
        "describe",
        help="Set or show description for an indexed project",
    )
    p_describe.add_argument(
        "directory", nargs="?", default=".", help="Project directory (default: .)"
    )
    p_describe.add_argument("text", nargs="*", help="Description text (omit to show current)")

    # llmdex catalog
    sub.add_parser(
        "catalog",
        aliases=["cat"],
        help="List all indexed projects with tags and descriptions (use for discovery)",
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
        "describe": cmd_describe,
        "catalog": cmd_catalog,
        "cat": cmd_catalog,
        "server": cmd_server,
        "s": cmd_server,
        "init": cmd_init,
        "i": cmd_init,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
