# llmdex

[![CI](https://github.com/anetrebskii/llmdex/actions/workflows/ci.yml/badge.svg)](https://github.com/anetrebskii/llmdex/actions/workflows/ci.yml)

Local semantic search for your projects. No API keys, no cloud — everything runs on your machine.

> **"How does auth work?"**
>
> | | Without llmdex | With llmdex |
> |---|---|---|
> | Time | ~2 min | **~20 sec** |
> | Cost (implementation task) | ~$1.20 | **~$0.50** |
> | Cost (explanation task) | ~$0.25 | **~$0.10** |
>
> *Real benchmarks with Claude Code on a production codebase.*

llmdex indexes your project files (Markdown, TypeScript, JSON) into a local vector database and lets you search by meaning, not just keywords.

Inspired by the original idea and Python scripts of [@lnetrebskii](https://github.com/lnetrebskii).

## Install

```bash
# uv (recommended)
uv tool install git+https://github.com/anetrebskii/llmdex

# or pipx
pipx install git+https://github.com/anetrebskii/llmdex

# pin a specific version
uv tool install git+https://github.com/anetrebskii/llmdex@v0.1.0
```

Requires Python 3.10+.

## Quick start

```bash
# 1. Index your project
cd ~/my-project
llmdex index

# 2. Search
llmdex query "how does authentication work"
```

That's it. The first query takes ~25s (model loading), all subsequent queries are instant.

### Using with Claude Code

```bash
# Add llmdex instructions to Claude Code (global — works in every project)
llmdex init global

# Or per-project only
llmdex init
```

This teaches Claude Code to use `llmdex` for semantic search automatically. See [Claude Code integration](#claude-code-integration) for details.

## Commands

### `llmdex index` — Build the index

```bash
# Index current directory
llmdex index

# Index a specific project
llmdex index /path/to/project

# Index specific file types
llmdex index /path/to/project -e .md .ts .py .json
```

### `llmdex reindex` — Re-index all registered projects

Rebuilds indexes for all previously registered projects. Uses the file extensions stored during the original `llmdex index` call.

```bash
llmdex reindex
```

Skips directories that no longer exist on disk.

### `llmdex list` — List all indexed projects

```bash
llmdex list
```

### `llmdex remove` — Remove a project from the registry

```bash
llmdex remove /path/to/project
```

Indexes are stored centrally in `~/.llmdex/indexes/` — project directories stay clean.

**What gets indexed by default:**
- `.md` files — parsed by headings and sections
- `.ts` files — parsed by code structure (functions, classes)
- `.json` files — parsed by text chunks

**What gets skipped:**
`node_modules`, `.git`, `dist`, `build`, `.next`, `.venv`, `__pycache__`, and other common build/cache directories.

### `llmdex query` — Search the index

```bash
# Basic search
llmdex query "database connection setup"

# Search a specific project
llmdex query -d /path/to/project "error handling"

# Search across all indexed projects
llmdex query -a "API endpoints"

# Get more results (default: 5)
llmdex query -k 10 "API endpoints"
```

**Output:**

```
Query: database connection setup
Top 5 results:

1. [0.742] /Users/you/my-project/src/db/connection.ts
   export async function connectDatabase(config: DbConfig) { const pool = new Pool({...

2. [0.698] /Users/you/my-project/docs/setup.md
   ## Database Configuration  Set the following environment variables...
```

Each result shows a relevance score (0-1), the full file path, and a text preview.

You can also filter queries by tag (see below):

```bash
# Search only indexes tagged "backend"
llmdex query -t backend "database connection"

# Combine tags (AND logic — matches indexes with ALL specified tags)
llmdex query -t backend -t api "error handling"
```

### `llmdex tag` — Set tags on an indexed project

Tags let you organize indexes and filter queries across multiple projects.

```bash
# Set tags on the current directory
llmdex tag . backend api

# Set tags on a specific project
llmdex tag /path/to/project frontend react

# Show current tags for a project
llmdex tag /path/to/project
```

Tags replace any previously set tags (they are not additive).

### `llmdex tags` — List all tags

Shows all tags and which indexed directories have each tag.

```bash
llmdex tags
```

Output:

```text
  backend
    /Users/you/api-service
    /Users/you/worker

  frontend
    /Users/you/web-app
```

### `llmdex server` — Manage the background server

The query server starts automatically on first `llmdex query` call. It keeps the embedding model in memory so subsequent queries are fast.

```bash
# Start manually
llmdex server

# Custom port (default: 7392)
llmdex server -p 8080

# Custom inactivity timeout in seconds (default: 1800 = 30 min)
llmdex server -t 3600

# Stop the server
llmdex server --stop

# Restart the server
llmdex server --restart
```

The server shuts down automatically after 30 minutes of inactivity. After a package update, the server automatically restarts on the next command when it detects a version mismatch.

## Server HTTP API

The server exposes a local HTTP API on `127.0.0.1:7392`. You can use it directly with `curl`:

```bash
# Index a project
curl -s http://127.0.0.1:7392/index \
  -H "Content-Type: application/json" \
  -d '{"directory": "/path/to/project", "extensions": [".md", ".ts", ".py"]}'

# Search
curl -s http://127.0.0.1:7392/query \
  -H "Content-Type: application/json" \
  -d '{"directory": "/path/to/project", "question": "auth flow", "top_k": 5}'

# Reload index after re-indexing
curl -s http://127.0.0.1:7392/invalidate \
  -H "Content-Type: application/json" \
  -d '{"directory": "/path/to/project"}'

# Health check
curl http://127.0.0.1:7392/health
```

## How it works

1. **Indexing** — Files are parsed into chunks using smart parsers (Markdown by headings, TypeScript by code structure, JSON by text boundaries). Each chunk is embedded into a vector using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), a small local model (~80MB). Vectors are stored in `~/.llmdex/indexes/`.

2. **Querying** — Your search query is embedded with the same model, then compared against all stored vectors to find the most semantically similar chunks. This means "how does login work" will find code about authentication even if the word "login" doesn't appear.

3. **Server** — A lightweight HTTP server keeps the embedding model and indexes in memory between queries. It auto-starts on first query and auto-stops after 30 minutes of inactivity. You can also stop it manually with `llmdex server --stop`.

## Typical workflow

```bash
# First time: index the project
cd ~/my-project
llmdex index

# Search anytime
llmdex query "payment processing"
llmdex query "how are emails sent"
llmdex query -k 10 "error handling in API routes"

# After major code changes: re-index
llmdex index
```

## Claude Code integration

### `llmdex init` — Integrate with Claude Code

Adds a `## LLMDEX — Semantic Search` section to `CLAUDE.md` that teaches Claude Code when and how to use `llmdex` for semantic search.

```bash
# Add to the current project's .claude/CLAUDE.md (default)
llmdex init

# Add globally to ~/.claude/CLAUDE.md — works in every project
llmdex init global
```

**Cost optimization:** llmdex reduces Claude Code API costs by 50-60% on typical tasks. Instead of scanning dozens of files to find relevant code, Claude gets precise results from llmdex and spends tokens on the actual work. The savings scale with codebase size -- larger projects see bigger gains.

- **Project scope** (default) — instructions are tailored for single-project use
- **Global scope** — instructions include cross-project features (tags, `-a` flag)
- Running it again is safe — it detects the existing section and won't duplicate it
- Creates `.claude/` directory and `CLAUDE.md` if they don't exist

## Updating

```bash
# uv
uv tool install --force git+https://github.com/anetrebskii/llmdex

# pipx
pipx install --force git+https://github.com/anetrebskii/llmdex
```

## Uninstall

```bash
# uv
uv tool uninstall llmdex

# pipx
pipx uninstall llmdex

# Remove all data (indexes + server PID)
rm -rf ~/.llmdex
```

## License

[MIT](LICENSE)
