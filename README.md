# llmdex

[![CI](https://github.com/anetrebskii/llmdex/actions/workflows/ci.yml/badge.svg)](https://github.com/anetrebskii/llmdex/actions/workflows/ci.yml)

Local semantic search for your projects. No API keys, no cloud — everything runs on your machine.

> **"How does auth work?"**
>
> | | Without llmdex | With llmdex |
> |---|---|---|
> | Time | ~2 min | **~20 sec** |
>
> *Real benchmark with Claude Code on a production codebase.*

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
llmdex-index

# 2. Search
llmdex-query "how does authentication work"
```

That's it. The first query takes ~25s (model loading), all subsequent queries are instant.

## Commands

### `llmdex-index` — Build the index

```bash
# Index current directory
llmdex-index

# Index a specific project
llmdex-index /path/to/project

# Index specific file types
llmdex-index /path/to/project -e .md .ts .py .json
```

**Manage indexed projects:**

```bash
# List all indexed projects
llmdex-index list

# Re-index all registered projects
llmdex-index reindex

# Remove a project from the registry
llmdex-index remove /path/to/project
```

Indexes are stored centrally in `~/.llmdex/indexes/` — project directories stay clean.

**What gets indexed by default:**
- `.md` files — parsed by headings and sections
- `.ts` files — parsed by code structure (functions, classes)
- `.json` files — parsed by text chunks

**What gets skipped:**
`node_modules`, `.git`, `dist`, `build`, `.next`, `.venv`, `__pycache__`, and other common build/cache directories.

### `llmdex-query` — Search the index

```bash
# Basic search
llmdex-query "database connection setup"

# Search a specific project
llmdex-query -d /path/to/project "error handling"

# Get more results (default: 5)
llmdex-query -k 10 "API endpoints"
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

### `llmdex-server` — Manage the background server

The query server starts automatically on first `llmdex-query` call. It keeps the embedding model in memory so subsequent queries are fast.

```bash
# Start manually
llmdex-server

# Custom port (default: 7392)
llmdex-server -p 8080

# Custom inactivity timeout in seconds (default: 1800 = 30 min)
llmdex-server -t 3600

# Stop the server
llmdex-server --stop

# Restart the server
llmdex-server --restart
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

3. **Server** — A lightweight HTTP server keeps the embedding model and indexes in memory between queries. It auto-starts on first query and auto-stops after 30 minutes of inactivity. You can also stop it manually with `llmdex-server --stop`.

## Typical workflow

```bash
# First time: index the project
cd ~/my-project
llmdex-index

# Search anytime
llmdex-query "payment processing"
llmdex-query "how are emails sent"
llmdex-query -k 10 "error handling in API routes"

# After major code changes: re-index
llmdex-index
```

## Claude Code integration

Add llmdex as a semantic search tool for Claude Code by adding this to your project's `CLAUDE.md`:

```markdown
## Semantic Search

This project is indexed with llmdex for semantic search.
When you need to find code by concept (not just filename or keyword), use:

\`\`\`bash
llmdex-query -d /path/to/project "your question" -k 10
\`\`\`

Use this before grep/glob when:
- Searching by concept ("how does auth work", "payment processing logic")
- You don't know the exact file or function name
- Exploring unfamiliar parts of the codebase

To re-index after major changes:
\`\`\`bash
llmdex-index /path/to/project
\`\`\`
```

You can also use the HTTP API via `curl` if the server is running:

```markdown
## Semantic Search (HTTP API)

llmdex server runs on port 7392. Use it for semantic code search:

\`\`\`bash
# Search
curl -s http://127.0.0.1:7392/query \
  -H "Content-Type: application/json" \
  -d '{"directory": "/path/to/project", "question": "your question", "top_k": 10}'

# Index
curl -s http://127.0.0.1:7392/index \
  -H "Content-Type: application/json" \
  -d '{"directory": "/path/to/project", "extensions": [".md", ".ts", ".json"]}'
\`\`\`
```

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
