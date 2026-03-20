# llmdex

Local semantic search for your projects. No API keys, no cloud — everything runs on your machine.

llmdex indexes your project files (Markdown, TypeScript, JSON) into a local vector database and lets you search by meaning, not just keywords.

## Install

```bash
# uv (recommended)
uv tool install git+https://github.com/anetrebskii/llmdex

# or pipx
pipx install git+https://github.com/anetrebskii/llmdex
```

Requires Python 3.10+.

## Quick start

```bash
# 1. Index your project
cd ~/my-project
llm-index

# 2. Search
llm-query "how does authentication work"
```

That's it. The first query takes ~25s (model loading), all subsequent queries are instant.

## Commands

### `llm-index` — Build the index

```bash
# Index current directory
llm-index

# Index a specific project
llm-index /path/to/project
```

Creates a `.llm-index/storage/` directory inside the project with the vector index.

**What gets indexed:**
- `.md` files — parsed by headings and sections
- `.ts` files — parsed by code structure (functions, classes)
- `.json` files — parsed by text chunks

**What gets skipped:**
`node_modules`, `.git`, `dist`, `build`, `.next`, `.venv`, `__pycache__`, and other common build/cache directories.

### `llm-query` — Search the index

```bash
# Basic search
llm-query "database connection setup"

# Search a specific project
llm-query -d /path/to/project "error handling"

# Get more results (default: 5)
llm-query -k 10 "API endpoints"
```

**Output:**

```
Query: database connection setup
Top 5 results:

1. [0.742] src/db/connection.ts
   export async function connectDatabase(config: DbConfig) { const pool = new Pool({...

2. [0.698] docs/setup.md
   ## Database Configuration  Set the following environment variables...
```

Each result shows a relevance score (0-1), the source file, and a text preview.

### `llm-server` — Manage the background server

The query server starts automatically on first `llm-query` call. It keeps the embedding model in memory so subsequent queries are fast.

```bash
# Start manually
llm-server

# Custom port (default: 7392)
llm-server -p 8080

# Custom inactivity timeout in seconds (default: 600 = 10 min)
llm-server -t 1800

# Stop the server
llm-server --stop
```

The server shuts down automatically after 10 minutes of inactivity.

## Server HTTP API

The server exposes a local HTTP API on `127.0.0.1:7392`. You can query it directly with `curl`:

```bash
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

1. **Indexing** — Files are parsed into chunks using smart parsers (Markdown by headings, TypeScript by code structure, JSON by text boundaries). Each chunk is embedded into a vector using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), a small local model (~80MB). Vectors are stored in a local index.

2. **Querying** — Your search query is embedded with the same model, then compared against all stored vectors to find the most semantically similar chunks. This means "how does login work" will find code about authentication even if the word "login" doesn't appear.

3. **Server** — A lightweight HTTP server keeps the embedding model and indexes in memory between queries. It auto-starts on first query and auto-stops after 10 minutes of inactivity.

## Typical workflow

```bash
# First time: index the project
cd ~/my-project
llm-index

# Search anytime
llm-query "payment processing"
llm-query "how are emails sent"
llm-query -k 10 "error handling in API routes"

# After major code changes: re-index
llm-index
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
uv tool uninstall llm-index

# pipx
pipx uninstall llm-index

# Remove server data
rm -rf ~/.llm-index

# Remove project indexes (in each project)
rm -rf .llm-index/storage
```
