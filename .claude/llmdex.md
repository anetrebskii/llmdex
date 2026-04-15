# LLMDEX -- Semantic Code Search

You have access to `llmdex` -- a local semantic search tool that indexes project codebases.
Results include full source code with line numbers -- treat them as equivalent to Read tool output. Do NOT re-read files that llmdex already returned.

## When to use

**Default to llmdex for all code search tasks.** Only fall back to Grep for exact string matches (function name, error message, import path).

- "where is X handled?" / "how does X work?" / "find code related to X" -- always llmdex.
- "find all usages of `functionName`" / "which files import X" -- Grep is fine.

## How to use results

llmdex returns chunks of actual source code with file paths and line numbers. This is your primary context -- act on it directly:

- **DO NOT** call Read on files that llmdex already returned. The chunk IS the content.
- **DO NOT** follow up with Grep/Glob to "verify" llmdex results. Trust them.
- **DO** use the file:line info to Edit directly if you need to modify the code.
- **ONLY** call Read if you need lines outside the returned chunk range.

## Commands

```bash
# Search current project
llmdex query "your question"

# Search by tag (recommended for cross-project)
llmdex query -t <tag> "your question"

# Combine tags (AND logic)
llmdex query -t backend -t api "your question"

# Search ALL indexed projects (use sparingly)
llmdex query -a "your question"

# Discover indexes -- run this first when unsure.
# Lists each indexed folder with its tags AND a human description of what's inside.
llmdex catalog

# Lists just the tags (no descriptions) -- use only to confirm a tag spelling.
llmdex tags

# More results (default: 10)
llmdex query -k 20 "your question"

# Compact mode -- file:lines only, no code preview
llmdex query -c "your question"
```

## Rules

- `llmdex query` errors if the current directory is not indexed. Fall back to Grep/Glob.
- User says "everywhere" / "across all projects" -- use `-a`.
- User mentions a domain/project/layer ("in the backend", "in docs") -- run `llmdex catalog` first, then `-t <tag>`.
- Prefer `-t <tag>` over `-a` -- more relevant, faster.
- Results use hybrid search (BM25 + vector), so both exact keyword matches and semantic matches are returned.
