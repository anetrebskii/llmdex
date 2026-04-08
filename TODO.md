# TODO

## Tier 1: Quick wins

- [ ] **Chunk enrichment** - prepend file path, language, parent class/function name to each chunk before embedding. 15-30% relevance boost. Implement in `parse_files()`.
- [ ] **Hybrid search (BM25 + vector)** - add BM25 keyword search alongside vector similarity, fuse with Reciprocal Rank Fusion (RRF). Catches exact identifier matches that embeddings miss. ~100 lines.
- [ ] **Better embedding model** - replace `all-MiniLM-L6-v2` (general-purpose, 384-dim) with a code-specific model:
  - `jina-embeddings-v2-base-code` - 768-dim, trained on code, local
  - `nomic-embed-text-v1.5` - Matryoshka embeddings, good for code
  - `CodeSage-small` (Microsoft) - code retrieval specific

## Tier 2: Medium effort

- [ ] **Cross-encoder re-ranking** - retrieve top-20 with vector, re-rank to top-5 with `bge-reranker-v2-m3`. LlamaIndex has `SentenceTransformerRerank` built-in.
- [ ] **AST-aware chunking (tree-sitter)** - split at function/class boundaries instead of line count. Keeps semantic units intact.

## Tier 3: Big projects

- [ ] **Symbol graph index** - secondary index of calls, imports, inheritance via tree-sitter/LSP. Retrieve callers/callees alongside matched chunks.
- [ ] **Multi-representation indexing** - index each unit as raw code + docstring + signature separately, query against all representations and merge.
