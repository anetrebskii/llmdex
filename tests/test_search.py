from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from llm_index import server


def _chunks(texts, files=None):
    return {
        "files": files or [f"src/f{i}.py" for i in range(len(texts))],
        "texts": texts,
        "start_lines": [1] * len(texts),
        "end_lines": [10] * len(texts),
    }


def _unit(rows):
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


class FakeEmbedder:
    def __init__(self, vector):
        self.vector = _unit([vector])[0]

    def embed_query(self, question):
        return self.vector


class Search:
    """Point server.cache at in-memory indexes: {workspace name: (texts, files, vectors)}."""

    def __init__(self, indexes, query_vector):
        self.data = {
            name: (_chunks(texts, files), _unit(vectors), server.BM25(texts))
            for name, (texts, files, vectors) in indexes.items()
        }
        self.embedder = FakeEmbedder(query_vector)

    def __enter__(self):
        self.patches = [
            patch.object(server.cache, "get_index", lambda w: self.data[str(w)][:2]),
            patch.object(server.cache, "get_bm25", lambda w: self.data[str(w)][2]),
            patch.object(server.cache, "get_embed_model", lambda: self.embedder),
        ]
        for p in self.patches:
            p.start()
        return server.QueryHandler.__new__(server.QueryHandler)

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()


def _search_before(workspace, question, top_k, folder=None):
    """The single-index search as it was before indexes were searched as one collection."""
    chunks, vectors = server.cache.get_index(workspace)
    fetch_k = min(top_k * 3 * (2 if folder else 1), len(vectors))
    similarity = vectors @ server.cache.get_embed_model().embed_query(question)
    top = np.argpartition(-similarity, fetch_k - 1)[:fetch_k]
    vector_rows = top[np.argsort(-similarity[top], kind="stable")]
    scores = server.cache.get_bm25(workspace).get_scores(server._tokenize_code(question))
    top = np.argpartition(-scores, fetch_k - 1)[:fetch_k]
    bm25_rows = [i for i in top[np.argsort(-scores[top], kind="stable")] if scores[i] > 0]
    fused = {}
    for rows in (vector_rows, bm25_rows):
        for rank, row in enumerate(rows):
            fused[int(row)] = fused.get(int(row), 0) + 1.0 / (60 + rank + 1)
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return [chunks["files"][r] for r, _ in ranked if not folder or chunks["files"][r].startswith(folder)][:top_k]


WORDS = "order payment card account session client queue event status balance limit retry".split()


def _random_index(rng, n, prefix):
    texts = [" ".join(rng.choice(WORDS, size=rng.integers(5, 40))) for _ in range(n)]
    files = [f"{prefix}/f{i}.py" for i in range(n)]
    return texts, files, rng.normal(size=(n, 8))


class BM25StatsTests(unittest.TestCase):
    def test_summed_statistics_score_as_one_corpus(self):
        rng = np.random.default_rng(1)
        a, _, _ = _random_index(rng, 40, "a")
        b, _, _ = _random_index(rng, 70, "b")
        # A word in fewer than half the chunks, so rank_bm25's floor for common words does not apply
        a[3] += " registerhook"
        b[5] += " registerhook registerhook"
        query = ["registerhook"]
        ia, ib = server.BM25(a), server.BM25(b)
        stats = server.CorpusStats.of([ia, ib], query)
        joined = server.BM25(a + b).get_scores(query)
        merged = np.concatenate([ia.get_scores(query, stats), ib.get_scores(query, stats)])
        np.testing.assert_allclose(merged, joined, rtol=1e-5)
        self.assertEqual(set(np.flatnonzero(joined)), {3, 45})
        # Scored alone, each index weighs the word by its own rarity, and the two lists disagree with the joined one
        self.assertFalse(np.allclose(np.concatenate([ia.get_scores(query), ib.get_scores(query)]), joined, rtol=1e-3))

    def test_one_index_keeps_its_own_numbers(self):
        rng = np.random.default_rng(2)
        texts, _, _ = _random_index(rng, 50, "a")
        index = server.BM25(texts)
        query = ["payment", "retry", "nothing"]
        np.testing.assert_allclose(
            index.get_scores(query, server.CorpusStats.of([index], query)),
            index.get_scores(query),
            rtol=1e-6,
        )


class SearchTests(unittest.TestCase):
    def test_one_index_ranks_as_before(self):
        rng = np.random.default_rng(3)
        for seed in range(20):
            texts, files, vectors = _random_index(np.random.default_rng(seed), 60, "repo")
            with Search({"/w": (texts, files, vectors)}, rng.normal(size=8)) as handler:
                for question in ("payment retry", "card status limit", "zzz"):
                    got = [r["source"] for r in handler._search([Path("/w")], question, 8)]
                    self.assertEqual(got, _search_before(Path("/w"), question, 8), (seed, question))

    def test_exact_name_from_another_index_survives_the_merge(self):
        rng = np.random.default_rng(4)
        query_vector = np.ones(8)
        # Repository a: chunks that mean much the same as the question but never use its words
        a_texts = ["notify the provider once the charge settles"] * 30
        a_vectors = query_vector + rng.normal(scale=0.05, size=(30, 8))
        # Repository b: the method itself, worded differently, further from the question in embedding space
        b_texts = ["filler text"] * 29 + ["InvoiceEvents RegisterHook validates the request"]
        b_vectors = rng.normal(size=(30, 8))
        b_files = [f"b/f{i}.cs" for i in range(29)] + ["b/InvoiceEvents.cs"]
        indexes = {"/a": (a_texts, [f"a/f{i}.py" for i in range(30)], a_vectors), "/b": (b_texts, b_files, b_vectors)}
        with Search(indexes, query_vector) as handler:
            got = [r["source"] for r in handler._search([Path("/a"), Path("/b")], "InvoiceEvents RegisterHook", 5)]
        self.assertIn("b/InvoiceEvents.cs", got)

    def test_results_carry_no_score(self):
        texts, files, vectors = _random_index(np.random.default_rng(5), 10, "r")
        with Search({"/r": (texts, files, vectors)}, np.ones(8)) as handler:
            for item in handler._search([Path("/r")], "payment", 3):
                self.assertEqual(set(item) - {"start_line", "end_line"}, {"source", "text"})


if __name__ == "__main__":
    unittest.main()
