from __future__ import annotations

import numpy as np
import pytest

from common.embeddings import HashEmbedder, normalize
from common.fixtures import RUNBOOKS
from common.infra import postgres_available, temp_schema
from day08.ann_experiment import exact_topk, recall_at_k, run_experiment, synthetic_vectors
from day08.chunking import Chunk, chunk_corpus, fixed_chunks, sentence_chunks
from day08.exact_index import ExactIndex
from day08.pgvector_store import PgVectorStore
from day08.similarity import cosine, dot, euclidean_sq


# ---- chunking ---------------------------------------------------------------------------------
def test_fixed_chunks_offsets_overlap_and_coverage() -> None:
    doc = RUNBOOKS[0]
    chunks = fixed_chunks(doc, 100, 20)
    assert all(c.text == doc["text"][c.start : c.end] for c in chunks)
    assert chunks[1].start == chunks[0].end - 20
    assert chunks[-1].end == len(doc["text"])
    assert all(len(c.text) <= 100 for c in chunks)


def test_fixed_chunks_can_cut_mid_word() -> None:
    assert any(not c.text.endswith((".", " ")) for c in fixed_chunks(RUNBOOKS[0], 100, 0)[:-1])


def test_sentence_chunks_respect_boundaries_limit_and_overlap() -> None:
    doc = RUNBOOKS[1]
    chunks = sentence_chunks(doc, 250, 1)
    assert len(chunks) > 1
    assert all(len(c.text) <= 250 for c in chunks)
    assert all(c.text == doc["text"][c.start : c.end] for c in chunks)
    assert all(c.text.rstrip().endswith((".", "!", "?")) for c in chunks)
    assert chunks[1].start < chunks[0].end  # overlap: last sentence repeated
    assert chunks[-1].end == len(doc["text"])


def test_sentence_chunks_zero_overlap_partitions_text() -> None:
    chunks = sentence_chunks(RUNBOOKS[1], 250, 0)
    assert all(a.end <= b.start for a, b in zip(chunks, chunks[1:], strict=False))


def test_oversized_sentence_is_hard_split() -> None:
    doc = {"id": "x", "title": "t", "tags": [], "text": "word " * 100 + "."}
    chunks = sentence_chunks(doc, 120, 1)
    assert len(chunks) >= 4 and all(len(c.text) <= 120 for c in chunks)


def test_chunking_validates_arguments() -> None:
    with pytest.raises(ValueError):
        fixed_chunks(RUNBOOKS[0], 10, 10)
    with pytest.raises(ValueError):
        sentence_chunks(RUNBOOKS[0], 0)
    with pytest.raises(ValueError):
        chunk_corpus(RUNBOOKS, "semantic")


def test_chunk_ids_are_stable() -> None:
    a = chunk_corpus(RUNBOOKS, "sentence", max_chars=300)
    b = chunk_corpus(RUNBOOKS, "sentence", max_chars=300)
    assert [c.id for c in a] == [c.id for c in b] and len({c.id for c in a}) == len(a)


# ---- similarity --------------------------------------------------------------------------------
def test_unit_vector_identities_and_scale_invariance() -> None:
    a, b = HashEmbedder().embed(["rollback the deploy", "restore previous release"])
    assert dot(a, b) == pytest.approx(cosine(a, b), abs=1e-5)
    assert euclidean_sq(a, b) == pytest.approx(2 - 2 * cosine(a, b), abs=1e-5)
    assert cosine(3 * a, 0.5 * b) == pytest.approx(cosine(a, b), abs=1e-5)
    assert dot(3 * a, b) != pytest.approx(dot(a, b))  # dot product is NOT scale invariant


def test_cosine_rejects_zero_vector() -> None:
    with pytest.raises(ValueError):
        cosine(np.zeros(3), np.ones(3))


# ---- exact index -------------------------------------------------------------------------------
def _built_index() -> tuple[ExactIndex, HashEmbedder, list[Chunk]]:
    emb = HashEmbedder()
    chunks = chunk_corpus(RUNBOOKS, "sentence", max_chars=300)
    idx = ExactIndex(emb.dim)
    idx.add([c.id for c in chunks], emb.embed([c.text for c in chunks]), [{"tags": c.tags, "doc": c.doc_id} for c in chunks])
    return idx, emb, chunks


def test_exact_index_finds_lexical_match_and_sorts_scores() -> None:
    idx, emb, _ = _built_index()
    hits = idx.search(emb.embed(["primary postgres is unreachable, promote a replica"])[0], k=5)
    assert hits[0].metadata["doc"] == "rb-db-failover"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_exact_index_prefilter_returns_k_and_respects_filter() -> None:
    idx, emb, _ = _built_index()
    hits = idx.search(emb.embed(["slow requests"])[0], k=3, where=lambda m: "security" in m["tags"])
    assert hits and all("security" in h.metadata["tags"] for h in hits)
    assert idx.search(emb.embed(["x"])[0], k=3, where=lambda m: False) == []


def test_exact_index_validation() -> None:
    idx, emb, _ = _built_index()
    with pytest.raises(ValueError, match="normalised"):
        idx.add(["new"], np.ones((1, emb.dim), dtype=np.float32))
    with pytest.raises(ValueError, match="duplicate"):
        idx.add([idx._ids[0]], emb.embed(["a"]))
    with pytest.raises(ValueError, match="dim"):
        idx.search(np.ones(3, dtype=np.float32))


def test_hash_embedder_is_lexical_not_semantic() -> None:
    """The honest limit: a paraphrase that shares no content words with its target does not rank it first.

    (Some paraphrases still work through incidental overlap, e.g. 'lost', 'new'. This one has none.)"""
    idx, emb, _ = _built_index()
    hits = idx.search(emb.embed(["everything slowed down when the memoisation layer was wiped"])[0], k=1)
    assert hits[0].metadata["doc"] != "rb-cache-stampede"


def test_synthetic_and_recall_helpers() -> None:
    corpus, queries = synthetic_vectors(500, 16, 5, seed=1)
    assert np.allclose(np.linalg.norm(corpus, axis=1), 1.0, atol=1e-5)
    truth = exact_topk(corpus, queries, 5)
    assert recall_at_k([t.tolist() for t in truth], truth) == 1.0
    assert recall_at_k([[-1] * 5 for _ in truth], truth) == 0.0
    assert normalize(np.zeros((1, 4), dtype=np.float32)).sum() == 0  # zero vector stays zero, no NaN


# ---- pgvector (live) ---------------------------------------------------------------------------
@pytest.mark.infra
@pytest.mark.skipif(not postgres_available(require_pgvector=True), reason="Postgres+pgvector unavailable")
def test_pgvector_matches_exact_index_and_upsert_is_idempotent() -> None:
    idx, emb, chunks = _built_index()
    vecs = emb.embed([c.text for c in chunks])
    q = emb.embed(["queue keeps growing and consumers restart in a loop"])[0]
    with temp_schema(require_pgvector=True) as (url, schema):
        store = PgVectorStore(url, schema, emb.dim)
        store.create_table()
        store.upsert(chunks, vecs)
        store.upsert(chunks, vecs)
        assert store.count() == len(chunks)
        store.create_hnsw_index()
        pg = store.search(q, 5, exact=True)
        ref = idx.search(q, 5)
        assert [h.id for h in pg] == [h.id for h in ref]
        assert pg[0].score == pytest.approx(ref[0].score, abs=1e-4)
        assert [h.id for h in store.search(q, 5, ef_search=100)][0] == ref[0].id
        sec = store.search(emb.embed(["leaked credential"])[0], 3, tag="security")
        assert sec and all("security" in h.metadata["tags"] for h in sec)


@pytest.mark.infra
@pytest.mark.skipif(not postgres_available(require_pgvector=True), reason="Postgres+pgvector unavailable")
def test_explain_shows_index_vs_seq_scan_and_validates_input() -> None:
    corpus, queries = synthetic_vectors(3000, 32, 1, seed=2)
    chunks = [Chunk(id=f"v{i}", doc_id="d", index=i, text="", start=0, end=0) for i in range(len(corpus))]
    with temp_schema(require_pgvector=True) as (url, schema):
        store = PgVectorStore(url, schema, 32)
        store.create_table()
        store.upsert(chunks, corpus)
        store.create_hnsw_index()
        assert "hnsw" in store.explain(queries[0], 5).lower()
        assert "hnsw" not in store.explain(queries[0], 5, exact=True).lower()
        with pytest.raises(ValueError):
            store.search(queries[0], 5, iterative_scan="nope")
        with pytest.raises(ValueError):
            PgVectorStore(url, "bad; DROP TABLE x", 32)
        with pytest.raises(ValueError):
            store.upsert(chunks[:1], np.zeros((1, 8), dtype=np.float32))


@pytest.mark.infra
@pytest.mark.skipif(not postgres_available(require_pgvector=True), reason="Postgres+pgvector unavailable")
def test_selective_filter_underfills_hnsw_unless_iterative_scan() -> None:
    n, dim = 4000, 32
    corpus, queries = synthetic_vectors(n, dim, 1, seed=3)
    chunks = [Chunk(id=f"v{i}", doc_id="d", index=i, text="", start=0, end=0, tags=["rare"] if i % 100 == 0 else ["common"]) for i in range(n)]
    with temp_schema(require_pgvector=True) as (url, schema):
        store = PgVectorStore(url, schema, dim)
        store.create_table()
        store.upsert(chunks, corpus)
        store.create_hnsw_index()
        exact = store.search(queries[0], 10, tag="rare", exact=True)
        assert len(exact) == 10
        # At this table size the planner may pick a sequential scan for the filtered query and
        # return full results. To show the index-scan behaviour deterministically we forbid
        # non-index plans on this connection.
        conn = store._connect()
        try:
            conn.execute("SET enable_seqscan = off")
            conn.execute("SET enable_bitmapscan = off")
            approx = store.search(queries[0], 10, tag="rare", ef_search=40, conn=conn)
            iterative = store.search(queries[0], 10, tag="rare", ef_search=40, iterative_scan="relaxed_order", conn=conn)
        finally:
            conn.close()
        assert len(approx) < 10  # filter applied after the index returned its ef_search candidates
        assert len(iterative) == 10


@pytest.mark.infra
@pytest.mark.skipif(not postgres_available(require_pgvector=True), reason="Postgres+pgvector unavailable")
def test_ann_experiment_recall_rises_with_ef_search() -> None:
    rows, build_s = run_experiment(n=4000, dim=32, n_queries=20, ef_values=(10, 400), noise=2.0)
    by = {r.setting: r for r in rows}
    assert by["exact (seq scan)"].recall == 1.0
    assert by["hnsw ef_search=400"].recall >= by["hnsw ef_search=10"].recall
    assert by["hnsw ef_search=400"].recall > 0.9
    assert build_s > 0
