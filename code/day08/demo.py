"""Day 8 demo: chunking, exact search, pgvector, and the measured ANN trade-off. Runs offline."""

from __future__ import annotations

from common.embeddings import HashEmbedder
from common.fixtures import RUNBOOKS
from common.infra import postgres_available, temp_schema
from day08.ann_experiment import format_table, run_experiment
from day08.chunking import chunk_corpus, fixed_chunks, sentence_chunks
from day08.exact_index import ExactIndex
from day08.pgvector_store import PgVectorStore
from day08.similarity import cosine, dot, euclidean_sq


def show_chunking() -> None:
    doc = RUNBOOKS[0]
    print(f"== chunking {doc['id']} ({len(doc['text'])} chars)")
    for name, chunks in (("fixed(150, overlap 30)", fixed_chunks(doc, 150, 30)), ("sentence(200, overlap 1)", sentence_chunks(doc, 200, 1))):
        print(f"-- {name}: {len(chunks)} chunks")
        for c in chunks:
            print(f"   {c.id:<24} [{c.start:>3}:{c.end:<3}] {c.text[:60]!r}...{c.text[-24:]!r}")


def show_metrics() -> None:
    emb = HashEmbedder()
    a, b = emb.embed(["roll back the bad deploy", "redeploy the previous version"])
    print("\n== metrics on unit vectors")
    print(f"cosine={cosine(a, b):.4f} dot={dot(a, b):.4f} euclid^2={euclidean_sq(a, b):.4f} (2-2*cos={2 - 2 * cosine(a, b):.4f})")


def show_search() -> tuple[list, object]:
    emb = HashEmbedder()
    chunks = chunk_corpus(RUNBOOKS, "sentence", max_chars=300, overlap_sentences=1)
    vecs = emb.embed([c.text for c in chunks])
    index = ExactIndex(emb.dim)
    index.add([c.id for c in chunks], vecs, [{"tags": c.tags} for c in chunks])
    print(f"\n== exact search over {len(chunks)} chunks from {len(RUNBOOKS)} runbooks (HashEmbedder, lexical)")
    for q in ("primary postgres is unreachable, promote a replica", "the dataset store lost its leader, how to elect a new one"):
        print(f"query: {q!r}")
        for h in index.search(emb.embed([q])[0], k=3):
            print(f"   {h.score:.3f}  {h.id}")
    print("tag filter 'security':", [h.id for h in index.search(emb.embed(["credential leaked"])[0], k=3, where=lambda m: "security" in m["tags"])])
    return chunks, vecs


def show_pgvector(chunks: list, vecs: object) -> None:
    emb = HashEmbedder()
    q = emb.embed(["queue keeps growing and consumers restart in a loop"])[0]
    with temp_schema(require_pgvector=True) as (url, schema):
        store = PgVectorStore(url, schema, emb.dim)
        store.create_table()
        store.upsert(chunks, vecs)  # type: ignore[arg-type]
        store.create_hnsw_index()
        print(f"\n== pgvector: {store.count()} rows, HNSW built")
        print("top-3:", [(h.id, round(h.score, 3)) for h in store.search(q, 3)])
        print("plan (default, tiny table):", store.explain(q, 3).splitlines()[1].strip())
        print("plan (exact=True):         ", store.explain(q, 3, exact=True).splitlines()[1].strip())


def main() -> None:
    show_chunking()
    show_metrics()
    chunks, vecs = show_search()
    if not postgres_available(require_pgvector=True):
        print("\n(Postgres+pgvector unreachable: skipping pgvector and ANN sections. Set DATABASE_URL.)")
        return
    show_pgvector(chunks, vecs)
    print("\n== ANN vs exact: 50,000 synthetic vectors, dim 128, 100 queries, k=10, HNSW m=16 ef_construction=64")
    rows, build_s = run_experiment(n=50000, dim=128, n_queries=100, k=10, noise=2.0)
    print(format_table(rows))
    print(f"index build: {build_s:.1f}s (timings are from this machine, shared with other processes)")


if __name__ == "__main__":
    main()
