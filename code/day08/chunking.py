"""Chunking: turn documents into retrievable units while keeping offsets and provenance.

Two strategies, both returning `Chunk` objects whose `text == source[start:end]`:

* `fixed_chunks`    - sliding character window with overlap. Predictable, ignores structure.
* `sentence_chunks` - greedily packs whole sentences up to `max_chars`, carrying the last
                      `overlap_sentences` sentences into the next chunk. Never cuts mid-sentence
                      unless a single sentence alone exceeds `max_chars`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from common.fixtures import Doc

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class Chunk(BaseModel):
    id: str  # "<doc_id>#<index>" - stable as long as the document text and chunk settings are stable
    doc_id: str
    index: int
    text: str
    start: int  # character offsets into the source text
    end: int
    title: str = ""
    tags: list[str] = Field(default_factory=list)


def _make(doc: Doc, index: int, start: int, end: int) -> Chunk:
    return Chunk(
        id=f"{doc['id']}#{index}", doc_id=doc["id"], index=index, text=doc["text"][start:end],
        start=start, end=end, title=doc["title"], tags=list(doc["tags"]),
    )


def fixed_chunks(doc: Doc, size: int = 300, overlap: int = 50) -> list[Chunk]:
    if size < 1:
        raise ValueError("size must be >= 1")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    text = doc["text"]
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(_make(doc, len(chunks), start, end))
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _SENTENCE_END.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans


def sentence_chunks(doc: Doc, max_chars: int = 300, overlap_sentences: int = 1) -> list[Chunk]:
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")
    text = doc["text"]
    # A sentence longer than max_chars is split by the fixed strategy so no chunk exceeds the limit.
    spans: list[tuple[int, int]] = []
    for s, e in _sentence_spans(text):
        if e - s <= max_chars:
            spans.append((s, e))
        else:
            spans.extend((s + c.start, s + c.end) for c in fixed_chunks({**doc, "text": text[s:e]}, max_chars, 0))

    chunks: list[Chunk] = []
    cur: list[tuple[int, int]] = []
    for span in spans:
        if cur and span[1] - cur[0][0] > max_chars:
            chunks.append(_make(doc, len(chunks), cur[0][0], cur[-1][1]))
            carry = cur[len(cur) - overlap_sentences :] if overlap_sentences else []
            # Drop the carried overlap if it would leave no room for the new sentence (guarantees progress).
            while carry and span[1] - carry[0][0] > max_chars:
                carry = carry[1:]
            cur = list(carry)
        cur.append(span)
    if cur:
        chunks.append(_make(doc, len(chunks), cur[0][0], cur[-1][1]))
    return chunks


def chunk_corpus(docs: Sequence[Doc], strategy: str = "sentence", **kwargs: int) -> list[Chunk]:
    fn = {"sentence": sentence_chunks, "fixed": fixed_chunks}.get(strategy)
    if fn is None:
        raise ValueError(f"unknown chunking strategy '{strategy}'; expected 'sentence' or 'fixed'")
    out: list[Chunk] = []
    for d in docs:
        out.extend(fn(d, **kwargs))  # type: ignore[arg-type]
    return out
