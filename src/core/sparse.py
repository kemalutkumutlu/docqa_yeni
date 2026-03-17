from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from rank_bm25 import BM25Okapi

from .models import Chunk


def simple_tokenize(text: str) -> List[str]:
    # TR/EN friendly-ish tokenizer (deterministic, dependency-free)
    text = text.lower()
    text = re.sub(r"[^0-9a-zçğıöşüâîû\-]+", " ", text)
    tokens = [t for t in text.split() if t]

    # Language-agnostic robustness:
    # add character 3-grams to improve matching for suffixes/typos (e.g., "adres" vs "adresini")
    # Threshold lowered to 4 chars to cover short Turkish words (tablo, form, amaç, test)
    ngrams: list[str] = []
    for tok in tokens:
        # avoid exploding token count on very long strings (IDs/addresses)
        if len(tok) < 4 or len(tok) > 24:
            continue
        # generate up to 12 trigrams per token
        limit = min(len(tok) - 2, 12)
        for i in range(limit):
            ng = tok[i : i + 3]
            ngrams.append(f"~{ng}")

    return tokens + ngrams


def _safe_bm25(corpus: List[List[str]]) -> BM25Okapi:
    # rank_bm25 raises ZeroDivisionError when every document token list is empty.
    # Use a private placeholder corpus for the degenerate case; callers already
    # guard on `self.ids`, so this placeholder is never surfaced to retrieval.
    non_empty = [doc for doc in corpus if doc]
    if not non_empty:
        return BM25Okapi([["__empty__"]])
    return BM25Okapi(non_empty)


@dataclass
class BM25Index:
    # chunk_id -> token list
    tokens_by_id: Dict[str, List[str]]
    bm25: BM25Okapi
    ids: List[str]

    @classmethod
    def build(cls, chunks: List[Chunk]) -> "BM25Index":
        # Index child and table chunks only.
        # Parent chunks are excluded: they contain heading + full body, which is a
        # superset of child chunk content.  Indexing parents alongside children causes
        # the same section to appear multiple extra times in BM25 results, dominating
        # hybrid top-k and reducing retrieval diversity.
        # Heading keyword matching is covered by the [heading_path] prefix that is
        # prepended to every child chunk during chunking.
        # Visual/toc chunks are excluded — they don't carry searchable text.
        indexable = [c for c in chunks if c.kind in ("child", "table")]
        pairs = [(c.chunk_id, simple_tokenize(c.text)) for c in indexable]
        pairs = [(cid, toks) for cid, toks in pairs if toks]
        ids = [cid for cid, _ in pairs]
        toks = [toks for _, toks in pairs]
        bm25 = _safe_bm25(toks)
        return cls(tokens_by_id=dict(zip(ids, toks)), bm25=bm25, ids=ids)

    def search(
        self,
        query: str,
        top_k: int,
        *,
        doc_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        q = simple_tokenize(query)
        if not q or not self.ids:
            return []
        scores = self.bm25.get_scores(q)

        # Optional doc filter (used for multi-doc session isolation).
        prefixes: Optional[tuple[str, ...]] = None
        if doc_ids:
            prefixes = tuple(f"{did}:" for did in sorted(doc_ids))

        # Rank all indices by score, then filter while collecting top_k matches.
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: List[Tuple[str, float]] = []
        for i in ranked:
            s = float(scores[i])
            if s <= 0:
                continue
            cid = self.ids[i]
            if prefixes is not None and not cid.startswith(prefixes):
                continue
            out.append((cid, s))
            if len(out) >= top_k:
                break
        return out

    def extend(self, new_chunks: List[Chunk]) -> "BM25Index":
        """
        Add new chunks to the index.  rank-bm25 does not support incremental
        updates, so we rebuild the BM25 object from the combined token lists.
        This is still much cheaper than re-embedding because tokenization is
        CPU-only and fast.
        """
        new_indexable = [c for c in new_chunks if c.kind in ("child", "table")]
        if not new_indexable:
            return self

        pairs = [(c.chunk_id, simple_tokenize(c.text)) for c in new_indexable]
        pairs = [(cid, toks) for cid, toks in pairs if toks]
        if not pairs:
            return self
        new_ids = [cid for cid, _ in pairs]
        new_toks = [toks for _, toks in pairs]

        # Extend internal state
        combined_ids = self.ids + new_ids
        combined_toks = [self.tokens_by_id[cid] for cid in self.ids] + new_toks
        combined_map = dict(zip(combined_ids, combined_toks))

        # Rebuild BM25 from combined tokens
        bm25 = _safe_bm25(combined_toks) if combined_toks else self.bm25

        self.ids = combined_ids
        self.tokens_by_id = combined_map
        self.bm25 = bm25
        return self

    def remove_doc_ids(self, doc_ids: Set[str]) -> "BM25Index":
        """
        Remove all entries belonging to given doc_ids.

        This is used when a document is re-indexed (same doc_id) and we want to
        avoid accumulating duplicate chunk_ids in the sparse index.
        """
        if not doc_ids or not self.ids:
            return self
        prefixes = tuple(f"{did}:" for did in sorted(doc_ids) if did)
        if not prefixes:
            return self

        keep_ids = [cid for cid in self.ids if not cid.startswith(prefixes)]
        if len(keep_ids) == len(self.ids):
            return self

        # Keep token map in sync.
        self.ids = keep_ids
        self.tokens_by_id = {cid: self.tokens_by_id[cid] for cid in keep_ids if cid in self.tokens_by_id}
        corpus = [self.tokens_by_id.get(cid, []) for cid in keep_ids]
        self.bm25 = _safe_bm25(corpus)
        return self

    # ── Persistence ──────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Persist BM25 state to disk (tokens + ids only; BM25Okapi is rebuilt on load)."""
        from pathlib import Path as _P
        _P(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "tokens_by_id": self.tokens_by_id,
            "ids": self.ids,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """Load BM25 state from disk and rebuild BM25Okapi."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokens_by_id: Dict[str, List[str]] = data["tokens_by_id"]
        ids: List[str] = data["ids"]
        corpus = [tokens_by_id[cid] for cid in ids]
        filtered = [(cid, toks) for cid, toks in zip(ids, corpus) if toks]
        ids = [cid for cid, _ in filtered]
        tokens_by_id = {cid: toks for cid, toks in filtered}
        corpus = [toks for _, toks in filtered]
        bm25 = _safe_bm25(corpus)
        return cls(tokens_by_id=tokens_by_id, bm25=bm25, ids=ids)

