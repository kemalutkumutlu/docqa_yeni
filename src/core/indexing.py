from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import List, Optional, Set

from .embedding import Embedder
from .hybrid import HybridResult, rrf_fuse
from .models import Chunk
from .sparse import BM25Index
from .vectorstore import ChromaStore


_CHROMA_WRITE_LOCKS: dict[str, threading.Lock] = {}
_CHROMA_WRITE_LOCKS_GUARD = threading.Lock()


def _chroma_write_lock(chroma_dir: Path | str) -> threading.Lock:
    key = str(Path(chroma_dir).resolve())
    with _CHROMA_WRITE_LOCKS_GUARD:
        lock = _CHROMA_WRITE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _CHROMA_WRITE_LOCKS[key] = lock
        return lock


@dataclass
class LocalIndex:
    """
    MVP index: Chroma (dense) + in-memory BM25 (sparse).

    - Store all chunks in Chroma (parents + children)
    - Build BM25 over child chunks only
    """

    embedder: Embedder
    store: ChromaStore
    bm25: BM25Index
    allowed_doc_ids: Set[str]
    chroma_dir: Optional[str] = None  # used to derive BM25 persist path

    @property
    def bm25_path(self) -> Optional[str]:
        """Path where BM25 index is persisted (alongside ChromaDB)."""
        if self.chroma_dir:
            return str(Path(self.chroma_dir) / "bm25_index.json")
        return None

    @classmethod
    def build(
        cls,
        chunks: List[Chunk],
        chroma_dir: Path,
        embedding_model: str,
        embedding_device: str = "auto",
        embedding_dimension: int = 3072,
        collection_name: str = "chunks",
    ) -> "LocalIndex":
        embedder = Embedder(
            model_name=embedding_model,
            device=embedding_device,
            output_dimension=embedding_dimension,
        )  # type: ignore[arg-type]
        # Embed first so we can derive embedding dimension deterministically.
        embeddings = embedder.embed_chunks(chunks)
        emb_dim = len(embeddings[0]) if embeddings else 0

        # Automation: avoid "384 vs 768" dimension mismatches when reusing the same persistent
        # CHROMA_DIR with a different embedding model (e.g. e5-small vs e5-base).
        #
        # If caller didn't explicitly override the collection name (default "chunks"),
        # namespace the collection by embedding dimension.
        collection_name_use = collection_name
        if collection_name == "chunks" and emb_dim:
            collection_name_use = f"chunks_d{emb_dim}"

        store = ChromaStore(persist_dir=str(chroma_dir), collection_name=collection_name_use)
        with _chroma_write_lock(chroma_dir):
            # Prevent stale chunks accumulating for the same doc_id(s) in the persistent store.
            doc_ids = sorted({c.doc_id for c in chunks})
            if doc_ids:
                if len(doc_ids) == 1:
                    store.delete_where(where={"doc_id": doc_ids[0]})
                else:
                    store.delete_where(where={"$or": [{"doc_id": did} for did in doc_ids]})

            # Upsert all chunks (parents + children)
            store.upsert_chunks(chunks, embeddings=embeddings)

        bm25 = BM25Index.build(chunks)
        allowed_doc_ids = {c.doc_id for c in chunks}
        idx = cls(
            embedder=embedder, store=store, bm25=bm25,
            allowed_doc_ids=allowed_doc_ids, chroma_dir=str(chroma_dir),
        )
        # Persist BM25 alongside ChromaDB
        if idx.bm25_path:
            bm25.save(idx.bm25_path)
        return idx

    @classmethod
    def from_persisted(
        cls,
        *,
        chroma_dir: Path,
        embedding_model: str,
        embedding_device: str = "auto",
        embedding_dimension: int = 3072,
        allowed_doc_ids: Set[str],
        collection_name: str = "chunks",
        chunks: Optional[List[Chunk]] = None,
    ) -> "LocalIndex":
        embedder = Embedder(
            model_name=embedding_model,
            device=embedding_device,
            output_dimension=embedding_dimension,
        )  # type: ignore[arg-type]
        store = ChromaStore(persist_dir=str(chroma_dir), collection_name=collection_name)
        bm25_path = Path(chroma_dir) / "bm25_index.json"
        if bm25_path.exists():
            bm25 = BM25Index.load(str(bm25_path))
        else:
            bm25 = BM25Index.build(chunks or [])
            if chunks:
                bm25.save(str(bm25_path))
        return cls(
            embedder=embedder,
            store=store,
            bm25=bm25,
            allowed_doc_ids=set(allowed_doc_ids),
            chroma_dir=str(chroma_dir),
        )

    def add_chunks(self, new_chunks: List[Chunk]) -> None:
        """
        Incrementally add chunks from a new document WITHOUT re-embedding
        previously indexed chunks.  Only the new chunks are embedded and
        upserted into Chroma; BM25 is extended (cheap CPU-only rebuild).

        This turns multi-document indexing from O(N_total) embeddings per
        add_document call to O(N_new) embeddings.
        """
        if not new_chunks:
            return

        with _chroma_write_lock(self.chroma_dir or ""):
            # Clean stale chunks for the new doc_id(s) in Chroma.
            new_doc_ids = sorted({c.doc_id for c in new_chunks})
            if new_doc_ids:
                if len(new_doc_ids) == 1:
                    self.store.delete_where(where={"doc_id": new_doc_ids[0]})
                else:
                    self.store.delete_where(where={"$or": [{"doc_id": did} for did in new_doc_ids]})

            # If any of these doc_ids were already indexed, we are REPLACING them.
            # Chroma is cleaned above; we must also prevent sparse duplicates.
            already_indexed = set(new_doc_ids) & set(self.allowed_doc_ids)
            if already_indexed:
                self.bm25.remove_doc_ids(already_indexed)

            # Embed only the NEW chunks.
            embeddings = self.embedder.embed_chunks(new_chunks)
            self.store.upsert_chunks(new_chunks, embeddings=embeddings)

            # Extend BM25 incrementally.
            self.bm25.extend(new_chunks)

            # Track new doc_ids.
            self.allowed_doc_ids.update(new_doc_ids)

            # Persist updated BM25
            if self.bm25_path:
                self.bm25.save(self.bm25_path)

    def _where_allowed_docs(self) -> Optional[dict]:
        """
        Build a Chroma `where` filter to avoid cross-document contamination.
        We never rely on clearing the persistent store; instead we restrict queries
        to doc_ids that were used to build this index instance.
        """
        if not self.allowed_doc_ids:
            return None
        if len(self.allowed_doc_ids) == 1:
            return {"doc_id": next(iter(self.allowed_doc_ids))}
        return {"$or": [{"doc_id": did} for did in sorted(self.allowed_doc_ids)]}

    def dense_search(self, query: str, top_k: int, where: Optional[dict] = None) -> List[str]:
        qemb = self.embedder.embed_query(query)
        # Always restrict dense search to the documents in this index (unless caller supplies where)
        where_final = where or self._where_allowed_docs()
        res = self.store.query(qemb, top_k=top_k, where=where_final)
        # Chroma returns list-of-lists
        return list(res.get("ids", [[]])[0])

    @staticmethod
    def _where_doc_ids(doc_ids: Set[str]) -> Optional[dict]:
        if not doc_ids:
            return None
        if len(doc_ids) == 1:
            return {"doc_id": next(iter(doc_ids))}
        return {"$or": [{"doc_id": did} for did in sorted(doc_ids)]}

    def sparse_search(self, query: str, top_k: int, *, doc_ids: Optional[Set[str]] = None) -> List[str]:
        allowed = set(self.allowed_doc_ids)
        if doc_ids is None:
            doc_ids_use = allowed
        else:
            doc_ids_use = allowed & set(doc_ids)
        return [cid for cid, _ in self.bm25.search(query, top_k=top_k, doc_ids=doc_ids_use)]

    def hybrid_search(
        self,
        query: str,
        dense_k: int = 10,
        sparse_k: int = 10,
        final_k: int = 10,
        *,
        doc_ids: Optional[Set[str]] = None,
    ) -> HybridResult:
        doc_ids_use: Optional[Set[str]] = None
        if doc_ids:
            # Only allow doc_ids that belong to this index instance.
            filtered = set(doc_ids) & set(self.allowed_doc_ids)
            if not filtered:
                # IMPORTANT: if caller asked to restrict to doc_ids but none are allowed,
                # we must NOT fall back to an unrestricted search (that would leak other docs).
                return HybridResult(ids=[], scores={})
            doc_ids_use = filtered

        where = self._where_doc_ids(doc_ids_use) if doc_ids_use else None
        dense_ids = self.dense_search(query, top_k=dense_k, where=where)
        sparse_ids = self.sparse_search(query, top_k=sparse_k, doc_ids=doc_ids_use)
        fused = rrf_fuse(dense_ids=dense_ids, sparse_ids=sparse_ids)
        top = fused[:final_k]
        ids = [cid for cid, _ in top]
        return HybridResult(ids=ids, scores=dict(top))

