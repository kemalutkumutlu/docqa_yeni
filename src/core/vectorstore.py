from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection

from .models import Chunk


def _chunk_metadata(c: Chunk) -> Dict[str, Any]:
    return {
        "doc_id": c.doc_id,
        "file_name": c.file_name,
        "section_id": c.section_id,
        "parent_id": c.parent_id or "",
        "heading_path": c.heading_path,
        "page_start": c.page_start,
        "page_end": c.page_end,
        "kind": c.kind,
        "modality": c.modality,
        "image_path": c.image_path,
        "region_label": c.region_label,
        "region_id": c.region_id,
        "crop_type": c.crop_type,
        "region_summary": c.region_summary,
        "proposal_source": c.proposal_source,
        "proposal_confidence": c.proposal_confidence,
        "bbox_left": c.bbox_left,
        "bbox_top": c.bbox_top,
        "bbox_right": c.bbox_right,
        "bbox_bottom": c.bbox_bottom,
        "bbox_left_norm": c.bbox_left_norm,
        "bbox_top_norm": c.bbox_top_norm,
        "bbox_right_norm": c.bbox_right_norm,
        "bbox_bottom_norm": c.bbox_bottom_norm,
    }


@dataclass
class ChromaStore:
    persist_dir: str
    collection_name: str = "chunks"

    _client: chromadb.PersistentClient | None = None
    _collection: Collection | None = None

    def _get_collection(self) -> Collection:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(name=self.collection_name)
        return self._collection

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have same length")
        col = self._get_collection()
        try:
            col.upsert(
                ids=[c.chunk_id for c in chunks],
                embeddings=embeddings,
                documents=[c.text for c in chunks],
                metadatas=[_chunk_metadata(c) for c in chunks],
            )
        except Exception as e:  # noqa: BLE001
            msg = str(e) or e.__class__.__name__
            if "expecting embedding with dimension" in msg.lower():
                # Common case when an existing persistent Chroma directory was created with a
                # different embedding model (e.g., e5-small=384 dims vs e5-base=768 dims).
                m = re.search(r"dimension of\s+(?P<exp>\d+)\s*,\s*got\s+(?P<got>\d+)", msg, flags=re.IGNORECASE)
                exp = m.group("exp") if m else "?"
                got = m.group("got") if m else "?"
                hint = ""
                if exp == "384" and got == "768":
                    hint = " (muhtemelen e5-small -> e5-base gecisi)"
                elif exp == "768" and got == "384":
                    hint = " (muhtemelen e5-base -> e5-small gecisi)"

                raise ValueError(
                    "Embedding boyutu uyusmuyor: mevcut Chroma index'i "
                    f"{exp} boyut bekliyor, ama bu calistirma {got} boyut uretiyor{hint}.\n\n"
                    "Cozum (birini secin):\n"
                    f"- CHROMA_DIR'i yeni/bos bir klasore alin (ornegin `CHROMA_DIR=./data/chroma_{got}`) ve tekrar indeksleyin\n"
                    "- veya mevcut index'i silip yeniden olusturun (ornegin `data/chroma/` klasorunu temizleyin)\n"
                    "- veya EMBEDDING_MODEL'i eski modele geri alin (index ile ayni boyuta gelecek sekilde)\n"
                ) from e
            raise

    def query(
        self,
        query_embedding: List[float],
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> dict:
        col = self._get_collection()
        return col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def get(self, ids: List[str]) -> dict:
        col = self._get_collection()
        return col.get(ids=ids, include=["documents", "metadatas"])

    def delete_where(self, where: Dict[str, Any]) -> None:
        """
        Delete records matching a metadata filter.

        This is used to prevent stale chunks for the same doc_id from accumulating
        across rebuilds (and to avoid cross-document contamination in persistent stores).
        """
        col = self._get_collection()
        col.delete(where=where)

