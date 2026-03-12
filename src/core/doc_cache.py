from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Optional

from .models import Chunk


@dataclass(frozen=True)
class CachedDocument:
    doc_id: str
    file_name: str
    page_count: int
    warnings: list[str]
    build_fingerprint: str
    collection_name: str
    chunks: list[Chunk]


class DocumentCache:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def _fingerprint_key(self, fingerprint: str) -> str:
        return sha256(fingerprint.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self, doc_id: str, fingerprint: str) -> Path:
        fp_key = self._fingerprint_key(fingerprint)
        return self.root_dir / doc_id / f"{fp_key}.json"

    def load(self, doc_id: str, fingerprint: str) -> Optional[CachedDocument]:
        path = self._cache_path(doc_id, fingerprint)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in payload.get("chunks", [])]
        return CachedDocument(
            doc_id=payload["doc_id"],
            file_name=payload["file_name"],
            page_count=int(payload["page_count"]),
            warnings=list(payload.get("warnings", [])),
            build_fingerprint=payload["build_fingerprint"],
            collection_name=payload.get("collection_name", "chunks"),
            chunks=chunks,
        )

    def save(
        self,
        *,
        doc_id: str,
        file_name: str,
        page_count: int,
        warnings: list[str],
        build_fingerprint: str,
        collection_name: str,
        chunks: list[Chunk],
    ) -> Path:
        path = self._cache_path(doc_id, build_fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "doc_id": doc_id,
            "file_name": file_name,
            "page_count": page_count,
            "warnings": warnings,
            "build_fingerprint": build_fingerprint,
            "collection_name": collection_name,
            "chunks": [asdict(chunk) for chunk in chunks],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return path
