"""
Full RAG pipeline — single entry point that wires:
  ingest → structure → chunk → index → retrieve → generate

Used by:
  - Chainlit UI (app.py)
  - CLI scripts
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional
from uuid import uuid4

from .generation import (
    GenerationResult,
    generate_answer,
    generate_answer_openai,
    generate_answer_stream,
    generate_answer_openai_stream,
    generate_answer_local,
    generate_answer_local_stream,
    generate_chat_answer,
    generate_chat_answer_openai,
    generate_chat_answer_local,
    generate_extractive_answer,
)
from .content_normalization import CONTENT_NORMALIZER_VERSION
from .doc_cache import CachedDocument, DocumentCache
from .local_llm import OllamaConfig
from .eventlog import JsonlEventLogger
from .indexing import LocalIndex
from .ingestion import ingest_any, IngestResult
from .layout_detector import resolve_sidecar_path
from .multimodal import MultimodalConfig, visual_chunks_from_ingest
from .ocr_backend import OCRConfig
from .table_structure import TableStructureConfig, table_chunks_from_ingest
from .vlm_extract import VLMConfig
from .models import Chunk
from .retrieval import RetrievalResult, retrieve, classify_query
from .structure import build_section_tree, section_tree_to_chunks
from .utils import sha256_file
from .vectorstore import ChromaStore


@dataclass
class DocumentState:
    """Holds parsed state for one uploaded document."""
    doc_id: str
    file_name: str
    chunks: List[Chunk]
    page_count: int
    warnings: List[str]
    build_fingerprint: str = ""
    collection_name: str = ""
    restored_from_cache: bool = False


@dataclass
class RAGPipeline:
    """
    Stateful RAG pipeline that manages multiple documents in one session.
    """
    # Config
    embedding_model: str
    chroma_dir: Path
    gemini_api_key: str
    gemini_model: str
    ocr_config: OCRConfig
    embedding_device: str = "auto"
    processing_mode: str = "classic"
    multimodal_answer_mode: str = "auto"
    visual_chunk_level: str = "page"
    visual_region_source: str = "heuristic"
    visual_detector_backend: str = "none"
    visual_detector_dir: Optional[Path] = None
    docai_project_id: str = ""
    docai_location: str = "us"
    docai_layout_processor_id: str = ""
    docai_layout_processor_version: str = "pretrained-layout-parser-v1.6-pro-2025-12-01"
    docai_timeout_seconds: int = 120
    docling_python_bin: str = ""
    docling_layout_model: str = "docling-layout-heron-101"
    docling_artifacts_path: Optional[Path] = None
    docling_device: str = "auto"
    table_structure_config: Optional[TableStructureConfig] = None
    vlm_config: Optional[VLMConfig] = None
    llm_provider: str = "gemini"  # "gemini" | "openai" | "local" | "none"
    ollama_config: Optional[OllamaConfig] = None
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    multimodal_assets_dir: Optional[Path] = None

    # State
    _documents: Dict[str, DocumentState] = field(default_factory=dict)
    _index: Optional[LocalIndex] = None
    _all_chunks: List[Chunk] = field(default_factory=list)
    _active_doc_id: Optional[str] = None
    _session_id: str = field(default_factory=lambda: uuid4().hex)
    _logger: Optional[JsonlEventLogger] = None

    def _get_logger(self) -> Optional[JsonlEventLogger]:
        """
        Lazy-create logger from env. Logging is OFF by default.
        """
        if self._logger is not None:
            return self._logger
        self._logger = JsonlEventLogger.from_env()
        return self._logger

    @staticmethod
    def _summarize_evidence(retrieval: RetrievalResult, *, limit: int = 4) -> list[str]:
        if not any(
            getattr(ev, "region_label", "") or getattr(ev, "region_id", "") or getattr(ev, "crop_type", "page") != "page"
            for ev in retrieval.evidences
        ):
            return []
        lines: list[str] = []
        seen: set[str] = set()
        for ev in retrieval.evidences:
            file_name = (ev.heading_path.split(" / ", 1)[0].strip() if " / " in (ev.heading_path or "") else (ev.heading_path or "").strip()) or "Belge"
            page_text = f"Sayfa {ev.page_start}" if ev.page_start else "Sayfa ?"
            region_bits: list[str] = []
            if getattr(ev, "region_label", ""):
                region_bits.append(f"Region {ev.region_label}")
            if getattr(ev, "region_id", ""):
                region_bits.append(str(ev.region_id))
            if getattr(ev, "crop_type", "") and getattr(ev, "crop_type", "") != "page":
                region_bits.append(str(ev.crop_type))
            if getattr(ev, "proposal_source", ""):
                region_bits.append(f"src={ev.proposal_source}")
            if float(getattr(ev, "proposal_confidence", 0.0) or 0.0) > 0:
                region_bits.append(f"conf={float(ev.proposal_confidence):.2f}")
            location = page_text
            if region_bits:
                location += " | " + " | ".join(region_bits)
            summary = (getattr(ev, "region_summary", "") or "").strip()
            line = f"- `{file_name}` | {location}"
            if summary:
                line += f" | {summary}"
            if any(int(getattr(ev, name, 0) or 0) > 0 for name in ("bbox_right", "bbox_bottom")):
                line += (
                    f" | bbox=({int(getattr(ev, 'bbox_left', 0) or 0)},"
                    f"{int(getattr(ev, 'bbox_top', 0) or 0)},"
                    f"{int(getattr(ev, 'bbox_right', 0) or 0)},"
                    f"{int(getattr(ev, 'bbox_bottom', 0) or 0)})"
                )
            key = f"{file_name}:{location}:{summary}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(line)
            if len(lines) >= limit:
                break
        return lines

    def _doc_cache(self) -> DocumentCache:
        return DocumentCache(self.chroma_dir.parent / "doc_cache")

    def _persisted_doc_available(self, state: CachedDocument | DocumentState) -> bool:
        if not state.chunks:
            return True
        probe_chunk_id = state.chunks[0].chunk_id
        store = ChromaStore(
            persist_dir=str(self.chroma_dir),
            collection_name=state.collection_name or "chunks",
        )
        try:
            existing = store.get([probe_chunk_id])
        except Exception:
            return False
        ids = existing.get("ids") or []
        return probe_chunk_id in ids

    def _hydrate_index_from_persisted(self) -> None:
        if not self._all_chunks:
            self._index = None
            return
        collection_name = next(
            (
                st.collection_name
                for st in self._documents.values()
                if st.chunks and st.collection_name
            ),
            next(
                (st.collection_name for st in self._documents.values() if st.collection_name),
                "chunks",
            ),
        )
        self._index = LocalIndex.from_persisted(
            chroma_dir=self.chroma_dir,
            embedding_model=self.embedding_model,
            embedding_device=self.embedding_device,
            allowed_doc_ids=set(self._documents.keys()),
            collection_name=collection_name,
            chunks=self._all_chunks,
        )

    @property
    def session_id(self) -> str:
        return self._session_id

    def add_document(
        self,
        file_path: Path,
        display_name: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> DocumentState:
        """
        Ingest a document, build structure, create chunks, and rebuild index.
        Returns the document state.
        """
        def _progress(step: str) -> None:
            if not progress_callback:
                return
            try:
                progress_callback(step)
            except Exception:
                # Progress is best-effort and should never break ingestion/indexing.
                pass

        _progress("Belge kimligi kontrol ediliyor...")

        # If the file bytes are identical AND ingestion/index settings are identical,
        # skip re-processing entirely to avoid redundant OCR/VLM + embedding work.
        #
        # doc_id is sha256(file_bytes) (see ingestion.py). Computing it here is cheap
        # compared to OCR/VLM and embeddings, and allows a fast early-exit.
        doc_id = sha256_file(file_path)

        def _fingerprint() -> str:
            o = self.ocr_config
            v = self.vlm_config
            # Keep fingerprint stable and conservative: include only settings that can
            # affect extracted text/chunking or embedding consistency.
            parts = [
                f"emb_model={self.embedding_model}",
                f"emb_dev={self.embedding_device}",
                f"processing_mode={self.processing_mode}",
                f"visual_chunk_level={self.visual_chunk_level}",
                f"visual_region_source={self.visual_region_source}",
                f"visual_detector_backend={self.visual_detector_backend}",
                f"docai_location={self.docai_location}",
                f"docai_processor_id={self.docai_layout_processor_id}",
                f"docai_processor_version={self.docai_layout_processor_version}",
                f"docling_python_bin={self.docling_python_bin}",
                f"docling_layout_model={self.docling_layout_model}",
                f"docling_artifacts_path={self.docling_artifacts_path or ''}",
                f"docling_device={self.docling_device}",
                f"content_norm={CONTENT_NORMALIZER_VERSION}",
                f"ocr_enabled={bool(o.enabled)}",
                f"ocr_backend={getattr(o, 'backend', 'tesseract_legacy')}",
                f"ocr_lang={o.lang}",
                f"ocr_device={getattr(o, 'device', 'auto')}",
                f"ocr_paddle_version={getattr(o, 'paddle_ocr_version', 'PP-OCRv5')}",
                f"ocr_docai_project={getattr(o, 'docai_project_id', '') or ''}",
                f"ocr_docai_location={getattr(o, 'docai_location', '') or ''}",
                f"ocr_docai_processor_id={getattr(o, 'docai_processor_id', '') or ''}",
                f"ocr_docai_processor_version={getattr(o, 'docai_processor_version', '') or ''}",
                f"ocr_docai_timeout={int(getattr(o, 'docai_timeout_seconds', 120) or 120)}",
                f"tess_cmd={o.tesseract_cmd or ''}",
                f"tessdata={o.tessdata_prefix or ''}",
                f"tess_cfg={getattr(o, 'tesseract_config', None) or ''}",
            ]
            if v is None:
                parts.append("vlm=none")
            else:
                parts.extend(
                    [
                        f"vlm_provider={getattr(v, 'provider', 'gemini')}",
                        f"vlm_mode={v.mode}",
                        f"vlm_model={v.model}",
                        f"vlm_max_pages={v.max_pages}",
                        f"vlm_has_key={bool(v.api_key)}",
                    ]
                )
            t = self.table_structure_config
            if t is None:
                parts.append("table_structure=none")
            else:
                parts.extend(
                    [
                        f"table_enabled={bool(t.enabled)}",
                        f"table_backend={t.backend}",
                        f"table_min_conf={t.min_confidence}",
                        f"table_gemini_model={t.gemini_model}",
                        f"table_docai_project={t.docai_project_id}",
                        f"table_docai_location={t.docai_location}",
                        f"table_docai_processor_id={t.docai_processor_id}",
                        f"table_docai_processor_version={t.docai_processor_version}",
                    ]
                )
            if self.visual_region_source == "detector" and self.visual_detector_backend == "sidecar":
                sidecar_path = resolve_sidecar_path(self.visual_detector_dir, display_name or file_path.name)
                if sidecar_path and sidecar_path.exists():
                    parts.append(f"detector_sidecar={sha256_file(sidecar_path)}")
                else:
                    parts.append("detector_sidecar=missing")
            return "|".join(parts)

        fp = _fingerprint()
        existing = self._documents.get(doc_id)
        if existing and (existing.build_fingerprint == fp):
            # Treat this upload as a "select active document" action.
            self._active_doc_id = doc_id
            _progress("Ayni belge bulundu, yeniden isleme atlandi.")
            return existing

        cached = self._doc_cache().load(doc_id, fp)
        if cached and self._persisted_doc_available(cached):
            _progress("Ayni belge icin kalici cache bulundu, yeniden isleme atlandi.")
            state = DocumentState(
                doc_id=cached.doc_id,
                file_name=cached.file_name,
                chunks=cached.chunks,
                page_count=cached.page_count,
                warnings=cached.warnings,
                build_fingerprint=cached.build_fingerprint,
                collection_name=cached.collection_name,
                restored_from_cache=True,
            )
            self._documents[state.doc_id] = state
            self._active_doc_id = state.doc_id
            self._all_chunks = []
            for doc_state in self._documents.values():
                self._all_chunks.extend(doc_state.chunks)
            self._hydrate_index_from_persisted()
            return state

        _progress("Metin cikarma basladi (PDF/OCR/VLM)...")
        assets_dir = self.multimodal_assets_dir or (self.chroma_dir.parent / "multimodal_assets")
        ingest = ingest_any(
            file_path,
            ocr=self.ocr_config,
            display_name=display_name,
            vlm=self.vlm_config,
            table_config=self.table_structure_config,
            multimodal=MultimodalConfig(
                enabled=self.processing_mode == "multimodal",
                assets_dir=assets_dir,
                chunk_level=self.visual_chunk_level,
                region_source=self.visual_region_source,
                detector_backend=self.visual_detector_backend,
                detector_dir=self.visual_detector_dir,
                docai_project_id=self.docai_project_id,
                docai_location=self.docai_location,
                docai_processor_id=self.docai_layout_processor_id,
                docai_processor_version=self.docai_layout_processor_version,
                docai_timeout_seconds=self.docai_timeout_seconds,
                docling_python_bin=self.docling_python_bin,
                docling_layout_model=self.docling_layout_model,
                docling_artifacts_path=self.docling_artifacts_path,
                docling_device=self.docling_device,
            ),
        )
        _progress("Belge yapisi analiz ediliyor...")
        root = build_section_tree(ingest)
        _progress("Chunk'lar olusturuluyor...")
        chunks = section_tree_to_chunks(ingest, root)
        if self.processing_mode == "multimodal":
            chunks.extend(visual_chunks_from_ingest(ingest))
        chunks.extend(table_chunks_from_ingest(ingest))

        state = DocumentState(
            doc_id=ingest.doc_id,
            file_name=ingest.file_name,
            chunks=chunks,
            page_count=len(ingest.pages),
            warnings=ingest.warnings,
            build_fingerprint=fp,
        )
        self._documents[ingest.doc_id] = state
        self._active_doc_id = ingest.doc_id

        # Update full chunk list.
        self._all_chunks = []
        for doc_state in self._documents.values():
            self._all_chunks.extend(doc_state.chunks)

        # If we couldn't extract any chunks (empty PDF, OCR missing, etc.), keep the
        # document state but avoid building an empty index (some backends reject empty upserts).
        if not self._all_chunks:
            state.warnings = list(state.warnings) + [
                "Bu belgeden metin/bolum cikarilamadi (bos veya OCR gerektiriyor olabilir)."
            ]
            self._index = None
            state.collection_name = "chunks"
            self._doc_cache().save(
                doc_id=state.doc_id,
                file_name=state.file_name,
                page_count=state.page_count,
                warnings=state.warnings,
                build_fingerprint=state.build_fingerprint,
                collection_name=state.collection_name,
                chunks=state.chunks,
            )
            _progress("Belgeden indekslenebilir metin cikmadi.")
            lg = self._get_logger()
            if lg:
                lg.log(
                    session_id=self._session_id,
                    event="doc_added",
                    payload={
                        "doc_name": state.file_name,
                        "doc_id": state.doc_id,
                        "page_count": state.page_count,
                        "chunk_count": len(state.chunks),
                        "warnings": state.warnings,
                    },
                )
            return state

        # Incremental indexing: if an index already exists AND the new doc has
        # chunks, add only the new doc's chunks (avoid re-embedding all previous).
        _t0 = time.perf_counter()
        if self._index is not None and chunks:
            _progress("Mevcut indekse yeni chunk'lar ekleniyor...")
            self._index.add_chunks(chunks)
        else:
            _progress("Vektor ve BM25 indeksleri olusturuluyor...")
            self._index = LocalIndex.build(
                chunks=self._all_chunks,
                chroma_dir=self.chroma_dir,
                embedding_model=self.embedding_model,
                embedding_device=self.embedding_device,
            )
        state.collection_name = self._index.store.collection_name if self._index else "chunks"
        _index_ms = (time.perf_counter() - _t0) * 1000
        _progress("Belge isleme tamamlandi.")

        self._doc_cache().save(
            doc_id=state.doc_id,
            file_name=state.file_name,
            page_count=state.page_count,
            warnings=state.warnings,
            build_fingerprint=state.build_fingerprint,
            collection_name=state.collection_name,
            chunks=state.chunks,
        )

        lg = self._get_logger()
        if lg:
            lg.log(
                session_id=self._session_id,
                event="doc_added",
                payload={
                    "doc_name": state.file_name,
                    "doc_id": state.doc_id,
                    "page_count": state.page_count,
                    "chunk_count": len(state.chunks),
                    "index_time_ms": round(_index_ms, 1),
                    "incremental": self._index is not None and len(self._documents) > 1,
                    "warnings": state.warnings,
                },
            )

        return state

    def list_documents(self) -> List[str]:
        """User-facing filenames currently loaded in this session."""
        return [st.file_name for st in self._documents.values()]

    @staticmethod
    def _normalize_doc_ref(text: str) -> str:
        """
        Normalize user text / filenames for fuzzy matching.
        Keeps only letters+digits, collapses separators, strips extension-like suffix.
        """
        s = (text or "").strip().lower()
        # Remove a common extension mention (user might omit it anyway)
        s = re.sub(r"\.(pdf|png|jpg|jpeg)\b", "", s)
        # Normalize separators to spaces
        s = re.sub(r"[_\-\.\(\)\[\]\{\}]+", " ", s)
        # Remove everything else except letters/digits (keep Turkish letters)
        s = re.sub(r"[^0-9a-zçğıöşüâîû\s]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @classmethod
    def _doc_match_score(cls, query: str, file_name: str) -> float:
        """
        Return a 0..1 confidence score that `query` refers to `file_name`.
        Conservative: exact equality/stem equality > substring > token overlap.
        """
        qn = cls._normalize_doc_ref(query)
        fn = cls._normalize_doc_ref(file_name)
        if not qn or not fn:
            return 0.0

        stem = cls._normalize_doc_ref(Path(file_name).stem)

        # Strong signals
        if qn == fn or qn == stem:
            return 1.0

        # Medium: query contains full filename/stem (common when user writes a partial phrase)
        if stem and len(stem) >= 3 and stem in qn:
            # Longer stems yield higher confidence
            return min(0.95, 0.60 + 0.35 * min(1.0, len(stem) / 18.0))
        if fn and len(fn) >= 6 and fn in qn:
            return 0.85

        # Weak: token overlap (require at least one meaningful token)
        q_tokens = {t for t in qn.split() if len(t) >= 3}
        f_tokens = {t for t in stem.split() if len(t) >= 3} or {t for t in fn.split() if len(t) >= 3}
        if not q_tokens or not f_tokens:
            return 0.0
        inter = q_tokens & f_tokens
        if not inter:
            return 0.0
        # Guard against overly-generic overlaps (e.g. only "doc").
        # If we only match a single short token (<=3) with no digits, treat as no match.
        if len(inter) == 1:
            t = next(iter(inter))
            if len(t) <= 3 and not any(ch.isdigit() for ch in t):
                return 0.0
        # Prefer higher overlap and longer tokens
        overlap = len(inter) / max(1, len(f_tokens))
        longest = max((len(t) for t in inter), default=0)
        return min(0.80, 0.25 + 0.55 * overlap + 0.05 * min(10, longest) / 10.0)

    def set_active_document(self, file_name: str) -> bool:
        """
        Set active document by (case-insensitive) filename match.
        Also supports unique partial matches (e.g., "case_study" matches "Case_Study_20260205.pdf").
        Returns True if matched, False otherwise.
        """
        target = (file_name or "").strip().lower()
        if not target:
            return False

        # 1) Exact (case-insensitive) match
        for did, st in self._documents.items():
            if (st.file_name or "").lower() == target:
                self._active_doc_id = did
                return True

        # 2) Unique fuzzy match (avoid surprising switches when ambiguous)
        scored: list[tuple[float, str]] = []
        for did, st in self._documents.items():
            sc = self._doc_match_score(target, st.file_name or "")
            if sc > 0:
                scored.append((sc, did))
        scored.sort(reverse=True, key=lambda x: x[0])
        if scored and scored[0][0] >= 0.55:
            # If top is clearly better than second, accept it.
            if len(scored) == 1 or (scored[0][0] - scored[1][0]) >= 0.12:
                self._active_doc_id = scored[0][1]
                return True
        return False

    def _resolve_doc_id_hint(self, query: str) -> Optional[str]:
        """
        Document-agnostic routing for multi-doc sessions.

        Rules:
        - If only one document is loaded → use it.
        - If query mentions a known filename (exact or partial) → route to that doc.
        - Else → route to the last uploaded / active doc (if any).
        """
        if not self._documents:
            return None
        if len(self._documents) == 1:
            return next(iter(self._documents.keys()))

        # Prefer explicit mention in the user's message.
        scored: list[tuple[float, str]] = []
        for did, st in self._documents.items():
            sc = self._doc_match_score(query, st.file_name or "")
            if sc > 0:
                scored.append((sc, did))
        scored.sort(reverse=True, key=lambda x: x[0])
        if scored and scored[0][0] >= 0.55:
            # Avoid wrong routing when ambiguous; fall back to active doc.
            if len(scored) == 1 or (scored[0][0] - scored[1][0]) >= 0.12:
                return scored[0][1]
        return self._active_doc_id

    @property
    def has_documents(self) -> bool:
        return bool(self._documents)

    @property
    def has_index(self) -> bool:
        """True if at least one chunk is indexed and retrieval can run."""
        return self._index is not None and bool(self._all_chunks)

    @property
    def active_document_name(self) -> Optional[str]:
        """User-facing active document filename (if any)."""
        if not self._active_doc_id:
            return None
        st = self._documents.get(self._active_doc_id)
        return st.file_name if st else None

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def total_chunks(self) -> int:
        return len(self._all_chunks)

    def reconfigure_runtime(
        self,
        *,
        ocr_enabled: Optional[bool] = None,
        ocr_backend: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_device: Optional[str] = None,
        processing_mode: Optional[str] = None,
        multimodal_answer_mode: Optional[str] = None,
        visual_chunk_level: Optional[str] = None,
        visual_region_source: Optional[str] = None,
        visual_detector_backend: Optional[str] = None,
        table_structure_enabled: Optional[bool] = None,
        table_structure_backend: Optional[str] = None,
        llm_provider: Optional[str] = None,
        gemini_model: Optional[str] = None,
        openai_model: Optional[str] = None,
        local_llm_model: Optional[str] = None,
        vlm_mode: Optional[str] = None,
        vlm_provider: Optional[str] = None,
        vlm_max_pages: Optional[int] = None,
    ) -> Dict[str, bool]:
        """
        Update runtime knobs from UI.

        - Embedding model/device changes rebuild the index from loaded chunks.
        - VLM changes affect future document ingestion (new uploads).
        """
        ocr_changed = False
        ocr_after = self.ocr_config
        if ocr_enabled is not None and bool(ocr_enabled) != bool(getattr(ocr_after, "enabled", True)):
            ocr_after = replace(ocr_after, enabled=bool(ocr_enabled))
        backend_next = (ocr_backend or "").strip().lower()
        if backend_next in ("docai", "paddle_vl", "paddle", "tesseract_legacy") and backend_next != getattr(ocr_after, "backend", "tesseract_legacy"):
            ocr_after = replace(ocr_after, backend=backend_next)
        if ocr_after != self.ocr_config:
            self.ocr_config = ocr_after
            ocr_changed = True

        embedding_changed = False
        model_next = (embedding_model or "").strip()
        if model_next and model_next != self.embedding_model:
            self.embedding_model = model_next
            embedding_changed = True

        device_next = (embedding_device or "").strip().lower()
        if device_next and device_next != self.embedding_device:
            self.embedding_device = device_next
            embedding_changed = True

        processing_mode_changed = False
        mode_next = (processing_mode or "").strip().lower()
        if mode_next in ("classic", "multimodal") and mode_next != self.processing_mode:
            self.processing_mode = mode_next
            processing_mode_changed = True

        multimodal_answer_mode_changed = False
        answer_mode_next = (multimodal_answer_mode or "").strip().lower()
        if answer_mode_next in ("off", "auto", "on") and answer_mode_next != self.multimodal_answer_mode:
            self.multimodal_answer_mode = answer_mode_next
            multimodal_answer_mode_changed = True

        visual_chunk_level_changed = False
        visual_level_next = (visual_chunk_level or "").strip().lower()
        if visual_level_next in ("page", "region") and visual_level_next != self.visual_chunk_level:
            self.visual_chunk_level = visual_level_next
            visual_chunk_level_changed = True

        visual_region_source_changed = False
        visual_region_source_next = (visual_region_source or "").strip().lower()
        if visual_region_source_next in ("heuristic", "detector") and visual_region_source_next != self.visual_region_source:
            self.visual_region_source = visual_region_source_next
            visual_region_source_changed = True

        visual_detector_backend_changed = False
        visual_detector_backend_next = (visual_detector_backend or "").strip().lower()
        if visual_detector_backend_next in ("none", "sidecar", "docai", "docling") and visual_detector_backend_next != self.visual_detector_backend:
            self.visual_detector_backend = visual_detector_backend_next
            visual_detector_backend_changed = True

        table_structure_changed = False
        if self.table_structure_config is None:
            self.table_structure_config = TableStructureConfig(
                enabled=False,
                backend="auto",
                gemini_api_key=self.gemini_api_key,
                gemini_model=self.gemini_model,
                docai_project_id=self.docai_project_id,
                docai_location=self.docai_location,
                docai_timeout_seconds=self.docai_timeout_seconds,
            )
        table_after = self.table_structure_config
        if table_structure_enabled is not None and bool(table_structure_enabled) != bool(getattr(table_after, "enabled", False)):
            table_after = replace(table_after, enabled=bool(table_structure_enabled))
        table_backend_next = (table_structure_backend or "").strip().lower()
        if table_backend_next in ("off", "auto", "docai", "gemini", "heuristic") and table_backend_next != getattr(table_after, "backend", "auto"):
            table_after = replace(table_after, backend=table_backend_next)
        if table_after != self.table_structure_config:
            self.table_structure_config = table_after
            table_structure_changed = True

        llm_changed = False
        llm_provider_next = (llm_provider or "").strip().lower()
        if llm_provider_next in ("gemini", "openai", "local", "none") and llm_provider_next != self.llm_provider:
            self.llm_provider = llm_provider_next
            llm_changed = True
        gemini_model_next = (gemini_model or "").strip()
        if gemini_model_next and gemini_model_next != self.gemini_model:
            self.gemini_model = gemini_model_next
            llm_changed = True
        openai_model_next = (openai_model or "").strip()
        if openai_model_next and openai_model_next != self.openai_model:
            self.openai_model = openai_model_next
            llm_changed = True
        local_llm_model_next = (local_llm_model or "").strip()
        if local_llm_model_next:
            if self.ollama_config is None:
                self.ollama_config = OllamaConfig(llm_model=local_llm_model_next)
                llm_changed = True
            elif local_llm_model_next != self.ollama_config.llm_model:
                self.ollama_config = replace(self.ollama_config, llm_model=local_llm_model_next)
                llm_changed = True

        if self.vlm_config is None:
            self.vlm_config = VLMConfig(
                api_key=self.gemini_api_key,
                model=self.gemini_model,
                mode="auto",
                max_pages=25,
                provider="gemini",
                ollama_base_url=(self.ollama_config.base_url if self.ollama_config else "http://localhost:11434"),
                ollama_vlm_model=(self.ollama_config.vlm_model if self.ollama_config else "llava:7b"),
                ollama_timeout=(self.ollama_config.timeout if self.ollama_config else 120),
            )

        vlm_before = self.vlm_config
        vlm_after = vlm_before
        if vlm_mode in ("off", "auto", "force"):
            vlm_after = replace(vlm_after, mode=vlm_mode)
        if vlm_provider in ("gemini", "local"):
            vlm_after = replace(vlm_after, provider=vlm_provider)
        if vlm_max_pages is not None:
            vlm_after = replace(vlm_after, max_pages=max(0, min(200, int(vlm_max_pages))))

        vlm_changed = vlm_after != vlm_before
        if vlm_changed:
            self.vlm_config = vlm_after

        index_rebuilt = False
        if embedding_changed and self._all_chunks:
            self._index = LocalIndex.build(
                chunks=self._all_chunks,
                chroma_dir=self.chroma_dir,
                embedding_model=self.embedding_model,
                embedding_device=self.embedding_device,
            )
            index_rebuilt = True

        return {
            "ocr_changed": ocr_changed,
            "embedding_changed": embedding_changed,
            "processing_mode_changed": processing_mode_changed,
            "multimodal_answer_mode_changed": multimodal_answer_mode_changed,
            "visual_chunk_level_changed": visual_chunk_level_changed,
            "visual_region_source_changed": visual_region_source_changed,
            "visual_detector_backend_changed": visual_detector_backend_changed,
            "table_structure_changed": table_structure_changed,
            "llm_changed": llm_changed,
            "vlm_changed": vlm_changed,
            "index_rebuilt": index_rebuilt,
        }

    def ask(self, query: str) -> GenerationResult:
        """
        Full pipeline: retrieve → generate answer.
        Raises ValueError if no documents have been indexed.
        """
        if self._index is None:
            # No chunks indexed (empty/scanned doc without OCR, etc.). Return safe "not found".
            empty = RetrievalResult(intent=classify_query(query), evidences=[], section_complete=False, coverage=None)
            if self.llm_provider == "none":
                result = generate_extractive_answer(retrieval=empty, query=query)
            elif self.llm_provider == "openai":
                result = generate_answer_openai(
                    retrieval=empty,
                    query=query,
                    openai_api_key=self.openai_api_key,
                    openai_model=self.openai_model,
                )
            elif self.llm_provider == "local" and self.ollama_config:
                result = generate_answer_local(
                    retrieval=empty,
                    query=query,
                    ollama_cfg=self.ollama_config,
                )
            else:
                result = generate_answer(
                    retrieval=empty,
                    query=query,
                    gemini_api_key=self.gemini_api_key,
                    gemini_model=self.gemini_model,
                    multimodal_answer_mode=self.multimodal_answer_mode,
                )
            result = replace(result, evidence_summary=self._summarize_evidence(empty))
            lg = self._get_logger()
            if lg:
                lg.log(
                    session_id=self._session_id,
                    event="qa",
                    payload={
                        "query": query,
                        "intent": empty.intent,
                        "active_doc_name": self.active_document_name,
                        "active_doc_id": self._active_doc_id,
                        "documents": self.list_documents(),
                        "doc_count": self.document_count,
                        "evidence_count": 0,
                        "section_complete": False,
                        "coverage_expected": None,
                        "coverage_actual": None,
                        "coverage_ok": None,
                        "citations_found": result.citations_found,
                        "multimodal_answer_mode": self.multimodal_answer_mode,
                        "answer": result.answer,
                        **(
                            {"context_preview": result.context_preview}
                            if (lg.include_context_preview and result.context_preview)
                            else {}
                        ),
                    },
                )
            return result

        _t_ret = time.perf_counter()
        doc_hint = self._resolve_doc_id_hint(query)
        ret = retrieve(self._index, query, doc_id=doc_hint)
        _retrieval_ms = (time.perf_counter() - _t_ret) * 1000

        _t_gen = time.perf_counter()
        if self.llm_provider == "none":
            result = generate_extractive_answer(retrieval=ret, query=query)
        elif self.llm_provider == "openai":
            result = generate_answer_openai(
                retrieval=ret,
                query=query,
                openai_api_key=self.openai_api_key,
                openai_model=self.openai_model,
            )
        elif self.llm_provider == "local" and self.ollama_config:
            result = generate_answer_local(
                retrieval=ret,
                query=query,
                ollama_cfg=self.ollama_config,
            )
        else:
            result = generate_answer(
                retrieval=ret,
                query=query,
                gemini_api_key=self.gemini_api_key,
                gemini_model=self.gemini_model,
                multimodal_answer_mode=self.multimodal_answer_mode,
            )
        result = replace(result, evidence_summary=self._summarize_evidence(ret))
        _gen_ms = (time.perf_counter() - _t_gen) * 1000
        lg = self._get_logger()
        if lg:
            lg.log(
                session_id=self._session_id,
                event="qa",
                payload={
                    "query": query,
                    "intent": ret.intent,
                    "active_doc_name": self.active_document_name,
                    "active_doc_id": doc_hint,
                    "documents": self.list_documents(),
                    "doc_count": self.document_count,
                    "evidence_count": len(ret.evidences),
                    "section_complete": bool(ret.section_complete),
                    "coverage_expected": ret.coverage.expected_items if ret.coverage else None,
                    "coverage_actual": result.coverage_actual,
                    "coverage_ok": result.coverage_ok,
                    "citations_found": result.citations_found,
                    "multimodal_answer_mode": self.multimodal_answer_mode,
                    "answer": result.answer,
                    "retrieval_ms": round(_retrieval_ms, 1),
                    "generation_ms": round(_gen_ms, 1),
                    **(
                        {"context_preview": result.context_preview}
                        if (lg.include_context_preview and result.context_preview)
                        else {}
                    ),
                },
            )

        return result

    def ask_stream(
        self,
        query: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> GenerationResult:
        """
        Streaming answer path for UI token rendering.

        Keeps retrieval/index architecture unchanged and coexists with ask().
        """
        if self._index is None:
            empty = RetrievalResult(intent=classify_query(query), evidences=[], section_complete=False, coverage=None)
            if self.llm_provider == "none":
                result = generate_extractive_answer(retrieval=empty, query=query)
                if on_token and result.answer:
                    on_token(result.answer)
            elif self.llm_provider == "openai":
                result = generate_answer_openai_stream(
                    retrieval=empty,
                    query=query,
                    openai_api_key=self.openai_api_key,
                    openai_model=self.openai_model,
                    on_token=on_token,
                )
            elif self.llm_provider == "local" and self.ollama_config:
                result = generate_answer_local_stream(
                    retrieval=empty,
                    query=query,
                    ollama_cfg=self.ollama_config,
                    on_token=on_token,
                )
            else:
                result = generate_answer_stream(
                    retrieval=empty,
                    query=query,
                    gemini_api_key=self.gemini_api_key,
                    gemini_model=self.gemini_model,
                    multimodal_answer_mode=self.multimodal_answer_mode,
                    on_token=on_token,
                )
            result = replace(result, evidence_summary=self._summarize_evidence(empty))
            return result

        _t_ret = time.perf_counter()
        doc_hint = self._resolve_doc_id_hint(query)
        ret = retrieve(self._index, query, doc_id=doc_hint)
        _retrieval_ms = (time.perf_counter() - _t_ret) * 1000

        _t_gen = time.perf_counter()
        if self.llm_provider == "none":
            result = generate_extractive_answer(retrieval=ret, query=query)
            if on_token and result.answer:
                on_token(result.answer)
        elif self.llm_provider == "openai":
            result = generate_answer_openai_stream(
                retrieval=ret,
                query=query,
                openai_api_key=self.openai_api_key,
                openai_model=self.openai_model,
                on_token=on_token,
            )
        elif self.llm_provider == "local" and self.ollama_config:
            result = generate_answer_local_stream(
                retrieval=ret,
                query=query,
                ollama_cfg=self.ollama_config,
                on_token=on_token,
            )
        else:
            result = generate_answer_stream(
                retrieval=ret,
                query=query,
                gemini_api_key=self.gemini_api_key,
                gemini_model=self.gemini_model,
                multimodal_answer_mode=self.multimodal_answer_mode,
                on_token=on_token,
            )
        result = replace(result, evidence_summary=self._summarize_evidence(ret))
        _gen_ms = (time.perf_counter() - _t_gen) * 1000

        lg = self._get_logger()
        if lg:
            lg.log(
                session_id=self._session_id,
                event="qa_stream",
                payload={
                    "query": query,
                    "intent": ret.intent,
                    "active_doc_name": self.active_document_name,
                    "active_doc_id": doc_hint,
                    "documents": self.list_documents(),
                    "doc_count": self.document_count,
                    "evidence_count": len(ret.evidences),
                    "section_complete": bool(ret.section_complete),
                    "coverage_expected": ret.coverage.expected_items if ret.coverage else None,
                    "coverage_actual": result.coverage_actual,
                    "coverage_ok": result.coverage_ok,
                    "citations_found": result.citations_found,
                    "multimodal_answer_mode": self.multimodal_answer_mode,
                    "answer_len": len(result.answer or ""),
                    "retrieval_ms": round(_retrieval_ms, 1),
                    "generation_ms": round(_gen_ms, 1),
                    **(
                        {"context_preview": result.context_preview}
                        if (lg.include_context_preview and result.context_preview)
                        else {}
                    ),
                },
            )

        return result

    def chat(self, query: str, chat_style: str = "neutral") -> str:
        """
        Chat-only mode (no retrieval).
        """
        if self.llm_provider == "none":
            return (
                "Sohbet modu devre disi (LLM provider: none). "
                "LLM secimi yapip tekrar deneyebilirsin."
            )
        if self.llm_provider == "openai":
            return generate_chat_answer_openai(
                query=query,
                openai_api_key=self.openai_api_key,
                openai_model=self.openai_model,
                chat_style=chat_style,
            )
        if self.llm_provider == "local" and self.ollama_config:
            return generate_chat_answer_local(
                query=query,
                ollama_cfg=self.ollama_config,
                chat_style=chat_style,
            )
        return generate_chat_answer(
            query=query,
            gemini_api_key=self.gemini_api_key,
            gemini_model=self.gemini_model,
            chat_style=chat_style,
        )

    def get_retrieval(self, query: str) -> RetrievalResult:
        """
        Retrieve evidence without generation (useful for debugging).
        """
        if self._index is None:
            return RetrievalResult(intent=classify_query(query), evidences=[], section_complete=False, coverage=None)
        doc_hint = self._resolve_doc_id_hint(query)
        return retrieve(self._index, query, doc_id=doc_hint)
