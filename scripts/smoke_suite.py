from __future__ import annotations

import sys
from pathlib import Path


def _setup_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]


def main() -> int:
    _setup_utf8()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from src.config import load_settings
    from src.core.ingestion import OCRConfig
    from src.core.pipeline import RAGPipeline
    from src.core.vlm_extract import VLMConfig

    settings = load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    pipe = RAGPipeline(
        embedding_model=settings.embedding_model,
        chroma_dir=settings.chroma_dir,
        gemini_api_key=settings.gemini_api_key,
        gemini_model=settings.gemini_model,
        ocr_config=OCRConfig(
            enabled=getattr(settings, "ocr_enabled", True),
            lang="tur+eng",
            tesseract_cmd=settings.tesseract_cmd,
            tessdata_prefix=settings.tessdata_prefix,
            tesseract_config=getattr(settings, "tesseract_config", None),
        ),
        vlm_config=VLMConfig(api_key=settings.gemini_api_key, model=settings.gemini_model, mode="off"),
    )

    # Provide PDF paths via CLI:
    #   python scripts/smoke_suite.py path1.pdf path2.pdf
    pdfs = [Path(p) for p in sys.argv[1:] if p.lower().endswith(".pdf")]
    if not pdfs:
        print("Usage: python scripts/smoke_suite.py <doc1.pdf> <doc2.pdf> ...")
        return 2

    for p in pdfs:
        st = pipe.add_document(p, display_name=p.name)
        print(f"[OK] indexed {st.file_name}: pages={st.page_count} chunks={len(st.chunks)}")

    # Cross-document contamination smoke check:
    # - ensure each retrieval returns only allowed doc_ids from the active index
    queries = [
        "adres bilgisi",
        "iş deneyimi",
        "fonksiyonel gereksinimler nelerdir",
        "teslimatlar nelerdir",
    ]
    idx = pipe._index  # noqa: SLF001 (smoke suite)
    assert idx is not None

    for q in queries:
        ret = pipe.get_retrieval(q)
        bad = []
        ev_ids = [ev.chunk_id for ev in ret.evidences]
        if ev_ids:
            got = idx.store.get(ev_ids)
            metas = got.get("metadatas", []) or []
            for cid, meta in zip(ev_ids, metas):
                did = (meta or {}).get("doc_id", "")
                fname = (meta or {}).get("file_name", "")
                if idx.allowed_doc_ids is not None and did not in idx.allowed_doc_ids:
                    bad.append((cid, did, fname))
        if bad:
            print(f"[FAIL] contamination for query={q!r}: {bad[:3]}")
            return 1
        print(f"[OK] retrieval query={q!r}: intent={ret.intent} evidences={len(ret.evidences)}")

    print("SMOKE SUITE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

