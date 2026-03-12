"""
End-to-end test: ingest → index → retrieve → generate answer via Gemini.
Usage:
    python scripts/test_generation.py Case_Study_20260205.pdf "fonksiyonel gereksinimler nelerdir"
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from src.config import load_settings
    from src.core.ingestion import OCRConfig, ingest_any
    from src.core.indexing import LocalIndex
    from src.core.structure import build_section_tree, section_tree_to_chunks
    from src.core.retrieval import retrieve
    from src.core.generation import generate_answer
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import load_settings  # type: ignore
    from src.core.ingestion import OCRConfig, ingest_any  # type: ignore
    from src.core.indexing import LocalIndex  # type: ignore
    from src.core.structure import build_section_tree, section_tree_to_chunks  # type: ignore
    from src.core.retrieval import retrieve  # type: ignore
    from src.core.generation import generate_answer  # type: ignore


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="PDF / PNG / JPG path")
    ap.add_argument("query", type=str, help="Natural language question")
    args = ap.parse_args()

    settings = load_settings()

    if settings.llm_provider != "gemini" or not settings.gemini_api_key:
        print("ERROR: Set LLM_PROVIDER=gemini and GEMINI_API_KEY in .env")
        return 1

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Ingesting document...")
    ocr = OCRConfig(
        enabled=getattr(settings, "ocr_enabled", True),
        lang="tur+eng",
        tesseract_cmd=settings.tesseract_cmd,
        tessdata_prefix=settings.tessdata_prefix,
        tesseract_config=getattr(settings, "tesseract_config", None),
    )
    ingest = ingest_any(Path(args.path), ocr=ocr)

    print("[2/4] Building index...")
    root = build_section_tree(ingest)
    chunks = section_tree_to_chunks(ingest, root)
    idx = LocalIndex.build(
        chunks=chunks,
        chroma_dir=settings.chroma_dir,
        embedding_model=settings.embedding_model,
    )

    print("[3/4] Retrieving...")
    ret = retrieve(idx, args.query)
    print(f"  intent={ret.intent}  evidences={len(ret.evidences)}  section_complete={ret.section_complete}")
    if ret.coverage:
        print(f"  coverage: expected_items={ret.coverage.expected_items}")

    print("[4/4] Generating answer via Gemini...")
    result = generate_answer(
        retrieval=ret,
        query=args.query,
        gemini_api_key=settings.gemini_api_key,
        gemini_model=settings.gemini_model,
    )

    print(f"\n{'='*70}")
    print(f"INTENT: {result.intent}")
    print(f"CITATIONS: {result.citations_found}")
    if result.coverage_expected is not None:
        print(f"COVERAGE: expected={result.coverage_expected} actual={result.coverage_actual} ok={result.coverage_ok}")
    print(f"{'='*70}")
    print(result.answer)
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
