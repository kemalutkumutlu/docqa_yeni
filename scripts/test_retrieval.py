from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from src.config import load_settings
    from src.core.ingestion import OCRConfig, ingest_any
    from src.core.indexing import LocalIndex
    from src.core.structure import build_section_tree, section_tree_to_chunks
    from src.core.retrieval import retrieve, classify_query
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import load_settings  # type: ignore
    from src.core.ingestion import OCRConfig, ingest_any  # type: ignore
    from src.core.indexing import LocalIndex  # type: ignore
    from src.core.structure import build_section_tree, section_tree_to_chunks  # type: ignore
    from src.core.retrieval import retrieve, classify_query  # type: ignore


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="PDF / PNG / JPG path")
    args = ap.parse_args()

    settings = load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    ocr = OCRConfig(
        enabled=getattr(settings, "ocr_enabled", True),
        lang="tur+eng",
        tesseract_cmd=settings.tesseract_cmd,
        tessdata_prefix=settings.tessdata_prefix,
        tesseract_config=getattr(settings, "tesseract_config", None),
    )
    ingest = ingest_any(Path(args.path), ocr=ocr)
    root = build_section_tree(ingest)
    chunks = section_tree_to_chunks(ingest, root)
    idx = LocalIndex.build(
        chunks=chunks,
        chroma_dir=settings.chroma_dir,
        embedding_model=settings.embedding_model,
    )

    queries = [
        "fonksiyonel gereksinimler nelerdir",
        "teslimatlar nelerdir",
        "teslim süresi nedir",
        "projenin amacı nedir",
    ]

    for q in queries:
        intent = classify_query(q)
        res = retrieve(idx, q)
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print(f"intent={res.intent} section_complete={res.section_complete} evidences={len(res.evidences)}")
        if res.coverage:
            print(f"coverage: expected_items={res.coverage.expected_items} section={res.coverage.section_id} heading={res.coverage.heading_path}")
        for ev in res.evidences[:3]:
            print(f"  [{ev.kind}] {ev.section_id} p{ev.page_start}-p{ev.page_end} :: {ev.heading_path}")
            print(f"    {ev.text[:200]}...")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
