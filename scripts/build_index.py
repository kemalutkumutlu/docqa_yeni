from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from src.config import load_settings
    from src.core.ingestion import OCRConfig, ingest_any
    from src.core.indexing import LocalIndex
    from src.core.structure import build_section_tree, section_tree_to_chunks
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import load_settings  # type: ignore  # noqa: E402
    from src.core.ingestion import OCRConfig, ingest_any  # type: ignore  # noqa: E402
    from src.core.indexing import LocalIndex  # type: ignore  # noqa: E402
    from src.core.structure import build_section_tree, section_tree_to_chunks  # type: ignore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="PDF / PNG / JPG path")
    ap.add_argument("--collection", type=str, default="chunks", help="Chroma collection name")
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
        collection_name=args.collection,
    )

    print(
        f"Indexed doc_id={ingest.doc_id} file={ingest.file_name} "
        f"chunks={len(chunks)} bm25_docs={len(idx.bm25.ids)} chroma_dir={settings.chroma_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

