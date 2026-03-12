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
    # Windows consoles can default to non-UTF8 encodings; force UTF-8 for Turkish chars.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="PDF / PNG / JPG path")
    ap.add_argument("query", type=str, help="Search query")
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

    res = idx.hybrid_search(args.query, dense_k=10, sparse_k=10, final_k=8)
    print(f"query={args.query!r}")
    print("top_ids:")
    for i, cid in enumerate(res.ids, start=1):
        print(f"{i:02d}. {cid} score={res.scores.get(cid):.4f}")

    got = idx.store.get(res.ids)
    got_ids = got.get("ids", [])
    docs = got.get("documents", [])
    metas = got.get("metadatas", [])
    print("\npreview:")
    for i, (cid, doc, meta) in enumerate(zip(got_ids, docs, metas), start=1):
        hp = meta.get("heading_path")
        pr = f"p{meta.get('page_start')}-p{meta.get('page_end')}"
        kind = meta.get("kind")
        print(f"\n--- {i:02d}. {cid} [{kind}] {pr} ---")
        print(f"heading_path: {hp}")
        print((doc or "")[:700])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

