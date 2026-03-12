from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from src.config import load_settings
    from src.core.ingestion import OCRConfig, ingest_any
    from src.core.structure import build_section_tree, flatten_sections, section_tree_to_chunks
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import load_settings  # type: ignore  # noqa: E402
    from src.core.ingestion import OCRConfig, ingest_any  # type: ignore  # noqa: E402
    from src.core.structure import (  # type: ignore  # noqa: E402
        build_section_tree,
        flatten_sections,
        section_tree_to_chunks,
    )


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
    ocr = OCRConfig(
        enabled=getattr(settings, "ocr_enabled", True),
        lang="tur+eng",
        tesseract_cmd=settings.tesseract_cmd,
        tessdata_prefix=settings.tessdata_prefix,
        tesseract_config=getattr(settings, "tesseract_config", None),
    )
    ingest = ingest_any(Path(args.path), ocr=ocr)

    root = build_section_tree(ingest)
    sections = flatten_sections(root)
    chunks = section_tree_to_chunks(ingest, root)

    print(f"sections={len(sections)} chunks={len(chunks)}")
    print()
    for s in sections:
        if s.section_id == "root":
            continue
        indent = "  " * max(0, s.level - 1)
        print(f"{indent}- [{s.section_id}] p{s.page_start}-p{s.page_end} :: {s.title}")

    # Show chunk stats
    parents = [c for c in chunks if c.kind == "parent"]
    children = [c for c in chunks if c.kind == "child"]
    print()
    print(f"parent_chunks={len(parents)} child_chunks={len(children)}")
    if parents:
        biggest = max(parents, key=lambda c: len(c.text))
        print(f"largest_parent_chars={len(biggest.text)} section={biggest.section_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

