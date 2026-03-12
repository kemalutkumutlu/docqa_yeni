from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from src.config import load_settings
    from src.core.ingestion import OCRConfig, ingest_any
except ModuleNotFoundError:
    # Allows running as: `python scripts/extract_text.py ...` without installing as a package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import load_settings  # type: ignore  # noqa: E402
    from src.core.ingestion import OCRConfig, ingest_any  # type: ignore  # noqa: E402


def main() -> int:
    # Windows consoles can default to non-UTF8 encodings; force UTF-8 for Turkish chars.
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
    res = ingest_any(Path(args.path), ocr=ocr)

    print(f"doc_id={res.doc_id} file={res.file_name} pages={len(res.pages)}")
    if res.warnings:
        print("\nWARNINGS:")
        for w in res.warnings:
            print(f"- {w}")
        print()

    for p in res.pages:
        print(f"\n=== Page {p.page_number} ({p.source}) ===\n")
        print(p.text[:4000])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

