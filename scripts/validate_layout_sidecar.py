from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # type: ignore[assignment]


def _load_payload(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"invalid json: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("sidecar root must be an object")
    return payload


def _validate_bbox(row: dict, *, page_key: str, idx: int, errors: list[str]) -> None:
    bbox = row.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        errors.append(f"page {page_key} region {idx}: bbox must be a 4-item list")
        return
    try:
        left, top, right, bottom = [int(v) for v in bbox]
    except Exception:
        errors.append(f"page {page_key} region {idx}: bbox values must be integers")
        return
    if right <= left or bottom <= top:
        errors.append(f"page {page_key} region {idx}: bbox must satisfy right>left and bottom>top")


def _validate_payload(payload: dict, *, expected_pages: int | None = None) -> list[str]:
    errors: list[str] = []
    pages = payload.get("pages")
    if not isinstance(pages, dict):
        errors.append("missing `pages` object")
        return errors

    for page_key, rows in pages.items():
        try:
            page_no = int(page_key)
        except Exception:
            errors.append(f"page key must be an integer string: {page_key!r}")
            continue
        if page_no <= 0:
            errors.append(f"page key must be >= 1: {page_key!r}")
        if expected_pages is not None and page_no > expected_pages:
            errors.append(f"page {page_no} exceeds document page count {expected_pages}")
        if not isinstance(rows, list):
            errors.append(f"page {page_key}: value must be a list")
            continue
        for idx, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                errors.append(f"page {page_key} region {idx}: row must be an object")
                continue
            _validate_bbox(row, page_key=page_key, idx=idx, errors=errors)
            conf = row.get("confidence", 0.0)
            try:
                conf_float = float(conf)
            except Exception:
                errors.append(f"page {page_key} region {idx}: confidence must be numeric")
                continue
            if conf_float < 0.0 or conf_float > 1.0:
                errors.append(f"page {page_key} region {idx}: confidence must be between 0 and 1")
    return errors


def _document_page_count(path: Path) -> int | None:
    if fitz is None:
        return None
    suffix = path.suffix.lower()
    if suffix not in {".pdf"}:
        return 1
    try:
        pdf = fitz.open(str(path))
    except Exception:
        return None
    try:
        return int(pdf.page_count)
    finally:
        pdf.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a layout detector sidecar JSON file.")
    ap.add_argument("sidecar", type=str, help="Path to sidecar JSON")
    ap.add_argument("--document", type=str, default="", help="Optional source document path for page-count validation")
    args = ap.parse_args()

    sidecar_path = Path(args.sidecar)
    if not sidecar_path.exists():
        raise SystemExit(f"sidecar not found: {sidecar_path}")

    expected_pages = None
    if args.document:
        doc_path = Path(args.document)
        if not doc_path.exists():
            raise SystemExit(f"document not found: {doc_path}")
        expected_pages = _document_page_count(doc_path)

    payload = _load_payload(sidecar_path)
    errors = _validate_payload(payload, expected_pages=expected_pages)
    if errors:
        print("INVALID")
        for err in errors:
            print(f"- {err}")
        return 1

    page_count = len(payload.get("pages", {}))
    print(f"VALID pages={page_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
