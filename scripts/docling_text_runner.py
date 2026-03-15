"""Docling-based PDF / image text extractor.

Outputs per-page text (with table markdown) as JSON:
  {"pages": {"1": "...", "2": "..."}, "warnings": [...]}

Usage:
  python docling_text_runner.py --source doc.pdf --output pages.json

Environment variables (same as docling_layout_runner):
  DOCLING_LAYOUT_MODEL   default: docling-layout-heron-101
  DOCLING_ARTIFACTS_PATH default: ""
  DOCLING_DEVICE         default: auto
  DOCLING_TABLE_STRUCTURE  "1" to enable table extraction (default: 1)
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


# ── Shared helpers (mirrors docling_layout_runner) ───────────────────────────

def _build_model_spec(model_name: str) -> Any:
    cleaned = (model_name or "").strip()
    if not cleaned:
        return None
    normalized = cleaned.replace("-", "_").strip().lower()
    repo_id = cleaned
    if "/" not in repo_id:
        repo_id = f"docling-project/{repo_id}"
    spec_dict = {
        "name": Path(cleaned).name.replace("-", "_"),
        "repo_id": repo_id,
        "revision": "main",
        "model_path": "",
    }
    try:
        from docling.datamodel.layout_model_specs import (
            DOCLING_LAYOUT_EGRET_LARGE,
            DOCLING_LAYOUT_EGRET_MEDIUM,
            DOCLING_LAYOUT_EGRET_XLARGE,
            DOCLING_LAYOUT_HERON,
            DOCLING_LAYOUT_HERON_101,
            DOCLING_LAYOUT_V2,
        )
        from docling.datamodel.pipeline_options import LayoutModelConfig

        preset_map = {
            "docling_layout_v2": DOCLING_LAYOUT_V2,
            "docling_layout_heron": DOCLING_LAYOUT_HERON,
            "docling_layout_heron_101": DOCLING_LAYOUT_HERON_101,
            "docling_layout_egret_medium": DOCLING_LAYOUT_EGRET_MEDIUM,
            "docling_layout_egret_large": DOCLING_LAYOUT_EGRET_LARGE,
            "docling_layout_egret_xlarge": DOCLING_LAYOUT_EGRET_XLARGE,
        }
        if normalized in preset_map:
            return preset_map[normalized]
        return LayoutModelConfig(**spec_dict)
    except Exception:
        return spec_dict


def _resolve_device(device_name: str) -> Any:
    cleaned = (device_name or "auto").strip().lower()
    if cleaned == "auto":
        try:
            import torch
            cleaned = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            cleaned = "cpu"
    try:
        from docling.datamodel.accelerator_options import AcceleratorDevice
        mapping = {
            "cpu": AcceleratorDevice.CPU,
            "cuda": AcceleratorDevice.CUDA,
            "mps": AcceleratorDevice.MPS,
            "xpu": AcceleratorDevice.XPU,
        }
        return mapping.get(cleaned, AcceleratorDevice.CPU)
    except Exception:
        return cleaned


def _make_converter(*, source_path: Path, model_name: str, artifacts_path: str, device_name: str, do_table_structure: bool):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter
    try:
        from docling.document_converter import ImageFormatOption as DoclingImageFormatOption
    except Exception:
        from docling.document_converter import PdfFormatOption as DoclingImageFormatOption  # type: ignore[assignment]
    from docling.document_converter import PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    for attr_name, attr_value in (
        ("do_ocr", False),
        ("do_table_structure", bool(do_table_structure)),
        ("generate_page_images", False),
        ("generate_picture_images", False),
    ):
        if hasattr(pipeline_options, attr_name):
            setattr(pipeline_options, attr_name, attr_value)

    if artifacts_path:
        setattr(pipeline_options, "artifacts_path", artifacts_path)

    try:
        from docling.datamodel.accelerator_options import AcceleratorOptions
        pipeline_options.accelerator_options = AcceleratorOptions(device=_resolve_device(device_name))
    except Exception:
        pass

    model_spec = _build_model_spec(model_name)
    if model_spec is not None:
        try:
            if getattr(pipeline_options, "layout_options", None) is not None:
                pipeline_options.layout_options.model_spec = model_spec
        except Exception:
            pass

    converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: DoclingImageFormatOption(pipeline_options=pipeline_options),
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    return converter.convert(source=source_path)


# ── Text extraction ──────────────────────────────────────────────────────────

def _text_from_item(item: Any) -> str:
    for attr_name in ("text", "orig", "content", "caption_text", "name"):
        value = getattr(item, attr_name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _table_to_markdown(item: Any) -> str:
    """Convert a Docling table item to markdown.  Tries export_to_markdown() first,
    then falls back to assembling from the cell grid."""
    try:
        md = item.export_to_markdown()
        if md and md.strip():
            return md.strip()
    except Exception:
        pass

    # Fallback: iterate data.grid
    try:
        grid = item.data.grid  # list[list[TableCell]]
        if not grid:
            return _text_from_item(item)
        rows: list[str] = []
        for row_idx, row in enumerate(grid):
            cells = []
            for cell in row:
                cell_text = str(getattr(cell, "text", "") or "").strip()
                cells.append(cell_text)
            rows.append("| " + " | ".join(cells) + " |")
            if row_idx == 0:
                rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
        return "\n".join(rows)
    except Exception:
        pass

    return _text_from_item(item)


def _extract_page_texts(document: Any) -> dict[int, str]:
    """Extract per-page text in document reading order.

    Tables are rendered as Markdown.  Pictures are skipped (captions kept).
    Returns {page_no: text_string}.
    """
    page_parts: dict[int, list[str]] = defaultdict(list)

    # iterate_items() yields (item, level) in reading order
    for item, _level in document.iterate_items():
        label = str(getattr(item, "label", "") or "").strip().lower()

        # Skip raw pictures (keep picture captions which appear as separate text items)
        if label == "picture":
            continue

        prov_list = getattr(item, "prov", None) or []
        if not prov_list:
            continue

        page_no = int(getattr(prov_list[0], "page_no", 1) or 1)

        if "table" in label:
            text = _table_to_markdown(item)
        else:
            text = _text_from_item(item)

        if text and text.strip():
            page_parts[page_no].append(text.strip())

    return {page_no: "\n\n".join(parts) for page_no, parts in sorted(page_parts.items())}


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-page text from PDF/image via Docling")
    parser.add_argument("--source", required=True, help="Path to PDF or image file")
    parser.add_argument("--output", required=True, help="Path for JSON output")
    args = parser.parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)

    model_name = (os.getenv("DOCLING_LAYOUT_MODEL", "docling-layout-heron-101") or "docling-layout-heron-101").strip()
    artifacts_path = (os.getenv("DOCLING_ARTIFACTS_PATH", "") or "").strip()
    device_name = (os.getenv("DOCLING_DEVICE", "auto") or "auto").strip()
    do_table_structure = (os.getenv("DOCLING_TABLE_STRUCTURE", "1") or "1").strip() not in ("0", "false", "no")

    warnings: list[str] = []
    pages: dict[int, str] = {}

    try:
        conversion = _make_converter(
            source_path=source_path,
            model_name=model_name,
            artifacts_path=artifacts_path,
            device_name=device_name,
            do_table_structure=do_table_structure,
        )
        document = getattr(conversion, "document", conversion)
        pages = _extract_page_texts(document)
        if not pages:
            warnings.append("Docling dönüşümü metin üretemedi.")
    except Exception as exc:
        warnings.append(f"Docling metin çıkarma başarısız: {exc}")

    payload = {
        "pages": {str(k): v for k, v in pages.items()},
        "warnings": warnings,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
