from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image


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


def _make_converter(*, source_path: Path, model_name: str, artifacts_path: str, device_name: str):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter
    try:
        from docling.document_converter import ImageFormatOption as DoclingImageFormatOption
    except Exception:
        from docling.document_converter import PdfFormatOption as DoclingImageFormatOption
    from docling.document_converter import PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    for attr_name, attr_value in (
        ("do_ocr", False),
        ("do_table_structure", False),
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


def _iter_pages(document) -> list[tuple[int, Any]]:
    pages = getattr(document, "pages", None) or {}
    if isinstance(pages, dict):
        rows: list[tuple[int, Any]] = []
        for key, value in pages.items():
            try:
                rows.append((int(key), value))
            except Exception:
                continue
        return sorted(rows, key=lambda item: item[0])
    if isinstance(pages, list):
        return [(idx + 1, page) for idx, page in enumerate(pages)]
    return []


def _page_dimensions(document) -> dict[int, tuple[float, float]]:
    dims: dict[int, tuple[float, float]] = {}
    for page_no, page in _iter_pages(document):
        size = getattr(page, "size", None)
        width = float(getattr(size, "width", 0.0) or getattr(page, "width", 0.0) or 0.0)
        height = float(getattr(size, "height", 0.0) or getattr(page, "height", 0.0) or 0.0)
        if width > 0 and height > 0:
            dims[page_no] = (width, height)
    return dims


def _text_from_item(item: Any) -> str:
    for attr_name in ("text", "orig", "content", "caption_text", "name"):
        value = getattr(item, attr_name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_bbox(
    bbox: Any,
    *,
    image_width: int,
    image_height: int,
    page_width: float,
    page_height: float,
) -> tuple[int, int, int, int] | None:
    left = float(getattr(bbox, "l", 0.0) or getattr(bbox, "left", 0.0) or 0.0)
    right = float(getattr(bbox, "r", 0.0) or getattr(bbox, "right", 0.0) or 0.0)
    top_raw = float(getattr(bbox, "t", 0.0) or getattr(bbox, "top", 0.0) or 0.0)
    bottom_raw = float(getattr(bbox, "b", 0.0) or getattr(bbox, "bottom", 0.0) or 0.0)
    if right <= left:
        return None
    page_width_safe = max(1.0, page_width)
    page_height_safe = max(1.0, page_height)
    scale_x = float(image_width) / page_width_safe
    scale_y = float(image_height) / page_height_safe
    y_from_top_a = page_height_safe - top_raw
    y_from_top_b = page_height_safe - bottom_raw
    top = min(y_from_top_a, y_from_top_b)
    bottom = max(y_from_top_a, y_from_top_b)
    left_i = max(0, min(image_width, int(round(left * scale_x))))
    right_i = max(0, min(image_width, int(round(right * scale_x))))
    top_i = max(0, min(image_height, int(round(top * scale_y))))
    bottom_i = max(0, min(image_height, int(round(bottom * scale_y))))
    if right_i <= left_i or bottom_i <= top_i:
        return None
    return left_i, top_i, right_i, bottom_i


def _collect_rows(
    document,
    *,
    image_width: int,
    image_height: int,
    page_number: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    page_dims = _page_dimensions(document)
    special_rows: list[dict[str, Any]] = []
    text_rows: list[dict[str, Any]] = []
    stats = {"special_candidates": 0, "text_candidates": 0}
    collection_specs = (
        ("tables", "table", "docling_table", 0.95, True),
        ("pictures", "picture", "docling_picture", 0.80, True),
        ("key_value_items", "form", "docling_form", 0.82, True),
        ("form_items", "form", "docling_form", 0.78, True),
        ("texts", "block", "docling_block", 0.64, False),
    )
    for attr_name, default_label, crop_type, confidence, special in collection_specs:
        items = getattr(document, attr_name, None) or []
        for item in items:
            prov_list = getattr(item, "prov", None) or []
            text = _text_from_item(item)
            label = str(getattr(item, "label", "") or default_label).strip().lower() or default_label
            for prov in prov_list:
                page_no = int(getattr(prov, "page_no", 1) or 1)
                if page_number is not None and page_no != int(page_number):
                    continue
                page_width, page_height = page_dims.get(page_no, (float(image_width), float(image_height)))
                bbox = _normalize_bbox(
                    getattr(prov, "bbox", None),
                    image_width=image_width,
                    image_height=image_height,
                    page_width=page_width,
                    page_height=page_height,
                )
                if bbox is None:
                    continue
                row = {
                    "bbox_left": bbox[0],
                    "bbox_top": bbox[1],
                    "bbox_right": bbox[2],
                    "bbox_bottom": bbox[3],
                    "label": label,
                    "crop_type": crop_type,
                    "confidence": confidence,
                    "summary_text": text,
                }
                if special:
                    stats["special_candidates"] += 1
                    special_rows.append(row)
                else:
                    stats["text_candidates"] += 1
                    text_rows.append(row)
    return special_rows, text_rows, stats


def _collect_iter_item_rows(
    document,
    *,
    image_width: int,
    image_height: int,
    page_number: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    page_dims = _page_dimensions(document)
    rows: list[dict[str, Any]] = []
    stats = {"iter_items_seen": 0, "iter_items_kept": 0}
    seen: set[tuple[int, int, int, int, str]] = set()
    for item, _level in document.iterate_items():
        stats["iter_items_seen"] += 1
        prov_list = getattr(item, "prov", None) or []
        if not prov_list:
            continue
        label = str(getattr(item, "label", "") or "").strip().lower()
        if label in ("picture", "table", "caption", "footnote", "page_footer", "page_header"):
            continue
        crop_type = "docling_block"
        if "table" in label:
            crop_type = "docling_table"
        elif "form" in label or "key" in label or "value" in label:
            crop_type = "docling_form"
        text = _text_from_item(item)
        for prov in prov_list:
            page_no = int(getattr(prov, "page_no", 1) or 1)
            if page_number is not None and page_no != int(page_number):
                continue
            page_width, page_height = page_dims.get(page_no, (float(image_width), float(image_height)))
            bbox = _normalize_bbox(
                getattr(prov, "bbox", None),
                image_width=image_width,
                image_height=image_height,
                page_width=page_width,
                page_height=page_height,
            )
            if bbox is None:
                continue
            key = (bbox[0], bbox[1], bbox[2], bbox[3], crop_type)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "bbox_left": bbox[0],
                    "bbox_top": bbox[1],
                    "bbox_right": bbox[2],
                    "bbox_bottom": bbox[3],
                    "label": label or "block",
                    "crop_type": crop_type,
                    "confidence": 0.58 if crop_type == "docling_block" else 0.74,
                    "summary_text": text,
                }
            )
            stats["iter_items_kept"] += 1
    return rows, stats


def _x_overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    left = max(int(a["bbox_left"]), int(b["bbox_left"]))
    right = min(int(a["bbox_right"]), int(b["bbox_right"]))
    overlap = max(0, right - left)
    width = max(1, min(int(a["bbox_right"]) - int(a["bbox_left"]), int(b["bbox_right"]) - int(b["bbox_left"])))
    return overlap / width


def _merge_text_rows(text_rows: list[dict[str, Any]], *, image_height: int) -> list[dict[str, Any]]:
    if not text_rows:
        return []
    rows = sorted(text_rows, key=lambda row: (int(row["bbox_top"]), int(row["bbox_left"])))
    gap_limit = max(24, int(image_height * 0.02))
    merged: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    texts: list[str] = []
    confidences: list[float] = []
    for row in rows:
        if current is None:
            current = dict(row)
            texts = [str(row.get("summary_text", "") or "").strip()]
            confidences = [float(row.get("confidence", 0.0) or 0.0)]
            continue
        gap = int(row["bbox_top"]) - int(current["bbox_bottom"])
        if gap <= gap_limit and _x_overlap_ratio(current, row) >= 0.15:
            current["bbox_left"] = min(int(current["bbox_left"]), int(row["bbox_left"]))
            current["bbox_top"] = min(int(current["bbox_top"]), int(row["bbox_top"]))
            current["bbox_right"] = max(int(current["bbox_right"]), int(row["bbox_right"]))
            current["bbox_bottom"] = max(int(current["bbox_bottom"]), int(row["bbox_bottom"]))
            texts.append(str(row.get("summary_text", "") or "").strip())
            confidences.append(float(row.get("confidence", 0.0) or 0.0))
            continue
        current["summary_text"] = " ".join(part for part in texts if part).strip()[:240]
        current["confidence"] = sum(confidences) / max(1, len(confidences))
        merged.append(current)
        current = dict(row)
        texts = [str(row.get("summary_text", "") or "").strip()]
        confidences = [float(row.get("confidence", 0.0) or 0.0)]
    if current is not None:
        current["summary_text"] = " ".join(part for part in texts if part).strip()[:240]
        current["confidence"] = sum(confidences) / max(1, len(confidences))
        merged.append(current)
    while len(merged) > 6:
        first = merged.pop()
        second = merged.pop()
        combined = {
            "bbox_left": min(int(first["bbox_left"]), int(second["bbox_left"])),
            "bbox_top": min(int(first["bbox_top"]), int(second["bbox_top"])),
            "bbox_right": max(int(first["bbox_right"]), int(second["bbox_right"])),
            "bbox_bottom": max(int(first["bbox_bottom"]), int(second["bbox_bottom"])),
            "label": "block",
            "crop_type": "docling_block",
            "confidence": (float(first.get("confidence", 0.0) or 0.0) + float(second.get("confidence", 0.0) or 0.0)) / 2.0,
            "summary_text": " ".join(
                part for part in (str(second.get("summary_text", "") or "").strip(), str(first.get("summary_text", "") or "").strip()) if part
            )[:240],
        }
        merged.append(combined)
        merged.sort(key=lambda row: (int(row["bbox_top"]), int(row["bbox_left"])))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="")
    parser.add_argument("--image", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--document-name", default="")
    parser.add_argument("--page-number", type=int, default=1)
    parser.add_argument("--image-width", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=0)
    args = parser.parse_args()

    source_arg = (args.source or args.image or "").strip()
    if not source_arg:
        raise SystemExit("source path is required")
    source_path = Path(source_arg)
    output_path = Path(args.output)
    if source_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"):
        image = Image.open(source_path)
        image_width = int(image.width)
        image_height = int(image.height)
    else:
        image_width = max(1, int(args.image_width or 0))
        image_height = max(1, int(args.image_height or 0))
        if image_width <= 1 or image_height <= 1:
            raise SystemExit("image width/height are required for non-image sources")

    conversion = _make_converter(
        source_path=source_path,
        model_name=(os.getenv("DOCLING_LAYOUT_MODEL", "docling-layout-heron-101") or "docling-layout-heron-101").strip(),
        artifacts_path=(os.getenv("DOCLING_ARTIFACTS_PATH", "") or "").strip(),
        device_name=(os.getenv("DOCLING_DEVICE", "auto") or "auto").strip(),
    )
    document = getattr(conversion, "document", conversion)
    special_rows, text_rows, collect_stats = _collect_rows(
        document,
        image_width=image_width,
        image_height=image_height,
        page_number=args.page_number,
    )
    rows = list(special_rows)
    merged_text = _merge_text_rows(text_rows, image_height=image_height)
    if merged_text:
        rows.extend(merged_text[: max(0, 8 - len(rows))])
    iter_rows: list[dict[str, Any]] = []
    iter_stats = {"iter_items_seen": 0, "iter_items_kept": 0}
    if not rows:
        iter_rows, iter_stats = _collect_iter_item_rows(
            document,
            image_width=image_width,
            image_height=image_height,
            page_number=args.page_number,
        )
        rows.extend(iter_rows[:8])
    rows.sort(key=lambda row: (int(row["bbox_top"]), int(row["bbox_left"])))
    warnings: list[str] = []
    if not rows:
        warnings.append(
            "Docling debug: page="
            f"{args.page_number} source={source_path.suffix.lower() or 'unknown'} "
            f"special={collect_stats['special_candidates']} text={collect_stats['text_candidates']} "
            f"iter_seen={iter_stats['iter_items_seen']} iter_kept={iter_stats['iter_items_kept']}"
        )
    payload = {
        "regions": rows[:8],
        "warnings": warnings,
        "debug": {
            "page_number": int(args.page_number),
            "source_suffix": source_path.suffix.lower(),
            "special_candidates": collect_stats["special_candidates"],
            "text_candidates": collect_stats["text_candidates"],
            "iter_items_seen": iter_stats["iter_items_seen"],
            "iter_items_kept": iter_stats["iter_items_kept"],
            "returned_regions": len(rows[:8]),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
