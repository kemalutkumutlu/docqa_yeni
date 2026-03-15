from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

from PIL import Image

from .layout_detector import detector_regions_available, get_layout_detector


VisualChunkLevel = Literal["page", "region"]
VisualRegionSource = Literal["heuristic", "detector"]


@dataclass(frozen=True)
class RegionProposal:
    page_number: int
    region_index: int
    region_count: int
    region_label: str
    region_id: str
    crop_type: str
    summary_text: str
    proposal_source: str
    proposal_confidence: float
    bbox_left: int
    bbox_top: int
    bbox_right: int
    bbox_bottom: int
    bbox_left_norm: float
    bbox_top_norm: float
    bbox_right_norm: float
    bbox_bottom_norm: float


def _region_labels(region_count: int) -> list[str]:
    if region_count <= 1:
        return [""]
    if region_count == 2:
        return ["top", "bottom"]
    return ["top", "middle", "bottom"][:region_count]


def _looks_table_like(summary_text: str) -> bool:
    text = (summary_text or "").strip()
    if not text:
        return False
    lines = [line for line in text.splitlines() if line.strip()]
    pipe_lines = sum(1 for line in lines if "|" in line)
    tab_lines = sum(1 for line in lines if "\t" in line)
    dense_gap_lines = sum(1 for line in lines if re.search(r"\S\s{2,}\S", line))
    return pipe_lines >= 2 or tab_lines >= 2 or dense_gap_lines >= 3


def _looks_toc_like(summary_text: str) -> bool:
    text = (summary_text or "").strip()
    if not text:
        return False
    lower = text.lower()
    if "içindekiler" in lower or "icindekiler" in lower or "\ncontents" in lower or lower.startswith("contents"):
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    leader_lines = sum(
        1
        for line in lines
        if re.search(r"(\.{3,}|\. \. \.|…)", line) or re.search(r"\b\d+\s*$", line)
    )
    return leader_lines >= 4


def _looks_spec_list_like(summary_text: str) -> bool:
    text = (summary_text or "").strip()
    if not text or _looks_toc_like(text):
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_lines = sum(
        1
        for line in lines
        if re.match(r"^([#*•\-]|\d+[.)]|[A-Z]\.)\s+", line)
    )
    keyword_lines = sum(
        1
        for line in lines
        if re.search(
            r"\b(teslimat|deliverable|gereksinim|requirement|madde|adım|step|checklist|çıktı|cikti)\b",
            line,
            re.IGNORECASE,
        )
    )
    short_structured = sum(1 for line in lines if len(line) <= 110 and (":" in line or "—" in line or "-" in line))
    return bullet_lines >= 3 or keyword_lines >= 2 or (bullet_lines >= 2 and short_structured >= 3)


def _looks_form_like(summary_text: str) -> bool:
    text = (summary_text or "").strip()
    if not text:
        return False
    if _looks_toc_like(text) or _looks_spec_list_like(text):
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    short_lines = [line for line in lines if len(line) <= 90]
    field_like = sum(
        1
        for line in short_lines
        if ":" in line
        or re.search(
            r"\b(ad|soyad|name|date|tarih|field|label|checkbox|tick|seçim|secim)\b",
            line,
            re.IGNORECASE,
        )
    )
    if field_like < 3:
        return False
    return (field_like / max(1, len(short_lines))) >= 0.35


def _target_region_plan(img: Image.Image, summary_text: str) -> tuple[int, str]:
    lines = [line.strip() for line in (summary_text or "").splitlines() if line.strip()]
    height = int(getattr(img, "height", 0) or 0)
    if _looks_toc_like(summary_text):
        if height >= 1200 and len(lines) >= 8:
            return 3, "toc_vertical"
        return 2, "toc_vertical"
    if _looks_spec_list_like(summary_text):
        if height >= 1200 and len(lines) >= 8:
            return 3, "spec_vertical"
        return 2, "spec_vertical"
    if _looks_table_like(summary_text):
        if height >= 1200 and len(lines) >= 8:
            return 3, "table_vertical"
        return 2, "table_vertical"
    if _looks_form_like(summary_text):
        if height >= 1000 and len(lines) >= 8:
            return 3, "form_vertical"
        return 2, "form_vertical"
    if height >= 1500 and len(lines) >= 12:
        return 3, "dense_vertical"
    if height >= 900 and len(lines) >= 6:
        return 2, "vertical_split"
    return 1, "page"


def _split_summary_for_regions(summary_text: str, region_count: int) -> list[str]:
    clean = (summary_text or "").strip()
    if region_count <= 1:
        return [clean]

    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    if lines:
        chunk_size = max(1, (len(lines) + region_count - 1) // region_count)
        out = ["\n".join(lines[idx: idx + chunk_size]).strip() for idx in range(0, len(lines), chunk_size)]
        out = [item for item in out if item]
        if len(out) >= region_count:
            return out[:region_count]

    words = clean.split()
    if words:
        chunk_size = max(1, (len(words) + region_count - 1) // region_count)
        out = [" ".join(words[idx: idx + chunk_size]).strip() for idx in range(0, len(words), chunk_size)]
        out = [item for item in out if item]
        if len(out) >= region_count:
            return out[:region_count]

    return [clean] if clean else [""]


def _weighted_vertical_region_boxes(
    width: int,
    height: int,
    weights: list[float],
) -> list[tuple[int, int, int, int]]:
    total = sum(weights) or 1.0
    ratios = [max(0.01, weight) / total for weight in weights]
    boxes: list[tuple[int, int, int, int]] = []
    top = 0
    consumed = 0
    for idx, ratio in enumerate(ratios, start=1):
        if idx == len(ratios):
            bottom = height
        else:
            consumed += int(round(height * ratio))
            bottom = min(height, max(top + 1, consumed))
        boxes.append((0, top, width, bottom))
        top = bottom
    return boxes


def _vertical_region_boxes(
    width: int,
    height: int,
    region_count: int,
    crop_type: str,
) -> list[tuple[int, int, int, int]]:
    if region_count <= 1:
        return [(0, 0, width, height)]
    if crop_type == "table_vertical":
        weights = [0.22, 0.39, 0.39] if region_count >= 3 else [0.34, 0.66]
        return _weighted_vertical_region_boxes(width, height, weights[:region_count])
    if crop_type == "form_vertical":
        weights = [0.28, 0.36, 0.36] if region_count >= 3 else [0.42, 0.58]
        return _weighted_vertical_region_boxes(width, height, weights[:region_count])
    if crop_type == "dense_vertical":
        weights = [0.30, 0.35, 0.35] if region_count >= 3 else [0.48, 0.52]
        return _weighted_vertical_region_boxes(width, height, weights[:region_count])
    if crop_type == "toc_vertical":
        weights = [0.20, 0.40, 0.40] if region_count >= 3 else [0.36, 0.64]
        return _weighted_vertical_region_boxes(width, height, weights[:region_count])
    if crop_type == "spec_vertical":
        weights = [0.24, 0.38, 0.38] if region_count >= 3 else [0.40, 0.60]
        return _weighted_vertical_region_boxes(width, height, weights[:region_count])
    step = max(1, height // region_count)
    return [
        (0, idx * step, width, height if idx == region_count - 1 else min(height, (idx + 1) * step))
        for idx in range(region_count)
    ]


def _normalized_box(
    box: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    left, top, right, bottom = box
    width_safe = max(1, width)
    height_safe = max(1, height)
    return (
        round(left / width_safe, 6),
        round(top / height_safe, 6),
        round(right / width_safe, 6),
        round(bottom / height_safe, 6),
    )


def _proposal_confidence(
    *,
    crop_type: str,
    region_count: int,
    source: str,
) -> float:
    if source == "detector":
        return 0.85
    base = 0.48
    if crop_type.startswith("table_") or crop_type.startswith("form_") or crop_type.startswith("toc_"):
        base = 0.62
    elif crop_type.startswith("spec_"):
        base = 0.58
    elif crop_type == "dense_vertical":
        base = 0.56
    elif region_count == 1:
        base = 0.42
    return round(base, 3)


def _heuristic_region_proposals(
    img: Image.Image,
    *,
    page_number: int,
    level: str,
    summary_text: str,
    proposal_source: str,
) -> list[RegionProposal]:
    width = int(getattr(img, "width", 0) or 0)
    height = int(getattr(img, "height", 0) or 0)
    if level != "region":
        left_norm, top_norm, right_norm, bottom_norm = _normalized_box((0, 0, width, height), width=width, height=height)
        return [
            RegionProposal(
                page_number=page_number,
                region_index=1,
                region_count=1,
                region_label="",
                region_id=f"p{page_number:04d}",
                crop_type="page",
                summary_text=(summary_text or "").strip(),
                proposal_source=proposal_source,
                proposal_confidence=_proposal_confidence(crop_type="page", region_count=1, source=proposal_source),
                bbox_left=0,
                bbox_top=0,
                bbox_right=width,
                bbox_bottom=height,
                bbox_left_norm=left_norm,
                bbox_top_norm=top_norm,
                bbox_right_norm=right_norm,
                bbox_bottom_norm=bottom_norm,
            )
        ]

    region_count, crop_type = _target_region_plan(img, summary_text)
    if region_count <= 1:
        left_norm, top_norm, right_norm, bottom_norm = _normalized_box((0, 0, width, height), width=width, height=height)
        return [
            RegionProposal(
                page_number=page_number,
                region_index=1,
                region_count=1,
                region_label="",
                region_id=f"p{page_number:04d}",
                crop_type="page",
                summary_text=(summary_text or "").strip(),
                proposal_source=proposal_source,
                proposal_confidence=_proposal_confidence(crop_type="page", region_count=1, source=proposal_source),
                bbox_left=0,
                bbox_top=0,
                bbox_right=width,
                bbox_bottom=height,
                bbox_left_norm=left_norm,
                bbox_top_norm=top_norm,
                bbox_right_norm=right_norm,
                bbox_bottom_norm=bottom_norm,
            )
        ]

    summaries = _split_summary_for_regions(summary_text, region_count)
    labels = _region_labels(region_count)
    boxes = _vertical_region_boxes(width, height, region_count, crop_type)
    proposals: list[RegionProposal] = []
    for idx, box in enumerate(boxes, start=1):
        left, top, right, bottom = box
        left_norm, top_norm, right_norm, bottom_norm = _normalized_box(box, width=width, height=height)
        summary = summaries[idx - 1] if idx - 1 < len(summaries) else (summary_text or "").strip()
        proposals.append(
            RegionProposal(
                page_number=page_number,
                region_index=idx,
                region_count=region_count,
                region_label=labels[idx - 1] if idx - 1 < len(labels) else f"region-{idx}",
                region_id=f"p{page_number:04d}:r{idx:02d}",
                crop_type=crop_type,
                summary_text=summary,
                proposal_source=proposal_source,
                proposal_confidence=_proposal_confidence(crop_type=crop_type, region_count=region_count, source=proposal_source),
                bbox_left=left,
                bbox_top=top,
                bbox_right=right,
                bbox_bottom=bottom,
                bbox_left_norm=left_norm,
                bbox_top_norm=top_norm,
                bbox_right_norm=right_norm,
                bbox_bottom_norm=bottom_norm,
            )
        )
    return proposals


def plan_regions(
    img: Image.Image,
    *,
    page_number: int,
    level: VisualChunkLevel,
    source: VisualRegionSource,
    summary_text: str,
    document_name: str = "",
    detector_backend: str = "none",
    detector_dir: Path | None = None,
    docai_project_id: str = "",
    docai_location: str = "us",
    docai_processor_id: str = "",
    docai_processor_version: str = "pretrained-layout-parser-v1.6-pro-2025-12-01",
    docai_timeout_seconds: int = 120,
    docling_python_bin: str = "",
    docling_layout_model: str = "docling-layout-heron-101",
    docling_artifacts_path: Path | None = None,
    docling_device: str = "auto",
) -> tuple[list[RegionProposal], list[str]]:
    requested_source = (source or "heuristic").strip().lower()
    if requested_source not in ("heuristic", "detector"):
        requested_source = "heuristic"

    if requested_source == "detector":
        detector = get_layout_detector(
            requested_source,
            backend=detector_backend,
            detector_dir=detector_dir,
            docai_project_id=docai_project_id,
            docai_location=docai_location,
            docai_processor_id=docai_processor_id,
            docai_processor_version=docai_processor_version,
            docai_timeout_seconds=docai_timeout_seconds,
            docling_python_bin=docling_python_bin,
            docling_layout_model=docling_layout_model,
            docling_artifacts_path=docling_artifacts_path,
            docling_device=docling_device,
        )
        result = detector.detect(
            img,
            page_number=page_number,
            summary_text=summary_text,
            document_name=document_name,
        )
        if detector_regions_available(result.regions):
            width = int(getattr(img, "width", 0) or 0)
            height = int(getattr(img, "height", 0) or 0)
            labels = _region_labels(len(result.regions))
            proposals: list[RegionProposal] = []
            for idx, region in enumerate(result.regions, start=1):
                left = max(0, int(region.bbox_left))
                top = max(0, int(region.bbox_top))
                right = max(left + 1, int(region.bbox_right))
                bottom = max(top + 1, int(region.bbox_bottom))
                left_norm, top_norm, right_norm, bottom_norm = _normalized_box(
                    (left, top, right, bottom),
                    width=width,
                    height=height,
                )
                proposals.append(
                    RegionProposal(
                        page_number=page_number,
                        region_index=idx,
                        region_count=len(result.regions),
                        region_label=(region.label or labels[idx - 1] if idx - 1 < len(labels) else f"region-{idx}"),
                        region_id=f"p{page_number:04d}:r{idx:02d}",
                        crop_type=(region.crop_type or "detector_region").strip(),
                        summary_text=(region.summary_text or summary_text or "").strip(),
                        proposal_source="detector",
                        proposal_confidence=max(0.0, min(1.0, float(region.confidence or 0.0))),
                        bbox_left=left,
                        bbox_top=top,
                        bbox_right=right,
                        bbox_bottom=bottom,
                        bbox_left_norm=left_norm,
                        bbox_top_norm=top_norm,
                        bbox_right_norm=right_norm,
                        bbox_bottom_norm=bottom_norm,
                    )
                )
            return proposals, list(result.warnings)

        return (
            _heuristic_region_proposals(
                img,
                page_number=page_number,
                level=level,
                summary_text=summary_text,
                proposal_source="heuristic",
            ),
            list(result.warnings),
        )

    return (
        _heuristic_region_proposals(
            img,
            page_number=page_number,
            level=level,
            summary_text=summary_text,
            proposal_source="heuristic",
        ),
        [],
    )
