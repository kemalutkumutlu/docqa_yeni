from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


TextSource = Literal[
    "pdf_text",
    "ocr",
    "image_ocr",
    "vlm",
    "docai_ocr",
    "docai_image_ocr",
    "paddle_ocr",
    "paddle_vl_ocr",
]
ChunkKind = Literal["parent", "child", "visual", "table", "toc"]
ChunkModality = Literal["text", "visual"]


@dataclass(frozen=True)
class PageText:
    doc_id: str
    file_name: str
    page_number: int  # 1-based
    text: str
    source: TextSource


@dataclass(frozen=True)
class VisualAsset:
    doc_id: str
    file_name: str
    page_number: int  # 1-based
    image_path: str
    summary_text: str = ""
    mime_type: str = "image/png"
    width: int = 0
    height: int = 0
    region_label: str = ""
    region_index: int = 0
    region_count: int = 1
    region_id: str = ""
    crop_type: str = "page"
    region_summary: str = ""
    proposal_source: str = "heuristic"
    proposal_confidence: float = 0.0
    bbox_left: int = 0
    bbox_top: int = 0
    bbox_right: int = 0
    bbox_bottom: int = 0
    bbox_left_norm: float = 0.0
    bbox_top_norm: float = 0.0
    bbox_right_norm: float = 1.0
    bbox_bottom_norm: float = 1.0


@dataclass(frozen=True)
class TableCell:
    row_index: int
    col_index: int
    text: str
    row_span: int = 1
    col_span: int = 1


@dataclass(frozen=True)
class StructuredTable:
    doc_id: str
    file_name: str
    page_number: int
    region_id: str
    image_path: str
    backend: str
    title: str = ""
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    markdown: str = ""
    csv_like: str = ""
    confidence: float = 0.0
    crop_type: str = "table"
    region_label: str = "table"
    proposal_source: str = "heuristic"
    proposal_confidence: float = 0.0
    bbox_left: int = 0
    bbox_top: int = 0
    bbox_right: int = 0
    bbox_bottom: int = 0
    bbox_left_norm: float = 0.0
    bbox_top_norm: float = 0.0
    bbox_right_norm: float = 1.0
    bbox_bottom_norm: float = 1.0
    cells: list[TableCell] = field(default_factory=list)


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    file_name: str
    pages: list[PageText]
    visual_assets: list[VisualAsset] = field(default_factory=list)
    structured_tables: list[StructuredTable] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    toc_text: str = ""  # Raw TOC/bookmark text (not prepended to pages)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    file_name: str
    section_id: str
    parent_id: Optional[str]
    heading_path: str
    page_start: int
    page_end: int
    text: str
    kind: ChunkKind
    modality: ChunkModality = "text"
    image_path: str = ""
    region_label: str = ""
    region_id: str = ""
    crop_type: str = "page"
    region_summary: str = ""
    proposal_source: str = "heuristic"
    proposal_confidence: float = 0.0
    bbox_left: int = 0
    bbox_top: int = 0
    bbox_right: int = 0
    bbox_bottom: int = 0
    bbox_left_norm: float = 0.0
    bbox_top_norm: float = 0.0
    bbox_right_norm: float = 1.0
    bbox_bottom_norm: float = 1.0

