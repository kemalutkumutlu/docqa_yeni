from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import Chunk, IngestResult


@dataclass(frozen=True)
class MultimodalConfig:
    enabled: bool = False
    assets_dir: Path | None = None
    attach_images_to_generation: bool = True
    chunk_level: str = "page"


def visual_chunks_from_ingest(ingest: IngestResult, *, max_summary_chars: int = 1800) -> list[Chunk]:
    if not ingest.visual_assets:
        return []

    page_text_by_num = {p.page_number: p.text for p in ingest.pages}
    chunks: list[Chunk] = []
    for asset in ingest.visual_assets:
        summary = (asset.summary_text or page_text_by_num.get(asset.page_number, "") or "").strip()
        if summary:
            summary = summary[:max_summary_chars].strip()
        else:
            region_hint = f" ({asset.region_label})" if asset.region_label else ""
            summary = f"Visual content from page {asset.page_number}{region_hint}."
        region_suffix = ""
        heading_suffix = ""
        if asset.region_count > 1:
            region_suffix = f":r{asset.region_index:02d}"
            heading_suffix = f" / Region {asset.region_label or asset.region_index}"
        chunks.append(
            Chunk(
                chunk_id=f"{ingest.doc_id}:visual:p{asset.page_number:04d}{region_suffix}",
                doc_id=ingest.doc_id,
                file_name=ingest.file_name,
                section_id=f"visual_p{asset.page_number:04d}{region_suffix.replace(':', '_')}",
                parent_id="root",
                heading_path=f"{ingest.file_name} / Visual Page {asset.page_number}{heading_suffix}",
                page_start=asset.page_number,
                page_end=asset.page_number,
                text=summary,
                kind="visual",
                modality="visual",
                image_path=asset.image_path,
            )
        )
    return chunks
