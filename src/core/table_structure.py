from __future__ import annotations

from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import re
from typing import Literal, Optional

from PIL import Image
from .models import Chunk, IngestResult, StructuredTable, TableCell, VisualAsset


TableStructureBackend = Literal["off", "auto", "docai", "gemini", "heuristic"]

_TABLE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "headers": {"type": "array", "items": {"type": "string"}},
        "rows": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    },
    "required": ["headers", "rows"],
}

_TABLE_PROMPT = """\
Extract the table from the image into structured data.

Rules:
- Return only rows and columns that are visibly present.
- Do not invent values for unreadable cells.
- Keep header names concise and literal.
- Preserve row order exactly as shown.
- If the table has merged cells, duplicate the visible value into affected cells.
- Return strict JSON matching the provided schema.
"""


def _gemini_helpers():
    from google.genai import types

    from .gemini_client import build_gemini_client, gemini_model_candidates, is_model_not_found_error

    return types, build_gemini_client, gemini_model_candidates, is_model_not_found_error


@dataclass(frozen=True)
class TableStructureConfig:
    enabled: bool = False
    backend: TableStructureBackend = "auto"
    min_confidence: float = 0.55
    gemini_api_key: str = ""
    gemini_model: str = ""
    docai_project_id: str = ""
    docai_location: str = "us"
    docai_processor_id: str = ""
    docai_processor_version: str = ""
    docai_timeout_seconds: int = 120


def _clean_cell(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _normalize_rows(rows: list[list[str]], *, headers: list[str]) -> list[list[str]]:
    width = max(len(headers), max((len(row) for row in rows), default=0))
    if width <= 0:
        return []
    out: list[list[str]] = []
    for row in rows:
        clean = [_clean_cell(item) for item in row]
        clean = clean[:width] + [""] * max(0, width - len(clean))
        if any(clean):
            out.append(clean)
    return out


def _headers_from_rows(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    first = [_clean_cell(item) for item in rows[0]]
    if not any(first):
        return [f"col_{idx + 1}" for idx in range(len(first))]
    return [item or f"col_{idx + 1}" for idx, item in enumerate(first)]


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not headers and not rows:
        return ""
    headers_use = headers or [f"col_{idx + 1}" for idx in range(max((len(row) for row in rows), default=0))]
    rows_use = _normalize_rows(rows, headers=headers_use)
    if not headers_use:
        return ""
    head = "| " + " | ".join(headers_use) + " |"
    sep = "| " + " | ".join("---" for _ in headers_use) + " |"
    body = ["| " + " | ".join(row[: len(headers_use)]) + " |" for row in rows_use]
    return "\n".join([head, sep] + body).strip()


def _csv_like(headers: list[str], rows: list[list[str]]) -> str:
    headers_use = headers or [f"col_{idx + 1}" for idx in range(max((len(row) for row in rows), default=0))]
    rows_use = _normalize_rows(rows, headers=headers_use)
    lines = [",".join(json.dumps(col, ensure_ascii=False) for col in headers_use)]
    lines.extend(",".join(json.dumps(col, ensure_ascii=False) for col in row[: len(headers_use)]) for row in rows_use)
    return "\n".join(lines).strip()


def _looks_table_asset(asset: VisualAsset) -> bool:
    label = (asset.region_label or "").strip().lower()
    crop_type = (asset.crop_type or "").strip().lower()
    summary = ((asset.region_summary or "") + "\n" + (asset.summary_text or "")).strip()
    if "table" in label or "table" in crop_type:
        return True
    lines = [line for line in summary.splitlines() if line.strip()]
    if sum(1 for line in lines if "|" in line) >= 2:
        return True
    if sum(1 for line in lines if "\t" in line or re.search(r"\S\s{2,}\S", line)) >= 3:
        return True
    return False


def _extract_text(text: str, text_anchor) -> str:
    if not text_anchor or not getattr(text_anchor, "text_segments", None):
        return ""
    parts: list[str] = []
    for segment in text_anchor.text_segments:
        start = int(getattr(segment, "start_index", 0) or 0)
        end = int(getattr(segment, "end_index", 0) or 0)
        if end > start:
            parts.append(text[start:end])
    return " ".join(part.strip() for part in parts if part and part.strip()).strip()


def _resolve_docai_project_id(explicit: str) -> str:
    if explicit:
        return explicit
    for name in ("DOCAI_PROJECT_ID", "VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCP_PROJECT"):
        value = (os.getenv(name, "") or "").strip()
        if value:
            return value
    try:
        import google.auth

        _, project_id = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if project_id:
            return project_id
    except Exception:
        pass
    raise ValueError("Document AI table extraction icin project id bulunamadi.")


def _build_table(
    *,
    asset: VisualAsset,
    backend: str,
    headers: list[str],
    rows: list[list[str]],
    title: str = "",
    cells: Optional[list[TableCell]] = None,
) -> StructuredTable:
    headers_use = [_clean_cell(item) for item in headers]
    rows_use = _normalize_rows(rows, headers=headers_use or _headers_from_rows(rows))
    headers_final = headers_use or _headers_from_rows(rows_use)
    markdown = _markdown_table(headers_final, rows_use)
    csv_text = _csv_like(headers_final, rows_use)
    return StructuredTable(
        doc_id=asset.doc_id,
        file_name=asset.file_name,
        page_number=asset.page_number,
        region_id=asset.region_id or f"p{asset.page_number:04d}:table",
        image_path=asset.image_path,
        backend=backend,
        title=_clean_cell(title or asset.region_summary or asset.summary_text),
        headers=headers_final,
        rows=rows_use,
        markdown=markdown,
        csv_like=csv_text,
        confidence=max(float(asset.proposal_confidence or 0.0), 0.0),
        crop_type=asset.crop_type,
        region_label=asset.region_label or "table",
        proposal_source=asset.proposal_source,
        proposal_confidence=asset.proposal_confidence,
        bbox_left=asset.bbox_left,
        bbox_top=asset.bbox_top,
        bbox_right=asset.bbox_right,
        bbox_bottom=asset.bbox_bottom,
        bbox_left_norm=asset.bbox_left_norm,
        bbox_top_norm=asset.bbox_top_norm,
        bbox_right_norm=asset.bbox_right_norm,
        bbox_bottom_norm=asset.bbox_bottom_norm,
        cells=list(cells or []),
    )


def _extract_via_docai(asset: VisualAsset, cfg: TableStructureConfig) -> StructuredTable:
    if not (cfg.docai_processor_id or "").strip():
        raise ValueError("DOCAI_TABLE_PROCESSOR_ID ayarlanmamis")
    try:
        from google.api_core.client_options import ClientOptions
        from google.cloud import documentai
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"google-cloud-documentai import edilemedi: {exc}") from exc

    project_id = _resolve_docai_project_id((cfg.docai_project_id or "").strip())
    location = (cfg.docai_location or "us").strip()
    processor_id = (cfg.docai_processor_id or "").strip()
    processor_version = (cfg.docai_processor_version or "").strip()
    endpoint = f"{location}-documentai.googleapis.com"
    client = documentai.DocumentProcessorServiceClient(client_options=ClientOptions(api_endpoint=endpoint))
    name = (
        client.processor_version_path(project_id, location, processor_id, processor_version)
        if processor_version
        else client.processor_path(project_id, location, processor_id)
    )

    img = Image.open(asset.image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=buf.getvalue(), mime_type="image/png"),
    )
    result = client.process_document(
        request=request,
        timeout=max(10, min(600, int(cfg.docai_timeout_seconds or 120))),
    )
    document = getattr(result, "document", None)
    pages = list(getattr(document, "pages", None) or [])
    if not pages:
        raise RuntimeError("Document AI table extraction sayfa dondurmedi")
    full_text = getattr(document, "text", "") or ""
    table = next((item for item in getattr(pages[0], "tables", None) or []), None)
    if table is None:
        raise RuntimeError("Document AI table bulamadi")

    headers: list[str] = []
    body_rows: list[list[str]] = []
    cells: list[TableCell] = []
    for row_idx, row in enumerate(getattr(table, "header_rows", None) or []):
        row_values: list[str] = []
        for col_idx, cell in enumerate(getattr(row, "cells", None) or []):
            text = _extract_text(full_text, getattr(cell, "layout", None).text_anchor if getattr(cell, "layout", None) else None)
            text = _clean_cell(text)
            row_values.append(text)
            cells.append(TableCell(row_index=row_idx, col_index=col_idx, text=text))
        if row_values:
            headers = row_values
    header_rows_count = len(getattr(table, "header_rows", None) or [])
    for row_offset, row in enumerate(getattr(table, "body_rows", None) or []):
        row_values: list[str] = []
        for col_idx, cell in enumerate(getattr(row, "cells", None) or []):
            text = _extract_text(full_text, getattr(cell, "layout", None).text_anchor if getattr(cell, "layout", None) else None)
            text = _clean_cell(text)
            row_values.append(text)
            cells.append(TableCell(row_index=header_rows_count + row_offset, col_index=col_idx, text=text))
        if any(row_values):
            body_rows.append(row_values)
    if not headers and not body_rows:
        raise RuntimeError("Document AI table yapisi bos sonuc dondurdu")
    return _build_table(
        asset=asset,
        backend="docai_table",
        headers=headers,
        rows=body_rows,
        title=asset.region_summary,
        cells=cells,
    )


def _extract_via_gemini(asset: VisualAsset, cfg: TableStructureConfig) -> StructuredTable:
    types, build_gemini_client, gemini_model_candidates, is_model_not_found_error = _gemini_helpers()
    img_bytes = Path(asset.image_path).read_bytes()
    last_error: Exception | None = None
    model_name = (cfg.gemini_model or "").strip()
    if not model_name:
        raise ValueError("Gemini table extraction modeli ayarlanmamis")
    for candidate in gemini_model_candidates(model_name):
        try:
            client = build_gemini_client(cfg.gemini_api_key, model_name=candidate)
            resp = client.models.generate_content(
                model=candidate,
                contents=[
                    types.Part.from_text(text=_TABLE_PROMPT),
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    maxOutputTokens=4096,
                    responseMimeType="application/json",
                    responseSchema=_TABLE_JSON_SCHEMA,
                ),
            )
            payload = json.loads((resp.text or "").strip() or "{}")
            headers = [str(item or "").strip() for item in payload.get("headers", []) or []]
            rows = [[str(cell or "").strip() for cell in row] for row in payload.get("rows", []) or [] if isinstance(row, list)]
            if not headers and not rows:
                raise RuntimeError("Gemini table extraction bos sonuc dondurdu")
            return _build_table(
                asset=asset,
                backend="gemini_table",
                headers=headers,
                rows=rows,
                title=str(payload.get("title", "") or ""),
            )
        except Exception as exc:
            last_error = exc
            if not is_model_not_found_error(exc):
                break
    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini table extraction basarisiz")


def _extract_via_heuristic(asset: VisualAsset) -> StructuredTable:
    raw = (asset.summary_text or asset.region_summary or "").strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    parsed: list[list[str]] = []
    for line in lines:
        if "|" in line:
            cols = [_clean_cell(col) for col in line.strip("|").split("|")]
        else:
            cols = [_clean_cell(col) for col in re.split(r"\t+|\s{2,}", line) if _clean_cell(col)]
        if cols and all(re.fullmatch(r"[:\- ]+", col or "") for col in cols):
            continue
        if len(cols) >= 2:
            parsed.append(cols)
    if len(parsed) < 2:
        raise RuntimeError("Heuristic table extraction icin yeterli yapisal satir bulunamadi")
    headers = parsed[0]
    rows = parsed[1:]
    return _build_table(
        asset=asset,
        backend="heuristic_table",
        headers=headers,
        rows=rows,
        title=asset.region_summary,
    )


def extract_tables_from_assets(
    assets: list[VisualAsset],
    *,
    cfg: TableStructureConfig,
) -> tuple[list[StructuredTable], list[str]]:
    if not cfg.enabled:
        return [], []

    tables: list[StructuredTable] = []
    warnings: list[str] = []
    seen: set[tuple[int, str]] = set()
    for asset in assets:
        key = (asset.page_number, asset.region_id or asset.image_path)
        if key in seen:
            continue
        seen.add(key)
        if not _looks_table_asset(asset):
            continue
        if asset.proposal_confidence and asset.proposal_confidence < float(cfg.min_confidence or 0.0):
            if "table" not in (asset.crop_type or "").lower() and "table" not in (asset.region_label or "").lower():
                continue
        try:
            if cfg.backend in ("auto", "docai") and (cfg.docai_processor_id or "").strip():
                tables.append(_extract_via_docai(asset, cfg))
                continue
        except Exception as exc:
            warnings.append(f"DocAI table extraction basarisiz ({asset.file_name} page {asset.page_number}): {exc}")
        try:
            if cfg.backend in ("auto", "gemini") and (cfg.gemini_model or "").strip():
                tables.append(_extract_via_gemini(asset, cfg))
                continue
        except Exception as exc:
            warnings.append(f"Gemini table extraction basarisiz ({asset.file_name} page {asset.page_number}): {exc}")
        try:
            if cfg.backend in ("auto", "heuristic", "gemini", "docai"):
                tables.append(_extract_via_heuristic(asset))
                continue
        except Exception as exc:
            warnings.append(f"Heuristic table extraction basarisiz ({asset.file_name} page {asset.page_number}): {exc}")
    return tables, warnings


def table_chunks_from_ingest(ingest: IngestResult, *, max_rows: int = 40) -> list[Chunk]:
    if not ingest.structured_tables:
        return []
    chunks: list[Chunk] = []
    for idx, table in enumerate(ingest.structured_tables, start=1):
        rows = table.rows[:max_rows]
        markdown = _markdown_table(table.headers, rows) or table.markdown
        summary_line = f"Structured table extracted via {table.backend}."
        if table.title:
            summary_line += f" Title: {table.title}."
        text_parts = [
            summary_line,
            f"Page: {table.page_number}",
            f"Region: {table.region_id or 'table'}",
        ]
        if markdown:
            text_parts.extend(["", markdown])
        text = "\n".join(text_parts).strip()
        region_suffix = f":{table.region_id}" if table.region_id else f":t{idx:02d}"
        chunks.append(
            Chunk(
                chunk_id=f"{ingest.doc_id}:table:p{table.page_number:04d}{region_suffix}",
                doc_id=ingest.doc_id,
                file_name=ingest.file_name,
                section_id=f"table_p{table.page_number:04d}_{idx:02d}",
                parent_id="root",
                heading_path=f"{ingest.file_name} / Table Page {table.page_number}",
                page_start=table.page_number,
                page_end=table.page_number,
                text=text,
                kind="table",
                modality="text",
                image_path=table.image_path,
                region_label=table.region_label,
                region_id=table.region_id,
                crop_type=table.crop_type,
                region_summary=table.title or (table.markdown.splitlines()[0] if table.markdown else ""),
                proposal_source=table.proposal_source,
                proposal_confidence=table.proposal_confidence or table.confidence,
                bbox_left=table.bbox_left,
                bbox_top=table.bbox_top,
                bbox_right=table.bbox_right,
                bbox_bottom=table.bbox_bottom,
                bbox_left_norm=table.bbox_left_norm,
                bbox_top_norm=table.bbox_top_norm,
                bbox_right_norm=table.bbox_right_norm,
                bbox_bottom_norm=table.bbox_bottom_norm,
            )
        )
    return chunks
