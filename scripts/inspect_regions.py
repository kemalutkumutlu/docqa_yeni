from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    from src.config import load_settings
    from src.core.ingestion import OCRConfig, ingest_any
    from src.core.multimodal import MultimodalConfig, visual_chunks_from_ingest
    from src.core.vlm_extract import VLMConfig
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import load_settings  # type: ignore  # noqa: E402
    from src.core.ingestion import OCRConfig, ingest_any  # type: ignore  # noqa: E402
    from src.core.multimodal import MultimodalConfig, visual_chunks_from_ingest  # type: ignore  # noqa: E402
    from src.core.vlm_extract import VLMConfig  # type: ignore  # noqa: E402


def _payload(
    path: Path,
    *,
    level: str,
    source: str,
    detector_backend: str,
    detector_dir: Path | None,
    docai_project_id: str,
    docai_location: str,
    docai_processor_id: str,
    docai_processor_version: str,
    docai_timeout_seconds: int,
    max_assets: int,
) -> dict:
    settings = load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    ocr = OCRConfig(
        enabled=getattr(settings, "ocr_enabled", True),
        lang="tur+eng",
        tesseract_cmd=settings.tesseract_cmd,
        tessdata_prefix=settings.tessdata_prefix,
        tesseract_config=getattr(settings, "tesseract_config", None),
    )
    vlm = VLMConfig(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        mode=settings.vlm_mode,
        max_pages=settings.vlm_max_pages,
        provider=settings.vlm_provider,
        ollama_base_url=settings.ollama_base_url,
        ollama_vlm_model=settings.ollama_vlm_model,
        ollama_timeout=settings.ollama_timeout,
    )
    multimodal = MultimodalConfig(
        enabled=True,
        assets_dir=settings.data_dir / "region_inspect_assets",
        chunk_level=level,
        region_source=source,
        detector_backend=detector_backend,
        detector_dir=detector_dir,
        docai_project_id=docai_project_id,
        docai_location=docai_location,
        docai_processor_id=docai_processor_id,
        docai_processor_version=docai_processor_version,
        docai_timeout_seconds=docai_timeout_seconds,
    )

    ingest = ingest_any(path, ocr=ocr, display_name=path.name, vlm=vlm, multimodal=multimodal)
    chunks = visual_chunks_from_ingest(ingest)

    asset_rows = []
    for asset in ingest.visual_assets[:max_assets]:
        asset_rows.append(
            {
                "page": asset.page_number,
                "region_label": asset.region_label,
                "region_id": asset.region_id,
                "crop_type": asset.crop_type,
                "region_summary": asset.region_summary,
                "proposal_source": asset.proposal_source,
                "proposal_confidence": asset.proposal_confidence,
                "bbox": [asset.bbox_left, asset.bbox_top, asset.bbox_right, asset.bbox_bottom],
                "bbox_norm": [
                    asset.bbox_left_norm,
                    asset.bbox_top_norm,
                    asset.bbox_right_norm,
                    asset.bbox_bottom_norm,
                ],
                "image_path": asset.image_path,
                "size": [asset.width, asset.height],
            }
        )

    chunk_rows = []
    for chunk in chunks[:max_assets]:
        chunk_rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "page": chunk.page_start,
                "region_label": chunk.region_label,
                "region_id": chunk.region_id,
                "crop_type": chunk.crop_type,
                "region_summary": chunk.region_summary,
                "proposal_source": chunk.proposal_source,
                "proposal_confidence": chunk.proposal_confidence,
                "bbox": [chunk.bbox_left, chunk.bbox_top, chunk.bbox_right, chunk.bbox_bottom],
                "bbox_norm": [
                    chunk.bbox_left_norm,
                    chunk.bbox_top_norm,
                    chunk.bbox_right_norm,
                    chunk.bbox_bottom_norm,
                ],
                "heading_path": chunk.heading_path,
                "text_preview": chunk.text[:220],
            }
        )

    return {
        "file": ingest.file_name,
        "doc_id": ingest.doc_id,
        "visual_chunk_level": level,
        "visual_region_source": source,
        "visual_detector_backend": detector_backend,
        "page_count": len(ingest.pages),
        "visual_asset_count": len(ingest.visual_assets),
        "visual_chunk_count": len(chunks),
        "warnings": ingest.warnings,
        "assets": asset_rows,
        "chunks": chunk_rows,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Inspect page/region visual chunk planning without building the full index.")
    ap.add_argument("path", type=str, help="PDF / PNG / JPG path")
    ap.add_argument("--level", choices=["page", "region"], default="region", help="Visual chunk granularity to inspect")
    ap.add_argument("--source", choices=["heuristic", "detector"], default="heuristic", help="Region proposal source")
    ap.add_argument("--detector-backend", choices=["none", "sidecar", "docai"], default="none", help="Detector backend")
    ap.add_argument("--detector-dir", type=str, default="", help="Sidecar detector directory")
    ap.add_argument("--docai-project-id", type=str, default="", help="Document AI project id override")
    ap.add_argument("--docai-location", type=str, default="", help="Document AI location override")
    ap.add_argument("--docai-processor-id", type=str, default="", help="Document AI layout processor id override")
    ap.add_argument("--docai-processor-version", type=str, default="", help="Document AI processor version override")
    ap.add_argument("--docai-timeout-seconds", type=int, default=0, help="Document AI timeout override")
    ap.add_argument("--max-assets", type=int, default=12, help="Max assets/chunks to print")
    ap.add_argument("--json", action="store_true", help="Emit JSON only")
    args = ap.parse_args()

    settings = load_settings()
    detector_dir = Path(args.detector_dir).expanduser().resolve() if args.detector_dir else None
    payload = _payload(
        Path(args.path),
        level=args.level,
        source=args.source,
        detector_backend=args.detector_backend,
        detector_dir=detector_dir,
        docai_project_id=(args.docai_project_id or settings.docai_project_id),
        docai_location=(args.docai_location or settings.docai_location),
        docai_processor_id=(args.docai_processor_id or settings.docai_layout_processor_id),
        docai_processor_version=(args.docai_processor_version or settings.docai_layout_processor_version),
        docai_timeout_seconds=(args.docai_timeout_seconds or settings.docai_timeout_seconds),
        max_assets=max(1, args.max_assets),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"file={payload['file']} doc_id={payload['doc_id']}")
    print(
        f"level={payload['visual_chunk_level']} source={payload['visual_region_source']} "
        f"backend={payload['visual_detector_backend']} pages={payload['page_count']} "
        f"visual_assets={payload['visual_asset_count']} visual_chunks={payload['visual_chunk_count']}"
    )
    if payload["warnings"]:
        print("warnings:")
        for warning in payload["warnings"]:
            print(f"  - {warning}")

    print("\nassets:")
    for row in payload["assets"]:
        region = row["region_label"] or "-"
        region_id = row["region_id"] or "-"
        crop_type = row["crop_type"] or "-"
        proposal_source = row["proposal_source"] or "-"
        proposal_conf = row["proposal_confidence"]
        size = tuple(row["size"])
        summary = row["region_summary"] or "-"
        print(
            f"  - page={row['page']} region={region} region_id={region_id} "
            f"crop_type={crop_type} source={proposal_source} conf={proposal_conf:.2f} "
            f"bbox={tuple(row['bbox'])} size={size} summary={summary}"
        )

    print("\nchunks:")
    for row in payload["chunks"]:
        print(
            f"  - {row['chunk_id']} | page={row['page']} | region={row['region_label'] or '-'} "
            f"| region_id={row['region_id'] or '-'} | crop_type={row['crop_type'] or '-'} "
            f"| source={row['proposal_source'] or '-'} | bbox={tuple(row['bbox'])}"
        )
        print(f"    heading={row['heading_path']}")
        print(f"    preview={row['text_preview']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
