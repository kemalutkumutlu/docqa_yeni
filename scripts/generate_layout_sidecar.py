from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    from src.config import load_settings
    from src.core.ingestion import OCRConfig, ingest_any
    from src.core.multimodal import MultimodalConfig
    from src.core.vlm_extract import VLMConfig
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import load_settings  # type: ignore  # noqa: E402
    from src.core.ingestion import OCRConfig, ingest_any  # type: ignore  # noqa: E402
    from src.core.multimodal import MultimodalConfig  # type: ignore  # noqa: E402
    from src.core.vlm_extract import VLMConfig  # type: ignore  # noqa: E402


def _build_ingest(path: Path, *, level: str):
    settings = load_settings()
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
        assets_dir=settings.data_dir / "layout_sidecar_seed_assets",
        chunk_level=level,
        region_source="heuristic",
    )
    return ingest_any(path, ocr=ocr, display_name=path.name, vlm=vlm, multimodal=multimodal)


def _sidecar_payload(path: Path, *, level: str) -> dict:
    ingest = _build_ingest(path, level=level)
    page_map: dict[str, list[dict]] = {}
    for asset in ingest.visual_assets:
        page_key = str(asset.page_number)
        rows = page_map.setdefault(page_key, [])
        rows.append(
            {
                "label": asset.region_label or f"page-{asset.page_number}",
                "crop_type": asset.crop_type or "page",
                "confidence": round(float(asset.proposal_confidence or 0.0), 3),
                "bbox": [
                    int(asset.bbox_left or 0),
                    int(asset.bbox_top or 0),
                    int(asset.bbox_right or 0),
                    int(asset.bbox_bottom or 0),
                ],
                "summary_text": (asset.region_summary or asset.summary_text or "").strip(),
            }
        )
    return {
        "document": path.name,
        "generator": "heuristic_seed",
        "pages": page_map,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a starter sidecar JSON from current heuristic region proposals.")
    ap.add_argument("path", type=str, help="PDF / PNG / JPG path")
    ap.add_argument("--level", choices=["page", "region"], default="region", help="Seed granularity")
    ap.add_argument("--output", type=str, default="", help="Output sidecar path")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"file not found: {path}")

    payload = _sidecar_payload(path, level=args.level)
    output = Path(args.output).expanduser() if args.output else Path(f"{path.stem}.regions.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
