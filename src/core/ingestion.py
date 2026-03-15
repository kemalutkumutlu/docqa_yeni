from __future__ import annotations

from pathlib import Path
from typing import Optional
import re
import fitz  # PyMuPDF
from PIL import Image
from PIL import ImageFilter, ImageOps

from .content_normalization import normalize_extracted_text
from .layout_regions import plan_regions
from .models import IngestResult, PageText, VisualAsset
from .multimodal import MultimodalConfig
from .ocr_backend import OCRConfig, configure_ocr_backend, ocr_image_text
from .table_structure import TableStructureConfig, extract_tables_from_assets
from .utils import normalize_whitespace, sha256_file
from .vlm_extract import VLMConfig, extract_text_from_image


def _text_quality_low(text: str) -> bool:
    """
    Document-agnostic heuristic to decide whether extracted text is low quality.
    """
    t = text.strip()
    if len(t) < 120:
        return True
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return True
    # Many single-token lines can indicate broken layout extraction.
    single_token_lines = sum(1 for ln in lines if len(ln.split()) <= 1 and len(ln) < 40)
    if single_token_lines / max(1, len(lines)) > 0.55:
        return True
    return False


_RE_ALPHA_NUM = re.compile(r"^(?P<alpha>[A-Z])\.(?P<num>\d+(?:\.\d+)*)\s+.+", re.IGNORECASE)
_RE_NUM_DOT = re.compile(r"^(?P<num>\d+(?:\.\d+)*)\.\s+.+")
_RE_NUM_DASH = re.compile(r"^(?P<num>\d+(?:\.\d+)*)\s*[-–—]\s*(?P<title>.+?)\s*$")


def _count_heading_like_lines(text: str) -> int:
    """
    Count numbered heading-like lines (document-agnostic).
    Used only to choose between multiple extraction candidates (pdf_text/ocr/vlm).
    """
    hits = 0
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        if _RE_ALPHA_NUM.match(s) or _RE_NUM_DOT.match(s):
            hits += 1
            continue
        m = _RE_NUM_DASH.match(s)
        if m:
            title = (m.group("title") or "").strip()
            # Guard against date/range artifacts.
            if title[:1].isdigit():
                continue
            hits += 1
    return hits


def _single_token_ratio(text: str) -> float:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return 1.0
    single = sum(1 for ln in lines if len(ln.split()) <= 1 and len(ln) < 40)
    return single / max(1, len(lines))


def _score_for_structure(text: str) -> float:
    """
    Prefer text that preserves document structure (headings, sane lines).
    """
    t = (text or "").strip()
    if not t:
        return -1e9
    heading_hits = _count_heading_like_lines(t)
    ratio_single = _single_token_ratio(t)
    # Length helps, but heading preservation matters more for sectioning.
    length = min(len(t), 12000) / 2000.0
    return 6.0 * heading_hits + 1.0 * length - 8.0 * ratio_single


def _pick_best_candidate(cands: list[tuple[str, str]]) -> tuple[str, str]:
    """
    Pick best (text, source) candidate by structure score.
    """
    best_text, best_source = cands[0]
    best_score = _score_for_structure(best_text)
    for txt, src in cands[1:]:
        sc = _score_for_structure(txt)
        if sc > best_score + 0.5:
            best_text, best_source, best_score = txt, src, sc
    return best_text, best_source


def _finalize_extracted_text(text: str, *, source: str) -> str:
    text_norm = normalize_whitespace(text)
    if not text_norm:
        return ""
    return normalize_extracted_text(text_norm, source=source)


def _safe_exif_transpose(img: Image.Image) -> Image.Image:
    """
    Normalize image orientation using EXIF if present.
    This is critical for phone photos / scanned images.
    """
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img


def _maybe_upscale_for_ocr(img: Image.Image, min_short_side: int = 1200, max_long_side: int = 3200) -> Image.Image:
    """
    Upscale small images to improve OCR quality, but cap size to avoid huge latency.
    Pillow-only (no OpenCV dependency).
    """
    try:
        w, h = img.size
        short = min(w, h)
        long = max(w, h)
        if short >= min_short_side:
            return img
        scale = min_short_side / max(1, short)
        # Cap long side to avoid extreme upscales.
        if long * scale > max_long_side:
            scale = max_long_side / max(1, long)
        if scale <= 1.01:
            return img
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        return img.resize((nw, nh), resample=Image.Resampling.LANCZOS)
    except Exception:
        return img


def _preprocess_variants_for_ocr(img_rgb: Image.Image) -> list[Image.Image]:
    """
    Build a small set of OCR-friendly variants.
    We keep the original in the candidate list to avoid regressions.
    """
    variants: list[Image.Image] = []
    base = img_rgb
    variants.append(base)

    up = _maybe_upscale_for_ocr(base)
    if up is not base:
        variants.append(up)

    try:
        g = ImageOps.grayscale(up)
        g = ImageOps.autocontrast(g)
        variants.append(g)
        # Light sharpening can help thin fonts.
        variants.append(g.filter(ImageFilter.UnsharpMask(radius=1.6, percent=160, threshold=3)))
        # Simple binarization (global threshold). Keep conservative to avoid wiping faint text.
        thr = 190
        bw = g.point(lambda p: 255 if p > thr else 0, mode="1")
        variants.append(bw.convert("L"))
    except Exception:
        pass

    # Deduplicate by size+mode to avoid repeated OCR work.
    uniq: list[Image.Image] = []
    seen: set[tuple[int, int, str]] = set()
    for im in variants:
        key = (im.size[0], im.size[1], im.mode)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(im)
    return uniq


def _save_visual_asset_image(img: Image.Image, *, out_path: Path) -> tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return img.size


def _region_summary_text(summary: str, *, limit: int = 120) -> str:
    clean = re.sub(r"\s+", " ", (summary or "").strip())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)] + "..."


def _build_visual_assets(
    img: Image.Image,
    *,
    ingest_doc_id: str,
    file_name: str,
    page_number: int,
    summary_text: str,
    multimodal: Optional[MultimodalConfig],
) -> tuple[list[VisualAsset], list[str]]:
    if not multimodal or not multimodal.enabled or multimodal.assets_dir is None:
        return [], []

    chunk_level = (multimodal.chunk_level or "page").strip().lower()
    region_source = (multimodal.region_source or "heuristic").strip().lower()
    proposals, warnings = plan_regions(
        img,
        page_number=page_number,
        level="region" if chunk_level == "region" else "page",
        source="detector" if region_source == "detector" else "heuristic",
        summary_text=summary_text,
        document_name=file_name,
        detector_backend=(multimodal.detector_backend or "none"),
        detector_dir=multimodal.detector_dir,
        docai_project_id=(multimodal.docai_project_id or ""),
        docai_location=(multimodal.docai_location or "us"),
        docai_processor_id=(multimodal.docai_processor_id or ""),
        docai_processor_version=(multimodal.docai_processor_version or "pretrained-layout-parser-v1.6-pro-2025-12-01"),
        docai_timeout_seconds=int(multimodal.docai_timeout_seconds or 120),
        docling_python_bin=(multimodal.docling_python_bin or ""),
        docling_layout_model=(multimodal.docling_layout_model or "docling-layout-heron-101"),
        docling_artifacts_path=multimodal.docling_artifacts_path,
        docling_device=(multimodal.docling_device or "auto"),
    )
    assets: list[VisualAsset] = []
    for proposal in proposals:
        if proposal.region_count > 1:
            suffix = f"_r{proposal.region_index:02d}"
        else:
            suffix = ""
        crop = img.crop((proposal.bbox_left, proposal.bbox_top, proposal.bbox_right, proposal.bbox_bottom))
        out_path = multimodal.assets_dir / ingest_doc_id / f"page_{page_number:04d}{suffix}.png"
        width, height = _save_visual_asset_image(crop, out_path=out_path)
        summary = (proposal.summary_text or summary_text or "").strip()
        assets.append(
            VisualAsset(
                doc_id=ingest_doc_id,
                file_name=file_name,
                page_number=page_number,
                image_path=str(out_path),
                summary_text=summary,
                mime_type="image/png",
                width=width,
                height=height,
                region_label=proposal.region_label,
                region_index=proposal.region_index,
                region_count=proposal.region_count,
                region_id=proposal.region_id,
                crop_type=proposal.crop_type,
                region_summary=_region_summary_text(summary),
                proposal_source=proposal.proposal_source,
                proposal_confidence=proposal.proposal_confidence,
                bbox_left=proposal.bbox_left,
                bbox_top=proposal.bbox_top,
                bbox_right=proposal.bbox_right,
                bbox_bottom=proposal.bbox_bottom,
                bbox_left_norm=proposal.bbox_left_norm,
                bbox_top_norm=proposal.bbox_top_norm,
                bbox_right_norm=proposal.bbox_right_norm,
                bbox_bottom_norm=proposal.bbox_bottom_norm,
            )
        )
    return assets, warnings


def _pdf_page_visual_assets(
    page: fitz.Page,
    *,
    ingest_doc_id: str,
    file_name: str,
    page_number: int,
    summary_text: str,
    multimodal: Optional[MultimodalConfig],
) -> tuple[list[VisualAsset], list[str]]:
    pix = page.get_pixmap(dpi=160)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return _build_visual_assets(
        img,
        ingest_doc_id=ingest_doc_id,
        file_name=file_name,
        page_number=page_number,
        summary_text=summary_text,
        multimodal=multimodal,
    )


def _image_visual_assets(
    img_rgb: Image.Image,
    *,
    ingest_doc_id: str,
    file_name: str,
    summary_text: str,
    multimodal: Optional[MultimodalConfig],
) -> tuple[list[VisualAsset], list[str]]:
    return _build_visual_assets(
        img_rgb,
        ingest_doc_id=ingest_doc_id,
        file_name=file_name,
        page_number=1,
        summary_text=summary_text,
        multimodal=multimodal,
    )


def ingest_pdf(
    path: Path,
    ocr: OCRConfig,
    display_name: Optional[str] = None,
    vlm: Optional[VLMConfig] = None,
    table_config: Optional[TableStructureConfig] = None,
    multimodal: Optional[MultimodalConfig] = None,
) -> IngestResult:
    """
    Extract page-bounded text from a PDF.

    - Prefer PDF text layer.
    - If a page has too little text and OCR is enabled, render to image and OCR.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    configure_ocr_backend(ocr)

    doc_id = sha256_file(path)
    # Chainlit uploads can be stored under a temporary UUID-like filename.
    # `display_name` lets callers preserve the original user-facing filename for citations.
    file_name = display_name or path.name
    warnings: list[str] = []
    pages: list[PageText] = []
    visual_assets: list[VisualAsset] = []
    structured_tables = []

    pdf = fitz.open(str(path))
    try:
        vlm_pages_used = 0
        for i in range(pdf.page_count):
            page = pdf.load_page(i)
            page_no = i + 1

            pdf_text = page.get_text("text") or ""
            pdf_text_raw = normalize_whitespace(pdf_text)
            pdf_text_norm = _finalize_extracted_text(pdf_text, source="pdf_text")
            text_norm = pdf_text_norm

            # Heuristic: if text layer is missing/too small, try OCR.
            source = "pdf_text"
            ocr_text_norm = ""
            # NOTE: some PDFs have a "text layer" that is present but unusable (broken layout,
            # single-token-per-line, etc.). Treat those pages as OCR candidates too.
            should_try_ocr = ocr.enabled and (
                len(pdf_text_raw) < 40 or _text_quality_low(pdf_text_raw)
            )
            if should_try_ocr:
                try:
                    pix = page.get_pixmap(dpi=200)  # good speed/quality tradeoff
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_result = ocr_image_text(
                        img,
                        cfg=ocr,
                        document_name=file_name,
                        page_number=page_no,
                        image_kind="pdf_page",
                    )
                    if ocr_result.warnings:
                        for warning in ocr_result.warnings:
                            if warning not in warnings:
                                warnings.append(warning)
                    ocr_text_norm = _finalize_extracted_text(ocr_result.text, source=ocr_result.source)
                    # Choose between pdf_text and ocr by structure score (not just length).
                    text_norm, source = _pick_best_candidate(
                        [(pdf_text_norm, "pdf_text"), (ocr_text_norm, ocr_result.source)]
                    )
                except Exception as e:  # noqa: BLE001
                    warnings.append(f"OCR failed on page {page_no}: {e}")
                    source = "pdf_text"

            # VLM fallback (extract-only) for low-quality pages.
            #
            # IMPORTANT: In local (Ollama) mode, VLM does not require an API key.
            # In Gemini mode, we require `api_key` to be present.
            if vlm and vlm.mode != "off" and vlm_pages_used < vlm.max_pages and (
                getattr(vlm, "provider", "gemini") == "local" or bool(getattr(vlm, "api_key", ""))
            ):
                try:
                    should_vlm = vlm.mode == "force" or (vlm.mode == "auto" and _text_quality_low(text_norm))
                    if should_vlm:
                        pix = page.get_pixmap(dpi=200)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        vlm_text = extract_text_from_image(img, cfg=vlm)
                        vlm_text_norm = _finalize_extracted_text(vlm_text, source="vlm")
                        # Dual-quality selection: keep whichever preserves structure better.
                        text_norm, source = _pick_best_candidate([(text_norm, source), (vlm_text_norm, "vlm")])
                        vlm_pages_used += 1
                except Exception as e:  # noqa: BLE001
                    warnings.append(f"VLM failed on page {page_no}: {e}")

            pages.append(
                PageText(
                    doc_id=doc_id,
                    file_name=file_name,
                    page_number=page_no,
                    text=text_norm,
                    source=source,  # type: ignore[arg-type]
                )
            )
            assets, asset_warnings = _pdf_page_visual_assets(
                page,
                ingest_doc_id=doc_id,
                file_name=file_name,
                page_number=page_no,
                summary_text=text_norm,
                multimodal=multimodal,
            )
            if asset_warnings:
                for warning in asset_warnings:
                    if warning not in warnings:
                        warnings.append(warning)
            if assets:
                visual_assets.extend(assets)
    finally:
        pdf.close()
    if table_config is not None:
        structured_tables, table_warnings = extract_tables_from_assets(visual_assets, cfg=table_config)
        for warning in table_warnings:
            if warning not in warnings:
                warnings.append(warning)

    return IngestResult(
        doc_id=doc_id,
        file_name=file_name,
        pages=pages,
        visual_assets=visual_assets,
        structured_tables=structured_tables,
        warnings=warnings,
    )


def ingest_image(
    path: Path,
    ocr: OCRConfig,
    display_name: Optional[str] = None,
    vlm: Optional[VLMConfig] = None,
    table_config: Optional[TableStructureConfig] = None,
    multimodal: Optional[MultimodalConfig] = None,
) -> IngestResult:
    """OCR a single image file into one 'page'."""
    if not path.exists():
        raise FileNotFoundError(str(path))

    configure_ocr_backend(ocr)

    doc_id = sha256_file(path)
    file_name = display_name or path.name
    warnings: list[str] = []
    visual_assets: list[VisualAsset] = []
    structured_tables = []

    text_norm = ""
    # Load image once; reuse for OCR/VLM candidates.
    try:
        img_rgb = _safe_exif_transpose(Image.open(path)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Image open failed: {e}")
        img_rgb = None

    # Candidate list: (text, source)
    cands: list[tuple[str, str]] = []

    if vlm and vlm.mode in ("force", "auto") and img_rgb is not None and (
        getattr(vlm, "provider", "gemini") == "local" or bool(getattr(vlm, "api_key", ""))
    ):
        try:
            vlm_text = extract_text_from_image(img_rgb, cfg=vlm)
            vlm_text_norm = _finalize_extracted_text(vlm_text, source="vlm")
            cands.append((vlm_text_norm, "vlm"))
        except Exception as e:  # noqa: BLE001
            warnings.append(f"VLM image extract failed: {e}")
            # fall through; OCR candidates may still succeed

    if ocr.enabled and img_rgb is not None:
        try:
            ocr_variants = [img_rgb]
            if (ocr.backend or "tesseract_legacy").strip().lower() == "tesseract_legacy":
                # Multi-pass OCR only for the legacy Tesseract path.
                ocr_variants = _preprocess_variants_for_ocr(img_rgb)[:4]
            for im in ocr_variants:
                ocr_result = ocr_image_text(
                    im,
                    cfg=ocr,
                    document_name=file_name,
                    page_number=1,
                    image_kind="image",
                )
                if ocr_result.warnings:
                    for warning in ocr_result.warnings:
                        if warning not in warnings:
                            warnings.append(warning)
                ocr_text_norm = _finalize_extracted_text(ocr_result.text, source=ocr_result.source)
                if ocr_text_norm:
                    cands.append((ocr_text_norm, ocr_result.source))
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Image OCR failed: {e}")

    if not cands:
        if img_rgb is None:
            warnings.append("Image ingestion produced empty text (failed to open image).")
        elif not ocr.enabled and not (vlm and (getattr(vlm, "provider", "gemini") == "local" or bool(vlm.api_key))):
            warnings.append("OCR disabled and VLM unavailable; image ingestion produced empty text.")
        else:
            warnings.append("Image ingestion produced empty text.")
        text_norm, source = "", "image_ocr"
    else:
        # Prefer whichever preserves structure best. This is safe because the
        # original OCR/VLM candidates are still present; switching requires score gain.
        text_norm, source = _pick_best_candidate(cands)

    pages = [
        PageText(
            doc_id=doc_id,
            file_name=file_name,
            page_number=1,
            text=text_norm,
            source=source,  # type: ignore[arg-type]
        )
    ]
    if img_rgb is not None:
        assets, asset_warnings = _image_visual_assets(
            img_rgb,
            ingest_doc_id=doc_id,
            file_name=file_name,
            summary_text=text_norm,
            multimodal=multimodal,
        )
        if asset_warnings:
            for warning in asset_warnings:
                if warning not in warnings:
                    warnings.append(warning)
        if assets:
            visual_assets.extend(assets)
    if table_config is not None:
        structured_tables, table_warnings = extract_tables_from_assets(visual_assets, cfg=table_config)
        for warning in table_warnings:
            if warning not in warnings:
                warnings.append(warning)
    return IngestResult(
        doc_id=doc_id,
        file_name=file_name,
        pages=pages,
        visual_assets=visual_assets,
        structured_tables=structured_tables,
        warnings=warnings,
    )


def ingest_any(
    path: Path,
    ocr: OCRConfig,
    display_name: Optional[str] = None,
    vlm: Optional[VLMConfig] = None,
    table_config: Optional[TableStructureConfig] = None,
    multimodal: Optional[MultimodalConfig] = None,
) -> IngestResult:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return ingest_pdf(path, ocr=ocr, display_name=display_name, vlm=vlm, table_config=table_config, multimodal=multimodal)
    if suffix in (".png", ".jpg", ".jpeg"):
        return ingest_image(path, ocr=ocr, display_name=display_name, vlm=vlm, table_config=table_config, multimodal=multimodal)
    raise ValueError(f"Unsupported file type: {suffix}")

