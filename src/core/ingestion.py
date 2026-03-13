from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import os
import re
import fitz  # PyMuPDF
from PIL import Image
from PIL import ImageFilter, ImageOps

from .content_normalization import normalize_extracted_text
from .models import IngestResult, PageText, VisualAsset
from .multimodal import MultimodalConfig
from .utils import normalize_whitespace, sha256_file
from .vlm_extract import VLMConfig, extract_text_from_image


@dataclass(frozen=True)
class OCRConfig:
    enabled: bool = True
    lang: str = "tur+eng"
    tesseract_cmd: Optional[str] = None
    tessdata_prefix: Optional[str] = None
    # Optional passthrough for pytesseract `config=` (e.g. "--psm 6 --oem 3").
    # Keep default None to preserve existing behavior.
    tesseract_config: Optional[str] = None


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


def _configure_tesseract(tesseract_cmd: Optional[str], tessdata_prefix: Optional[str]) -> None:
    if not tesseract_cmd:
        # Even if cmd isn't set, allow callers to set TESSDATA_PREFIX for system installs.
        if tessdata_prefix:
            os.environ["TESSDATA_PREFIX"] = tessdata_prefix
        return
    try:
        import pytesseract  # noqa: WPS433

        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        if tessdata_prefix:
            os.environ["TESSDATA_PREFIX"] = tessdata_prefix
    except Exception:
        # If pytesseract isn't installed or fails, leave as-is; caller will get warning.
        return


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


def _ocr_image_text(img: Image.Image, lang: str, cfg: OCRConfig) -> str:
    """
    OCR one image with pytesseract, returning raw text (may be empty).
    """
    import pytesseract  # noqa: WPS433

    # pytesseract supports L/RGB images; keep as-is.
    config = (cfg.tesseract_config or "").strip()
    return pytesseract.image_to_string(img, lang=lang, config=config) or ""


def _save_visual_asset_image(img: Image.Image, *, out_path: Path) -> tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return img.size


def _pdf_page_visual_asset(
    page: fitz.Page,
    *,
    ingest_doc_id: str,
    file_name: str,
    page_number: int,
    summary_text: str,
    multimodal: Optional[MultimodalConfig],
) -> Optional[VisualAsset]:
    if not multimodal or not multimodal.enabled or multimodal.assets_dir is None:
        return None
    pix = page.get_pixmap(dpi=160)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    out_path = multimodal.assets_dir / ingest_doc_id / f"page_{page_number:04d}.png"
    width, height = _save_visual_asset_image(img, out_path=out_path)
    return VisualAsset(
        doc_id=ingest_doc_id,
        file_name=file_name,
        page_number=page_number,
        image_path=str(out_path),
        summary_text=summary_text,
        mime_type="image/png",
        width=width,
        height=height,
    )


def _image_visual_asset(
    img_rgb: Image.Image,
    *,
    ingest_doc_id: str,
    file_name: str,
    summary_text: str,
    multimodal: Optional[MultimodalConfig],
) -> Optional[VisualAsset]:
    if not multimodal or not multimodal.enabled or multimodal.assets_dir is None:
        return None
    out_path = multimodal.assets_dir / ingest_doc_id / f"page_{1:04d}.png"
    width, height = _save_visual_asset_image(img_rgb, out_path=out_path)
    return VisualAsset(
        doc_id=ingest_doc_id,
        file_name=file_name,
        page_number=1,
        image_path=str(out_path),
        summary_text=summary_text,
        mime_type="image/png",
        width=width,
        height=height,
    )


def ingest_pdf(
    path: Path,
    ocr: OCRConfig,
    display_name: Optional[str] = None,
    vlm: Optional[VLMConfig] = None,
    multimodal: Optional[MultimodalConfig] = None,
) -> IngestResult:
    """
    Extract page-bounded text from a PDF.

    - Prefer PDF text layer.
    - If a page has too little text and OCR is enabled, render to image and OCR.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    _configure_tesseract(ocr.tesseract_cmd, ocr.tessdata_prefix)

    doc_id = sha256_file(path)
    # Chainlit uploads can be stored under a temporary UUID-like filename.
    # `display_name` lets callers preserve the original user-facing filename for citations.
    file_name = display_name or path.name
    warnings: list[str] = []
    pages: list[PageText] = []
    visual_assets: list[VisualAsset] = []

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
                    import pytesseract  # noqa: WPS433

                    pix = page.get_pixmap(dpi=200)  # good speed/quality tradeoff
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img, lang=ocr.lang) or ""
                    ocr_text_norm = _finalize_extracted_text(ocr_text, source="ocr")
                    # Choose between pdf_text and ocr by structure score (not just length).
                    text_norm, source = _pick_best_candidate(
                        [(pdf_text_norm, "pdf_text"), (ocr_text_norm, "ocr")]
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
            asset = _pdf_page_visual_asset(
                page,
                ingest_doc_id=doc_id,
                file_name=file_name,
                page_number=page_no,
                summary_text=text_norm,
                multimodal=multimodal,
            )
            if asset is not None:
                visual_assets.append(asset)
    finally:
        pdf.close()

    return IngestResult(
        doc_id=doc_id,
        file_name=file_name,
        pages=pages,
        visual_assets=visual_assets,
        warnings=warnings,
    )


def ingest_image(
    path: Path,
    ocr: OCRConfig,
    display_name: Optional[str] = None,
    vlm: Optional[VLMConfig] = None,
    multimodal: Optional[MultimodalConfig] = None,
) -> IngestResult:
    """OCR a single image file into one 'page'."""
    if not path.exists():
        raise FileNotFoundError(str(path))

    _configure_tesseract(ocr.tesseract_cmd, ocr.tessdata_prefix)

    doc_id = sha256_file(path)
    file_name = display_name or path.name
    warnings: list[str] = []
    visual_assets: list[VisualAsset] = []

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
            # Multi-pass OCR: try original + a few safe preprocess variants, then pick
            # the best by structure score (same idea as dual-quality in PDFs).
            ocr_variants = _preprocess_variants_for_ocr(img_rgb)
            # Cap work: keep only first few (ordered by cheap->expensive).
            ocr_variants = ocr_variants[:4]
            for im in ocr_variants:
                ocr_text = _ocr_image_text(im, lang=ocr.lang, cfg=ocr)
                ocr_text_norm = _finalize_extracted_text(ocr_text, source="image_ocr")
                if ocr_text_norm:
                    cands.append((ocr_text_norm, "image_ocr"))
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
        asset = _image_visual_asset(
            img_rgb,
            ingest_doc_id=doc_id,
            file_name=file_name,
            summary_text=text_norm,
            multimodal=multimodal,
        )
        if asset is not None:
            visual_assets.append(asset)
    return IngestResult(
        doc_id=doc_id,
        file_name=file_name,
        pages=pages,
        visual_assets=visual_assets,
        warnings=warnings,
    )


def ingest_any(
    path: Path,
    ocr: OCRConfig,
    display_name: Optional[str] = None,
    vlm: Optional[VLMConfig] = None,
    multimodal: Optional[MultimodalConfig] = None,
) -> IngestResult:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return ingest_pdf(path, ocr=ocr, display_name=display_name, vlm=vlm, multimodal=multimodal)
    if suffix in (".png", ".jpg", ".jpeg"):
        return ingest_image(path, ocr=ocr, display_name=display_name, vlm=vlm, multimodal=multimodal)
    raise ValueError(f"Unsupported file type: {suffix}")

