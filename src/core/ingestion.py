from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
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


def _page_is_visual_heavy(page: "fitz.Page", *, image_area_threshold: float = 0.10) -> bool:
    """
    Returns True if the page contains significant visual content (images, charts, etc.).
    Uses PyMuPDF image bounding boxes to compute image-area / page-area ratio.
    Threshold default 10% — pages with a small logo are not considered visual-heavy.
    """
    try:
        images = page.get_images(full=True)
        if not images:
            return False
        page_area = page.rect.width * page.rect.height
        if page_area <= 0:
            return False
        total_img_area = 0.0
        for img in images:
            xref = img[0]
            for rect in page.get_image_rects(xref):
                total_img_area += rect.width * rect.height
        return (total_img_area / page_area) >= image_area_threshold
    except Exception:  # noqa: BLE001
        return False


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
    source_document_path: Optional[Path] = None,
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
        source_document_path=source_document_path,
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
    source_document_path: Optional[Path] = None,
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
        source_document_path=source_document_path,
    )


def _image_visual_assets(
    img_rgb: Image.Image,
    *,
    ingest_doc_id: str,
    file_name: str,
    summary_text: str,
    multimodal: Optional[MultimodalConfig],
    source_document_path: Optional[Path] = None,
) -> tuple[list[VisualAsset], list[str]]:
    return _build_visual_assets(
        img_rgb,
        ingest_doc_id=ingest_doc_id,
        file_name=file_name,
        page_number=1,
        summary_text=summary_text,
        multimodal=multimodal,
        source_document_path=source_document_path,
    )


# ─── PDF enhancement helpers ──────────────────────────────────────────────────

def _extract_toc_text(pdf: fitz.Document) -> str:
    """Extract bookmark / table-of-contents outline as a structured text prefix.

    Prepending this to page-1 text helps structure.py detect section headings
    it might otherwise miss (e.g. titles without numbered prefixes).
    """
    try:
        toc = pdf.get_toc(simple=False)  # [(level, title, page_no, dest), ...]
        if not toc:
            return ""
        lines = ["=== İÇİNDEKİLER ==="]
        for entry in toc:
            level, title, page_num = entry[0], entry[1], entry[2]
            indent = "  " * max(0, level - 1)
            lines.append(f"{indent}{title} (s.{page_num})")
        return "\n".join(lines) + "\n"
    except Exception:  # noqa: BLE001
        return ""


def _extract_form_fields_text(page: fitz.Page) -> str:
    """Extract filled form-field values from a PDF page (e.g. PDF forms)."""
    try:
        fields = []
        for widget in page.widgets() or []:
            name = (getattr(widget, "field_name", None) or "").strip()
            value = str(getattr(widget, "field_value", None) or "").strip()
            if value:
                fields.append(f"{name}: {value}" if name else value)
        return "\n".join(fields)
    except Exception:  # noqa: BLE001
        return ""


def _extract_native_tables_text(page: fitz.Page) -> str:
    """Use PyMuPDF's native table finder (>= 1.23) to extract tables as markdown.

    Returns empty string when unavailable or when no tables are detected.
    Only called for text-based pages (source == 'pdf_text') to avoid double work.
    """
    try:
        tabs = page.find_tables()
        if not tabs or not tabs.tables:
            return ""
        parts: list[str] = []
        for tab in tabs.tables:
            try:
                md = tab.to_markdown()
                if md and md.strip():
                    parts.append(md.strip())
            except Exception:  # noqa: BLE001
                pass
        return "\n\n".join(parts) if parts else ""
    except AttributeError:
        # find_tables() introduced in PyMuPDF 1.23 – degrade gracefully
        return ""
    except Exception:  # noqa: BLE001
        return ""


def _extract_page_text_ordered(page: fitz.Page) -> str:
    """Extract page text with multi-column layout awareness.

    get_text("text") reads lines in raw PDF order which can interleave columns.
    This function uses get_text("blocks") to detect two-column layouts and reads
    each column top-to-bottom before concatenating.  Falls back to the simple
    get_text("text") path if block extraction fails or layout is single-column.
    """
    try:
        raw_blocks = page.get_text("blocks") or []
        # Each block: (x0, y0, x1, y1, text, block_no, block_type)
        # block_type 0 = text, 1 = image
        text_blocks = [
            (float(x0), float(y0), float(x1), float(y1), str(txt))
            for x0, y0, x1, y1, txt, _bn, bt in raw_blocks
            if int(bt) == 0 and str(txt).strip()
        ]
        if not text_blocks:
            return page.get_text("text") or ""

        page_width = float(page.rect.width) or 1.0
        mid_x = page_width / 2.0

        # Two-column detection: no block spans more than 65% of page width,
        # AND both halves contain meaningful content (≥3 blocks each).
        wide_blocks = [b for b in text_blocks if (b[2] - b[0]) > page_width * 0.65]
        if not wide_blocks:
            left_col = [b for b in text_blocks if b[2] <= mid_x * 1.15]
            right_col = [b for b in text_blocks if b[0] >= mid_x * 0.85]
            if len(left_col) >= 3 and len(right_col) >= 3:
                left_sorted = sorted(left_col, key=lambda b: (round(b[1] / 10), b[0]))
                right_sorted = sorted(right_col, key=lambda b: (round(b[1] / 10), b[0]))
                parts = [b[4].strip() for b in left_sorted] + [b[4].strip() for b in right_sorted]
                return "\n".join(p for p in parts if p)

        # Single-column or mixed: sort by row (y0 bucketed to 10px), then left-to-right
        sorted_blocks = sorted(text_blocks, key=lambda b: (round(b[1] / 10), b[0]))
        return "\n".join(b[4].strip() for b in sorted_blocks if b[4].strip())

    except Exception:  # noqa: BLE001
        return page.get_text("text") or ""


# ─── Docling text-extraction helpers ─────────────────────────────────────────

@dataclass(frozen=True)
class DoclingTextConfig:
    """Configuration for optional Docling-based PDF text extraction.

    When provided to ingest_pdf(), Docling runs on the whole PDF in a subprocess
    (separate venv), extracts per-page text with table markdown, and the result
    is offered as a candidate alongside the native PyMuPDF text.  The best
    candidate (by structure score) is selected per page.
    """
    python_bin: str = ""                                     # path to docling venv python
    model_name: str = "docling-layout-heron-101"
    artifacts_path: Optional[Path] = None
    device: str = "auto"
    do_table_structure: bool = True
    timeout: int = 300                                       # subprocess timeout in seconds


def _docling_text_env(cfg: DoclingTextConfig) -> dict[str, str]:
    """Build subprocess environment for docling_text_runner.py."""
    pass_through = (
        "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TERM",
        "TMPDIR", "TMP", "TEMP", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "ALL_PROXY",
        "XDG_CACHE_HOME", "HF_HOME", "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH",
    )
    env: dict[str, str] = {k: v for k in pass_through if (v := os.environ.get(k))}
    python_bin = (cfg.python_bin or sys.executable).strip() or sys.executable
    env["PATH"] = str(Path(python_bin).resolve().parent) + os.pathsep + os.environ.get("PATH", "")
    env["PYTHONNOUSERSITE"] = "1"
    env["DOCLING_LAYOUT_MODEL"] = cfg.model_name or "docling-layout-heron-101"
    env["DOCLING_DEVICE"] = cfg.device or "auto"
    env["DOCLING_TABLE_STRUCTURE"] = "1" if cfg.do_table_structure else "0"
    if cfg.artifacts_path is not None:
        env["DOCLING_ARTIFACTS_PATH"] = str(cfg.artifacts_path)
    return env


def _docling_extract_page_texts(path: Path, cfg: DoclingTextConfig) -> dict[int, str]:
    """Run docling_text_runner in a subprocess; return {page_no: text}.

    Returns empty dict on any failure so callers can gracefully fall back to
    the PyMuPDF text layer.
    """
    runner_path = Path(__file__).resolve().parents[2] / "scripts" / "docling_text_runner.py"
    if not runner_path.exists():
        return {}

    python_bin = (cfg.python_bin or sys.executable).strip() or sys.executable

    with tempfile.TemporaryDirectory(prefix="docling_text_") as tmp_dir:
        output_path = Path(tmp_dir) / "pages.json"
        cmd = [
            python_bin,
            str(runner_path),
            "--source", str(path),
            "--output", str(output_path),
        ]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=cfg.timeout,
                check=False,
                env=_docling_text_env(cfg),
            )
        except FileNotFoundError:
            return {}
        except Exception:  # noqa: BLE001
            return {}

        if completed.returncode != 0 or not output_path.exists():
            return {}

        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}

        pages_raw = payload.get("pages", {}) if isinstance(payload, dict) else {}
        return {int(k): str(v) for k, v in pages_raw.items() if str(v).strip()}


# ─────────────────────────────────────────────────────────────────────────────


def ingest_pdf(
    path: Path,
    ocr: OCRConfig,
    display_name: Optional[str] = None,
    vlm: Optional[VLMConfig] = None,
    table_config: Optional[TableStructureConfig] = None,
    multimodal: Optional[MultimodalConfig] = None,
    docling_text: Optional[DoclingTextConfig] = None,
) -> IngestResult:
    """
    Extract page-bounded text from a PDF.

    - Prefer PDF text layer (with multi-column and table-aware extraction).
    - If a page has too little text and OCR is enabled, render to image and OCR.
    - Prepends TOC/bookmarks to page-1 text when available.
    - Supplements text-layer pages with native table markdown and form fields.
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

    # ── Open PDF (handle encrypted / corrupted files explicitly) ────────────
    try:
        pdf = fitz.open(str(path))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"PDF dosyasi acilamadi ({file_name}): {e}") from e

    if pdf.needs_pass:
        pdf.close()
        raise ValueError(f"PDF sifreli/kilitli, parola gerekiyor: {file_name}")

    # ── Extract bookmark / TOC (prepended to page-1 text) ───────────────────
    toc_prefix = _extract_toc_text(pdf)

    # ── Optional Docling text extraction (runs once for whole PDF) ───────────
    docling_page_texts: dict[int, str] = {}
    if docling_text is not None:
        try:
            docling_page_texts = _docling_extract_page_texts(path, docling_text)
            if not docling_page_texts:
                warnings.append("Docling PDF metin çıkarma sonuç vermedi; PyMuPDF kullanılıyor.")
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Docling PDF metin çıkarma başarısız: {e}")

    try:
        vlm_pages_used = 0
        for i in range(pdf.page_count):
            page = pdf.load_page(i)
            page_no = i + 1

            pdf_text = _extract_page_text_ordered(page)
            pdf_text_raw = normalize_whitespace(pdf_text)
            pdf_text_norm = _finalize_extracted_text(pdf_text, source="pdf_text")
            text_norm = pdf_text_norm

            # ── Docling text candidate ────────────────────────────────────────
            source = "pdf_text"
            if docling_page_texts.get(page_no):
                docling_raw = docling_page_texts[page_no]
                docling_norm = _finalize_extracted_text(docling_raw, source="docling_text")
                if docling_norm:
                    text_norm, source = _pick_best_candidate(
                        [(pdf_text_norm, "pdf_text"), (docling_norm, "docling_text")]
                    )

            # Heuristic: if text layer is missing/too small, try OCR.
            # Check quality of the best candidate so far (PyMuPDF or Docling).
            ocr_text_norm = ""
            # NOTE: some PDFs have a "text layer" that is present but unusable (broken layout,
            # single-token-per-line, etc.). Treat those pages as OCR candidates too.
            should_try_ocr = ocr.enabled and (
                len(text_norm.strip()) < 40 or _text_quality_low(text_norm)
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
                    should_vlm = (
                        vlm.mode == "force"
                        or (vlm.mode == "auto" and _text_quality_low(text_norm))
                        or (vlm.mode == "smart" and (
                            _text_quality_low(text_norm) or _page_is_visual_heavy(page)
                        ))
                    )
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

            # ── Supplement text-layer pages with native table markdown & form fields ──
            # Only when PyMuPDF won (Docling already handles tables natively).
            if source == "pdf_text":
                native_tables = _extract_native_tables_text(page)
                if native_tables:
                    text_norm = text_norm + "\n\n" + native_tables
                form_fields = _extract_form_fields_text(page)
                if form_fields:
                    text_norm = text_norm + "\n" + form_fields

            # ── Prepend TOC/bookmarks to first page ──────────────────────────────
            if page_no == 1 and toc_prefix:
                text_norm = toc_prefix + "\n" + text_norm

            pages.append(
                PageText(
                    doc_id=doc_id,
                    file_name=file_name,
                    page_number=page_no,
                    text=text_norm,
                    source=source,  # type: ignore[arg-type]
                )
            )
            # Processing smart mode: skip visual assets for non-visual pages.
            _effective_multimodal = multimodal
            if multimodal and getattr(multimodal, "smart_mode", False) and not _page_is_visual_heavy(page):
                _effective_multimodal = None
            assets, asset_warnings = _pdf_page_visual_assets(
                page,
                ingest_doc_id=doc_id,
                file_name=file_name,
                page_number=page_no,
                summary_text=text_norm,
                multimodal=_effective_multimodal,
                source_document_path=path,
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
        # Table smart mode: skip external extraction for pages with native PDF text layer
        # (PyMuPDF find_tables already handled them). Only run on OCR/VLM pages.
        table_assets = visual_assets
        if getattr(table_config, "smart", False):
            ocr_pages = {p.page_number for p in pages if p.source not in ("pdf_text", "docling_text")}
            table_assets = [a for a in visual_assets if a.page_number in ocr_pages]
        structured_tables, table_warnings = extract_tables_from_assets(table_assets, cfg=table_config)
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
            source_document_path=path,
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
    docling_text: Optional[DoclingTextConfig] = None,
) -> IngestResult:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return ingest_pdf(
            path,
            ocr=ocr,
            display_name=display_name,
            vlm=vlm,
            table_config=table_config,
            multimodal=multimodal,
            docling_text=docling_text,
        )
    if suffix in (".png", ".jpg", ".jpeg"):
        return ingest_image(path, ocr=ocr, display_name=display_name, vlm=vlm, table_config=table_config, multimodal=multimodal)
    raise ValueError(f"Unsupported file type: {suffix}")

