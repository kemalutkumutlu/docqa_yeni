from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

from PIL import Image


@dataclass(frozen=True)
class VLMConfig:
    """
    Vision-Language extraction config.

    mode:
      - "off": never call VLM
      - "auto": call VLM only when text quality is low
      - "smart": call VLM when text quality is low OR page has visual content (images/charts)
      - "force": always call VLM on every page (expensive)

    provider:
      - "gemini": use Google Gemini API (default, backward compatible)
      - "local": use local Ollama vision model
    """

    api_key: str
    model: str = "gemini-2.0-flash"
    mode: str = "auto"  # off | auto | force
    max_pages: int = 25  # safety cap per document
    provider: str = "gemini"  # "gemini" | "local"
    # Ollama settings (only used when provider == "local")
    ollama_base_url: str = "http://localhost:11434"
    ollama_vlm_model: str = "llava:7b"
    ollama_timeout: int = 120


_EXTRACT_ONLY_PROMPT = """\
You are a precise document extraction engine. Your only job is to faithfully extract \
every piece of visible text from the image, preserving its structure and associations.

## Core Rule
Extract ONLY what is visually present. Do NOT add, infer, summarize, or translate.

## Reading Order
- Single-column content and tables: LEFT → RIGHT, then TOP → BOTTOM.
- Multi-column document layout (sidebars, CV layout, magazine-style):
  Read each COLUMN fully top-to-bottom before moving to the next column.
  Exception: compact label-value tables where rows clearly pair a name/label on the left
  with a value/company on the right — read those rows LEFT → RIGHT.

## Layout Handling

### Headings & Sections
- Numbered headings (e.g. "1.", "2.3", "A.1"): keep on their own line, preserve the number.
- Bold or large-text headings: keep on their own line.
- Section hierarchy must be preserved exactly as seen.

### Multi-Column Layouts (CVs, articles, brochures)
- Independent sidebar sections (e.g. Education sidebar, Work Experience sidebar, Skills sidebar):
  Each section heading and its contents must be output separately, top-to-bottom within
  that column. Do NOT merge sidebar section headings with the opposite column's content.
- Compact label-value tables or lists (e.g. reference lists, property lists):
  Output each row as: `Label: Value` (on the same line, colon-separated).
  Example: if "Oğuzcan Özdemir" is on the left and "ASELSAN" is on the right,
  output: `Oğuzcan Özdemir: ASELSAN`

### Tables
- Represent as Markdown tables: `| col1 | col2 | col3 |`
- CRITICAL — merged cells: if a cell visually spans multiple columns, write its
  value ONCE in the first cell, leave the others empty: `| value | | |`
- Do NOT repeat the merged cell value across every column it spans.
- Row-column association must be exact: each value must appear in its correct cell.

### Lists
- Numbered lists: preserve the number prefix exactly (1., 2., a), b), etc.)
- Bullet lists: use `- ` prefix.
- Nested lists: indent with 2 spaces per level.

### Forms & Checkboxes
- Checkbox checked: `[x] Label`
- Checkbox unchecked: `[ ] Label`
- Text field: `Field name: field value` (if value is filled) or `Field name: ___` (if empty)

### Key-Value Pairs (specs, metadata, reference lists)
- If text is arranged as label + value pairs (even without explicit table borders),
  output as: `Label: Value`

### Headers & Footers
- Include page headers and footers as-is, on their own lines.

### Unreadable Regions
- If a region is genuinely unreadable, omit it entirely. Do NOT guess.

### Language
- Do NOT translate. Output text in the exact language it appears.
- Preserve special characters (Turkish: ğ ü ş ı ö ç Ğ Ü Ş İ Ö Ç).
"""


_OCR_GROUNDING_PROMPT = """\

---
SUPPLEMENTARY OCR REFERENCE (character accuracy only):
A separate OCR engine has extracted the raw text below. Its CHARACTER and NUMBER \
recognition is highly accurate, but its reading ORDER and LAYOUT may be wrong \
(e.g. it may read columns top-to-bottom instead of left-to-right).

Instructions:
- Use the IMAGE as the sole authority for layout, reading order, and associations.
- Use this OCR text ONLY to verify exact spellings, numbers, dates, and punctuation.
- If the OCR text contradicts the image layout, ALWAYS follow the image layout.
- Do NOT copy the OCR text order into your output.

[OCR TEXT START]
{ocr_context}
[OCR TEXT END]
"""


def _gemini_helpers():
    from google.genai import types

    from .gemini_client import build_gemini_client, gemini_model_candidates, is_model_not_found_error

    return types, build_gemini_client, gemini_model_candidates, is_model_not_found_error


def extract_text_from_image(image: Image.Image, cfg: VLMConfig, ocr_context: Optional[str] = None) -> str:
    """
    Extract text from an image using a multimodal model (extract-only).
    Dispatches to Gemini API or local Ollama based on cfg.provider.
    Returns extracted text (may be empty).
    """
    if cfg.provider == "local":
        return _extract_via_ollama(image, cfg, ocr_context)
    return _extract_via_gemini(image, cfg, ocr_context)


def _extract_via_gemini(image: Image.Image, cfg: VLMConfig, ocr_context: Optional[str] = None) -> str:
    """Gemini API path (existing behavior)."""
    types, build_gemini_client, gemini_model_candidates, is_model_not_found_error = _gemini_helpers()
    
    prompt = _EXTRACT_ONLY_PROMPT
    if ocr_context and ocr_context.strip():
        # Limit the context size defensively (e.g. 5000 chars) to prevent prompt blowup
        prompt += _OCR_GROUNDING_PROMPT.format(ocr_context=ocr_context.strip()[:5000])
    # Encode image as PNG bytes
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    last_model_error: Exception | None = None
    for model_name in gemini_model_candidates(cfg.model):
        try:
            client = build_gemini_client(cfg.api_key, model_name=model_name)
            resp = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=8192,
                ),
            )
            return (resp.text or "").strip()
        except Exception as exc:
            last_model_error = exc
            if not is_model_not_found_error(exc):
                raise
    if last_model_error is not None:
        raise last_model_error
    return ""


def _extract_via_ollama(image: Image.Image, cfg: VLMConfig, ocr_context: Optional[str] = None) -> str:
    """Local Ollama vision-model path."""
    from .local_llm import OllamaConfig, ollama_vision_extract
    
    prompt = _EXTRACT_ONLY_PROMPT
    if ocr_context and ocr_context.strip():
        prompt += _OCR_GROUNDING_PROMPT.format(ocr_context=ocr_context.strip()[:5000])

    ollama_cfg = OllamaConfig(
        base_url=cfg.ollama_base_url,
        vlm_model=cfg.ollama_vlm_model,
        timeout=cfg.ollama_timeout,
    )
    return ollama_vision_extract(
        cfg=ollama_cfg,
        image=image,
        prompt=prompt,
        temperature=0.0,
        max_tokens=8192,
    )

