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
You are an OCR+layout extraction engine.

Task: Extract ONLY the text that is visible in the image. Do not add, infer, or summarize.

Rules:
- Output plain text or Markdown that preserves layout as best as possible.
- If you see headings, keep them on their own lines.
- If you see lists, keep bullet/numbering.
- If you see tables, represent them as Markdown tables if possible; otherwise keep rows line-by-line.
- Do not translate.
- If a region is unreadable, omit it (do not guess).
"""



_OCR_GROUNDING_PROMPT = """\

---
SUPPLEMENTARY OCR TRUTH:
Below is the raw text extracted by a standard formatting engine. It may have poor layout formatting, but its character and number recognition is highly accurate.
Use this text strictly as a grounding reference to prevent hallucinations.
Correct the layout using the visual image, but rely on the OCR text for exact spellings, numbers, and punctuation.

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
                    max_output_tokens=4096,
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
        max_tokens=4096,
    )

