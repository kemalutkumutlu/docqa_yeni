from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types
from PIL import Image

from .gemini_client import build_gemini_client


@dataclass(frozen=True)
class VLMConfig:
    """
    Vision-Language extraction config.

    mode:
      - "off": never call VLM
      - "auto": call VLM only when text quality is low
      - "force": always call VLM (expensive)

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


def extract_text_from_image(image: Image.Image, cfg: VLMConfig) -> str:
    """
    Extract text from an image using a multimodal model (extract-only).
    Dispatches to Gemini API or local Ollama based on cfg.provider.
    Returns extracted text (may be empty).
    """
    if cfg.provider == "local":
        return _extract_via_ollama(image, cfg)
    return _extract_via_gemini(image, cfg)


def _extract_via_gemini(image: Image.Image, cfg: VLMConfig) -> str:
    """Gemini API path (existing behavior)."""
    # Encode image as PNG bytes
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    client = build_gemini_client(cfg.api_key)
    resp = client.models.generate_content(
        model=cfg.model,
        contents=[
            types.Part.from_text(text=_EXTRACT_ONLY_PROMPT),
            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=4096,
        ),
    )
    return (resp.text or "").strip()


def _extract_via_ollama(image: Image.Image, cfg: VLMConfig) -> str:
    """Local Ollama vision-model path."""
    from .local_llm import OllamaConfig, ollama_vision_extract

    ollama_cfg = OllamaConfig(
        base_url=cfg.ollama_base_url,
        vlm_model=cfg.ollama_vlm_model,
        timeout=cfg.ollama_timeout,
    )
    return ollama_vision_extract(
        cfg=ollama_cfg,
        image=image,
        prompt=_EXTRACT_ONLY_PROMPT,
        temperature=0.0,
        max_tokens=4096,
    )

