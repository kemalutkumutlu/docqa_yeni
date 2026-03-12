from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_settings  # noqa: E402
from src.core.embedding import Embedder  # noqa: E402
from src.core.gemini_client import (  # noqa: E402
    build_gemini_client,
    gemini_model_candidates,
    is_model_not_found_error,
    use_vertex_ai,
)
from src.core.vlm_extract import VLMConfig, extract_text_from_image  # noqa: E402


def _print(msg: str) -> None:
    print(msg, flush=True)


def _test_generation(settings) -> str:
    client = build_gemini_client(settings.gemini_api_key)
    last_error: Exception | None = None
    for model_name in gemini_model_candidates(settings.gemini_model):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents="Reply with exactly: ok",
            )
            text = (resp.text or "").strip()
            if not text:
                raise RuntimeError(f"Empty response from model {model_name}")
            _print(f"[OK] generation model={model_name} response={text!r}")
            if model_name != settings.gemini_model:
                _print(f"[WARN] primary model unavailable, fallback used: {model_name}")
            return model_name
        except Exception as exc:
            last_error = exc
            if not is_model_not_found_error(exc):
                raise
    raise RuntimeError(f"Generation preflight failed: {last_error}")


def _test_embedding(settings) -> None:
    model_name = settings.embedding_model
    vec = Embedder(model_name=model_name, device=settings.embedding_device).embed_query(
        "vertex embedding smoke test"
    )
    _print(f"[OK] embedding model={model_name} dim={len(vec)}")


def _make_test_image() -> Image.Image:
    img = Image.new("RGB", (640, 180), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 60), "PRECHECK VLM TEST", fill="black")
    return img


def _test_vlm(settings) -> None:
    if settings.vlm_mode == "off":
        _print("[SKIP] VLM disabled")
        return

    cfg = VLMConfig(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        mode=settings.vlm_mode,
        max_pages=settings.vlm_max_pages,
        provider=settings.vlm_provider,
        ollama_base_url=settings.ollama_base_url,
        ollama_vlm_model=settings.ollama_vlm_model,
        ollama_timeout=settings.ollama_timeout,
    )
    text = extract_text_from_image(_make_test_image(), cfg=cfg)
    cleaned = " ".join((text or "").split())
    if not cleaned:
        raise RuntimeError("VLM preflight returned empty text")
    _print(f"[OK] vlm provider={cfg.provider} sample={cleaned[:80]!r}")


def main() -> int:
    settings = load_settings()
    _print(
        "[INFO] mode="
        f"{'vertex' if use_vertex_ai() else 'ai_studio'} "
        f"llm={settings.gemini_model} emb={settings.embedding_model} "
        f"vlm={settings.vlm_provider}/{settings.vlm_mode}"
    )
    _test_generation(settings)
    _test_embedding(settings)
    _test_vlm(settings)
    _print("[OK] preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
