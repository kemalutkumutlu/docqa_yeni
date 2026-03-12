"""
Local LLM / VLM backend via Ollama HTTP API.

This module provides thin wrappers around the Ollama REST API so that:
  - generate_answer_local() can replace Gemini for RAG answer generation
  - extract_text_from_image_local() can replace Gemini VLM for extract-only ingestion

No external dependency beyond `urllib` (stdlib).  Ollama must be running locally.
"""
from __future__ import annotations

import base64
import io
import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Callable, Optional

from PIL import Image


@dataclass(frozen=True)
class OllamaConfig:
    """
    Configuration for the local Ollama backend.

    base_url: e.g. "http://localhost:11434"
    llm_model: text model name in Ollama (e.g. "qwen2.5:7b-instruct-q4_K_M")
    vlm_model: vision model name in Ollama (e.g. "llava:7b")
    timeout: HTTP request timeout in seconds
    """

    base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"
    vlm_model: str = "llava:7b"
    timeout: int = 120


def ollama_is_available(cfg: OllamaConfig) -> tuple[bool, str]:
    """
    Lightweight health check for the local Ollama service.
    Returns (available, detail).
    """
    url = f"{cfg.base_url.rstrip('/')}/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=min(int(cfg.timeout), 10)) as resp:
            if 200 <= getattr(resp, "status", 0) < 300:
                return True, "ok"
            return False, f"HTTP {getattr(resp, 'status', 'unknown')}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _ollama_generate(
    base_url: str,
    model: str,
    prompt: str,
    system: str = "",
    images: Optional[list[str]] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: int = 120,
) -> str:
    """
    Call Ollama /api/generate (streaming disabled) and return the response text.

    images: list of base64-encoded image strings (for vision models).
    """
    url = f"{base_url.rstrip('/')}/api/generate"

    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system:
        payload["system"] = system
    if images:
        payload["images"] = images

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # Simple retry for transient connection errors (Ollama might be loading a model).
    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return (body.get("response") or "").strip()
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last_err = e
            if attempt >= 3:
                raise
            time.sleep(min(8.0, 1.5 * (2 ** (attempt - 1))))

    raise last_err  # type: ignore[misc]  # unreachable


def _ollama_generate_stream(
    base_url: str,
    model: str,
    prompt: str,
    system: str = "",
    images: Optional[list[str]] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: int = 120,
    on_token: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Call Ollama /api/generate with streaming enabled and accumulate final text.

    Notes:
      - Each line from Ollama is a JSON event.
      - If a transient connection error happens before any token arrives, we retry.
      - If it happens after tokens started, we raise to avoid duplicating output.
    """
    url = f"{base_url.rstrip('/')}/api/generate"
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system:
        payload["system"] = system
    if images:
        payload["images"] = images

    data = json.dumps(payload).encode("utf-8")
    text_parts: list[str] = []
    last_err: Optional[Exception] = None

    for attempt in range(1, 4):
        emitted_any = bool(text_parts)
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    token = (event.get("response") or "")
                    if token:
                        text_parts.append(token)
                        if on_token:
                            on_token(token)
                        emitted_any = True
                    if event.get("done"):
                        return "".join(text_parts).strip()
                return "".join(text_parts).strip()
        except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            if attempt >= 3 or emitted_any:
                raise
            time.sleep(min(8.0, 1.5 * (2 ** (attempt - 1))))

    raise last_err  # type: ignore[misc]


def ollama_chat(
    cfg: OllamaConfig,
    system: str,
    user_message: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> str:
    """
    Text-only generation via Ollama (used for RAG answers and chat).
    """
    return _ollama_generate(
        base_url=cfg.base_url,
        model=cfg.llm_model,
        prompt=user_message,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=cfg.timeout,
    )


def ollama_chat_stream(
    cfg: OllamaConfig,
    system: str,
    user_message: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    on_token: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Text generation via Ollama with token streaming callback.
    """
    return _ollama_generate_stream(
        base_url=cfg.base_url,
        model=cfg.llm_model,
        prompt=user_message,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=cfg.timeout,
        on_token=on_token,
    )


def ollama_vision_extract(
    cfg: OllamaConfig,
    image: Image.Image,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """
    Vision (multimodal) generation via Ollama — send an image + text prompt.
    Used for extract-only VLM ingestion.
    """
    # Encode image as PNG → base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return _ollama_generate(
        base_url=cfg.base_url,
        model=cfg.vlm_model,
        prompt=prompt,
        images=[img_b64],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=cfg.timeout,
    )
