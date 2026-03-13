from __future__ import annotations

import json
import os

import google.auth
from dotenv import load_dotenv
from google import genai
from google.genai import types


def _env_true(name: str) -> bool:
    value = (os.getenv(name, "") or "").strip().lower()
    return value in ("1", "true", "yes", "y", "on")


def use_vertex_ai() -> bool:
    return _env_true("VERTEX_ENABLED") or _env_true("GOOGLE_GENAI_USE_VERTEXAI")


def _resolve_project_id() -> str:
    explicit_project = (os.getenv("VERTEX_PROJECT_ID", "") or "").strip()
    if explicit_project:
        return explicit_project

    credentials_path = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "") or "").strip()
    if credentials_path and os.path.exists(credentials_path):
        try:
            with open(credentials_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            project_id = (payload.get("project_id") or "").strip()
            if project_id:
                return project_id
        except Exception:
            pass

    explicit_project = (
        os.getenv("GOOGLE_CLOUD_PROJECT", "")
        or os.getenv("GCP_PROJECT", "")
        or ""
    ).strip()
    if explicit_project:
        return explicit_project

    try:
        _, adc_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        if adc_project:
            return adc_project
    except Exception:
        pass

    raise ValueError(
        "Google Cloud project id bulunamadi. VERTEX_PROJECT_ID veya GOOGLE_CLOUD_PROJECT ayarlayin."
    )


def _resolve_vertex_location(model_name: str = "") -> str:
    primary = (
        (os.getenv("VERTEX_LOCATION", "") or "").strip()
        or "global"
    )
    fallback = (
        (os.getenv("VERTEX_FALLBACK_LOCATION", "") or "").strip()
        or (os.getenv("GOOGLE_CLOUD_LOCATION", "") or "").strip()
        or primary
    )
    model = (model_name or "").strip().lower()
    if "gemini-2.5-pro" in model:
        return fallback
    if "gemini-3.1-pro" in model:
        return primary
    return primary


def _resolve_timeout_ms() -> int:
    raw = (
        os.getenv("VERTEX_REQUEST_TIMEOUT_MS", "")
        or os.getenv("GEMINI_REQUEST_TIMEOUT_MS", "")
        or "120000"
    ).strip()
    try:
        timeout_ms = int(raw)
    except Exception:
        timeout_ms = 120000
    return max(1000, min(600000, timeout_ms))


def build_gemini_client(api_key: str = "", model_name: str = ""):
    """
    Build a Google GenAI client for either:
    - Gemini Developer API (AI Studio) via API key, or
    - Vertex AI via ADC / service account credentials.

    Vertex AI takes precedence when VERTEX_ENABLED / GOOGLE_GENAI_USE_VERTEXAI is enabled.
    """
    load_dotenv(override=False)

    if use_vertex_ai():
        project = _resolve_project_id()
        location = _resolve_vertex_location(model_name=model_name)
        timeout_ms = _resolve_timeout_ms()
        return genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )

    key = (api_key or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")).strip()
    if not key:
        raise ValueError(
            "Gemini kullanimi icin GEMINI_API_KEY (veya GOOGLE_API_KEY) tanimli olmali."
        )
    return genai.Client(api_key=key)


def gemini_model_candidates(primary_model: str, fallback_env: str = "GEMINI_FALLBACK_MODEL") -> list[str]:
    primary = (primary_model or "").strip()
    fallback = (os.getenv(fallback_env, "") or "").strip()
    out: list[str] = []
    for item in (primary, fallback):
        if item and item not in out:
            out.append(item)
    return out


def vertex_location_for_model(model_name: str) -> str:
    return _resolve_vertex_location(model_name=model_name)


def is_model_not_found_error(exc: Exception) -> bool:
    msg = str(exc or "")
    lowered = msg.lower()
    return (
        "not_found" in lowered
        or "publisher model" in lowered
        or "was not found" in lowered
        or "does not have access to it" in lowered
    )
