from __future__ import annotations

import json
import os
import socket
from typing import Any

from dotenv import load_dotenv


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
        import google.auth

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
    from google import genai
    from google.genai import types

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


def gemini_model_candidates(
    primary_model: str,
    fallback_env: str = "GEMINI_FALLBACK_MODEL",
    fallback_model: str = "",
) -> list[str]:
    primary = (primary_model or "").strip()
    fallback = (fallback_model or os.getenv(fallback_env, "") or "").strip()
    out: list[str] = []
    for item in (primary, fallback):
        if item and item not in out:
            out.append(item)
    return out


def vertex_location_for_model(model_name: str) -> str:
    return _resolve_vertex_location(model_name=model_name)


def _coerce_status_code(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        try:
            return int(value.strip())
        except Exception:
            return None
    return None


def _status_candidates(exc: Exception) -> list[Any]:
    response = getattr(exc, "response", None)
    payload = getattr(exc, "body", None)
    items: list[Any] = [
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(exc, "status", None),
        getattr(exc, "reason", None),
    ]
    if response is not None:
        items.extend(
            [
                getattr(response, "status_code", None),
                getattr(response, "status", None),
                getattr(response, "reason_phrase", None),
            ]
        )
        try:
            payload = response.json()
        except Exception:
            pass
    if isinstance(payload, dict):
        error_payload = payload.get("error") if isinstance(payload.get("error"), dict) else payload
        items.extend(
            [
                error_payload.get("code"),
                error_payload.get("status"),
                error_payload.get("message"),
            ]
        )
    return [item for item in items if item not in (None, "")]


def _error_status_code(exc: Exception) -> int | None:
    for candidate in _status_candidates(exc):
        code = _coerce_status_code(candidate)
        if code is not None:
            return code
    return None


def _error_status_names(exc: Exception) -> set[str]:
    names: set[str] = set()
    for candidate in _status_candidates(exc):
        if isinstance(candidate, str):
            normalized = candidate.strip().upper().replace(" ", "_")
            if normalized:
                names.add(normalized)
    return names


def is_retryable_api_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, OSError, socket.timeout)):
        return True

    status_code = _error_status_code(exc)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True

    status_names = _error_status_names(exc)
    if status_names.intersection(
        {
            "RESOURCE_EXHAUSTED",
            "UNAVAILABLE",
            "INTERNAL",
            "DEADLINE_EXCEEDED",
            "ABORTED",
            "TOO_MANY_REQUESTS",
            "SERVICE_UNAVAILABLE",
            "GATEWAY_TIMEOUT",
            "BAD_GATEWAY",
        }
    ):
        return True
    return False


def is_model_not_found_error(exc: Exception) -> bool:
    status_code = _error_status_code(exc)
    if status_code == 404:
        return True

    status_names = _error_status_names(exc)
    if "NOT_FOUND" in status_names:
        return True

    msg = str(exc or "").lower()
    if not msg:
        return False
    return (
        "publisher model" in msg
        or "was not found" in msg
        or "does not have access to it" in msg
    )
