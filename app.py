"""
Chainlit UI — Document Q&A with Multimodal Hierarchical RAG.

Run:
    python -m chainlit run app.py -w
"""
from __future__ import annotations

import asyncio
import html
import json
import os
import re
import shutil
import tempfile
from queue import Empty, SimpleQueue
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Select, Slider

from src.config import load_settings
from src.core.ingestion import OCRConfig
from src.core.local_llm import OllamaConfig, ollama_is_available
from src.core.pipeline import RAGPipeline
from src.core.vlm_extract import VLMConfig
from src.core.gemini_client import use_vertex_ai


# ── Helpers ──────────────────────────────────────────────────────────────────

# Auto-exit (dev convenience): if enabled, kill the server when the last client disconnects.
# This prevents "port already in use" when you close the browser tab but forget the terminal.
_ACTIVE_CHAT_SESSIONS = 0
_EXIT_TASK: asyncio.Task | None = None
_EXIT_LOCK = asyncio.Lock()
_THREAD_TAG_RE = re.compile(r"^<!--THREAD:([A-Za-z0-9:_\-.]+)-->\s*")
_OPEN_THREAD_CMD_RE = re.compile(r"^/open_thread(?:\s+([A-Za-z0-9:_\-.]+))?\s*$", re.IGNORECASE)
_THREAD_MEMORY: dict[str, list[dict[str, str]]] = {}
_THREAD_PIPELINES: dict[str, RAGPipeline] = {}
_THREAD_MEMORY_MAX_MSGS = 120
_SIDEBAR_REV_KEY = "sidebar_render_rev"


def _auto_exit_enabled() -> bool:
    v = (os.getenv("AUTO_EXIT_ON_NO_CLIENTS", "") or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _auto_exit_grace_seconds() -> float:
    raw = (os.getenv("AUTO_EXIT_GRACE_SECONDS", "8") or "").strip()
    try:
        sec = float(raw)
    except Exception:
        sec = 8.0
    return max(0.0, min(120.0, sec))


def _extract_thread_marker(text: str) -> tuple[str | None, str]:
    raw = (text or "").strip()
    m = _THREAD_TAG_RE.match(raw)
    if not m:
        return None, raw
    thread_id = (m.group(1) or "").strip()
    rest = raw[m.end():].strip()
    return (thread_id or None), rest


def _thread_history_dir() -> Path:
    configured = (os.getenv("THREAD_HISTORY_DIR", "") or "").strip()
    if configured:
        return Path(configured)
    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    return data_dir / "thread_history"


def _thread_history_path(thread_id: str | None) -> Path | None:
    tid = (thread_id or "").strip()
    if not tid:
        return None
    return _thread_history_dir() / f"{tid}.json"


def _thread_memory_load(thread_id: str | None) -> list[dict[str, str]]:
    tid = (thread_id or "").strip()
    if not tid:
        return []
    cached = _THREAD_MEMORY.get(tid)
    if cached is not None:
        return cached

    path = _thread_history_path(tid)
    if path is None or not path.exists():
        return []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    items = raw.get("messages") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return []

    loaded: list[dict[str, str]] = []
    for item in items[-_THREAD_MEMORY_MAX_MSGS:]:
        if not isinstance(item, dict):
            continue
        role = (item.get("role") or "").strip()
        content = re.sub(r"\s+", " ", (item.get("content") or "").strip())
        if role not in ("user", "assistant") or not content:
            continue
        loaded.append({"role": role, "content": content})
    if loaded:
        _THREAD_MEMORY[tid] = loaded
    return loaded


def _thread_memory_persist(thread_id: str | None) -> None:
    tid = (thread_id or "").strip()
    if not tid:
        return
    path = _thread_history_path(tid)
    if path is None:
        return
    messages = _THREAD_MEMORY.get(tid, [])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "thread_id": tid,
            "message_count": len(messages),
            "messages": messages[-_THREAD_MEMORY_MAX_MSGS:],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # Thread persistence is best-effort and must not break chat flow.
        return


def _thread_memory_add(thread_id: str | None, role: str, content: str) -> None:
    tid = (thread_id or "").strip()
    msg = re.sub(r"\s+", " ", (content or "").strip())
    if not tid or not msg or role not in ("user", "assistant"):
        return
    buf = list(_thread_memory_load(tid))
    if buf and buf[-1].get("role") == role and buf[-1].get("content") == msg:
        return
    buf.append({"role": role, "content": msg})
    if len(buf) > _THREAD_MEMORY_MAX_MSGS:
        buf = buf[-_THREAD_MEMORY_MAX_MSGS:]
    _THREAD_MEMORY[tid] = buf
    _thread_memory_persist(tid)


def _thread_pipeline_get(thread_id: str | None) -> RAGPipeline | None:
    tid = (thread_id or "").strip()
    if not tid:
        return None
    return _THREAD_PIPELINES.get(tid)


def _thread_pipeline_set(thread_id: str | None, pipeline: RAGPipeline | None) -> None:
    tid = (thread_id or "").strip()
    if not tid or pipeline is None:
        return
    _THREAD_PIPELINES[tid] = pipeline


def _resolve_sidebar_pipeline(preferred: RAGPipeline | None = None) -> RAGPipeline | None:
    # When an explicit pipeline is passed (e.g. after file upload), prefer it
    # over thread pipeline — the caller has the most up-to-date reference.
    if preferred is not None:
        return preferred

    active_tid = (cl.user_session.get("ui_thread_id") or "").strip()
    if active_tid:
        thread_pipeline = _thread_pipeline_get(active_tid)
        if thread_pipeline is not None:
            return thread_pipeline

    session_pipeline = cl.user_session.get("pipeline")
    if session_pipeline is not None:
        return session_pipeline

    return None


def _next_sidebar_rev() -> int:
    cur = cl.user_session.get(_SIDEBAR_REV_KEY)
    if not isinstance(cur, int):
        cur = 0
    cur += 1
    cl.user_session.set(_SIDEBAR_REV_KEY, cur)
    return cur


async def _cancel_exit_task() -> None:
    global _EXIT_TASK
    if _EXIT_TASK and not _EXIT_TASK.done():
        _EXIT_TASK.cancel()
    _EXIT_TASK = None


async def _schedule_auto_exit_if_idle() -> None:
    """
    If enabled and no active sessions remain, exit the process after a grace period.
    """
    global _EXIT_TASK
    if not _auto_exit_enabled():
        return
    grace = _auto_exit_grace_seconds()

    async def _worker() -> None:
        await asyncio.sleep(grace)
        # Double-check state right before exiting.
        async with _EXIT_LOCK:
            if _ACTIVE_CHAT_SESSIONS <= 0:
                os._exit(0)  # noqa: S404 - intentional hard-exit for dev convenience

    await _cancel_exit_task()
    _EXIT_TASK = asyncio.create_task(_worker())


ACCEPTED_MIME = [
    "application/pdf",
    "image/png",
    "image/jpeg",
]

_SMALLTALK_PATTERNS = [
    # Turkish
    r"^(merhaba|selam|slm|selamlar)\b",
    r"\bnasılsın\b|\bnaber\b|\bnasılsınız\b",
    r"^(teşekkür(ler)?|tesekkur(ler)?|sağ ol|sagol|eyvallah|rica ederim)\b",
    r"^(günaydın|iyi akşamlar|iyi geceler|iyi günler)\b",
    r"\bkimsin\b|\bsen kimsin\b|\bne yapıyorsun\b",
    # Follow-up smalltalk
    r"\bben\s+nas\w*ls\w*m\b",
    r"\bsorm\w*\s+m\w*s\w*n\b",  # "sormicak mısın / sormayacak mısın" etc.
    r"\bemin\s+m\w*s\w*n\b|\bgercekten\s+mi\b|\bciddi\s+misin\b",
    # English
    r"^(hi|hello|hey)\b",
    r"\bhow are you\b|\bhow's it going\b",
    r"^(thanks|thank you)\b",
    r"\bwho are you\b",
    r"\bare you sure\b|\breally\??\b",
]

_PRAISE_PATTERNS = [
    # Turkish praise / compliments
    r"^(aferin|bravo|helal|tebrik(ler)?|güzel|guzel|iyi\s*i[şs])\b",
    r"\b(harikasın|harikasin|mükemmel|mukemmel|süpersin|supersin|kralsın|kralsin)\b",
    # English praise
    r"\b(great job|well done|nice work|awesome|you are awesome|you're awesome|congrats)\b",
]

_NEGATIVE_FEELING_PATTERNS = [
    # Turkish negative mood
    r"\b(üzgünüm|uzgunum|moralim bozuk|canım sıkkın|canim sikkin)\b",
    r"\b(kötüyüm|kotuyum|berbatım|berbatim|cok kotuyum|çok kötüyüm)\b",
    r"\b(stresliyim|kaygılıyım|kaygiliyim|endişeliyim|endiseliyim|yoruldum|bıktım|biktim)\b",
    # English negative mood
    r"\b(i am sad|i'm sad|i feel bad|i am upset|i'm upset|bad day|feeling down)\b",
]

_CHAT_MODE_REQUEST_PATTERNS = [
    # Turkish
    r"\bsohbet\s+modu\b",
    r"\bsohbet\s+moduna\b.*\b(geç|gec|aç|ac)\w*\b",
    r"\bchat\s+modu\b",
    r"\bchat\s+moduna\b.*\b(geç|gec)\w*\b",
    r"\bsohbete\b.*\b(geç|gec)\w*\b",
    # English
    r"\bchat\s+mode\b",
    r"\bswitch\s+to\s+chat\b",
]

_DOC_CUE_PATTERNS = [
    r"\bbelge\b|\bdoküman\b|\bdokuman\b|\bpdf\b|\bdosya\b",
    r"\bsayfa\b|\bbaşlık\b|\bbölüm\b|\bmadde\b|\biçerik\b|\bicerik\b",
    r"\bnelerdir\b|\blistele\b|\bsırala\b|\bsirala\b|\bhepsi\b|\btümü\b|\btumu\b",
]

_DOC_MODE_REQUEST_PATTERNS = [
    # Turkish
    r"\bbelge\s+modu\b|\bdoküman\s+modu\b|\bdokuman\s+modu\b",
    r"\bbelge\s+moduna\b.*\b(dön|don|geç|gec)\w*\b",
    r"\bbelge\s+moduna\s+nasıl\b|\bbelge\s+moduna\s+nas\w*l\b",
    r"\bbelge\s+moduna\s+nas\w*l\s+d\w*n\w*",
    # English
    r"\bdoc\s+mode\b|\bdocument\s+mode\b",
]

_CHAT_PROFILE_TO_PROVIDER = {
    "Gemini": "gemini",
    "OpenAI": "openai",
    "Local": "local",
    "Extractive": "none",
}
_CHAT_HISTORY_KEY = "recent_user_messages"
_CHAT_HISTORY_MAX = 12
_RUNTIME_OVERRIDES_KEY = "runtime_overrides"
_PROCESSING_MODE_VALUES = ["classic", "multimodal"]
_MULTIMODAL_ANSWER_MODE_VALUES = ["off", "auto", "on"]
_EMBEDDING_MODEL_PRESETS = [
    "gemini-embedding-001",
    "gemini-embedding-2-preview",
    "auto",
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
]
_EMBEDDING_DEVICE_VALUES = ["auto", "cpu", "cuda"]
_VLM_MODE_VALUES = ["off", "auto", "force"]
_VLM_PROVIDER_VALUES = ["gemini", "local"]
_VLM_MAX_PAGES_LIMIT = 200
_VLM_MAX_PAGES_DEFAULT = 25
_PROFILE_WARNING_KEY = "profile_warning"


def _get_cached_settings():
    settings = cl.user_session.get("app_settings")
    if settings is None:
        settings = load_settings()
        cl.user_session.set("app_settings", settings)
    return settings


def _default_profile_name() -> str:
    settings = load_settings()
    provider = (settings.llm_provider or "gemini").strip().lower()
    for profile_name, profile_provider in _CHAT_PROFILE_TO_PROVIDER.items():
        if profile_provider == provider:
            return profile_name
    return "Gemini"


def _cuda_available() -> bool:
    try:
        import torch  # noqa: WPS433

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _clamp_int(value, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(float(value))
    except Exception:
        parsed = default
    return max(min_value, min(max_value, parsed))


def _sanitize_select_value(value, allowed: list[str], fallback: str) -> str:
    text = (str(value or "")).strip().lower()
    if text in allowed:
        return text
    return fallback


def _resolve_embedding_model_choice(choice: str, embedding_device: str) -> str:
    selected = (choice or "").strip().lower()
    if selected == "auto":
        use_cuda = _cuda_available() and embedding_device != "cpu"
        return "intfloat/multilingual-e5-base" if use_cuda else "intfloat/multilingual-e5-small"
    return (choice or "").strip()


def _runtime_overrides() -> dict:
    raw = cl.user_session.get(_RUNTIME_OVERRIDES_KEY)
    return raw if isinstance(raw, dict) else {}


def _get_runtime_value(key: str, fallback):
    return _runtime_overrides().get(key, fallback)


def _has_gemini_backend(settings) -> bool:
    if use_vertex_ai():
        return True
    return bool((settings.gemini_api_key or "").strip())


def _resolve_llm_provider_choice(requested_provider: str, settings, ollama_cfg: OllamaConfig) -> tuple[str, str | None]:
    provider = (requested_provider or "").strip().lower()
    gemini_ready = _has_gemini_backend(settings)
    openai_ready = bool((settings.openai_api_key or "").strip())
    ollama_ready, ollama_detail = ollama_is_available(ollama_cfg)

    if provider == "none":
        return "none", None
    if provider == "gemini":
        if gemini_ready:
            return "gemini", None
        if openai_ready:
            return "openai", "Gemini auth hazir degil. Gecici olarak `OpenAI` kullaniliyor."
        if ollama_ready:
            return "local", "Gemini auth hazir degil. Gecici olarak `Local` kullaniliyor."
        return "none", "Gemini auth hazir degil. Gecici olarak `Extractive` moda dusuldu."
    if provider == "openai":
        if openai_ready:
            return "openai", None
        if gemini_ready:
            return "gemini", "Secilen profil `OpenAI`, ancak `OPENAI_API_KEY` tanimli degil. `Gemini` fallback kullaniliyor."
        if ollama_ready:
            return "local", "Secilen profil `OpenAI`, ancak `OPENAI_API_KEY` tanimli degil. `Local` fallback kullaniliyor."
        return "none", "Secilen profil `OpenAI`, ancak `OPENAI_API_KEY` tanimli degil. `Extractive` fallback kullaniliyor."
    if provider == "local":
        if ollama_ready:
            return "local", None
        if gemini_ready:
            return "gemini", f"Secilen profil `Local`, ancak Ollama erisilemedi (`{ollama_detail}`). `Gemini` fallback kullaniliyor."
        if openai_ready:
            return "openai", f"Secilen profil `Local`, ancak Ollama erisilemedi (`{ollama_detail}`). `OpenAI` fallback kullaniliyor."
        return "none", f"Secilen profil `Local`, ancak Ollama erisilemedi (`{ollama_detail}`). `Extractive` fallback kullaniliyor."
    return "gemini", None


def _resolve_vlm_provider_choice(requested_provider: str, settings, ollama_cfg: OllamaConfig) -> tuple[str, str | None]:
    provider = (requested_provider or "").strip().lower()
    gemini_ready = _has_gemini_backend(settings)
    ollama_ready, ollama_detail = ollama_is_available(ollama_cfg)

    if provider == "local":
        if ollama_ready:
            return "local", None
        if gemini_ready:
            return "gemini", f"Secilen `VLM Provider=local`, ancak Ollama erisilemedi (`{ollama_detail}`). `gemini` fallback kullaniliyor."
        return "local", f"Secilen `VLM Provider=local`, ancak Ollama erisilemedi (`{ollama_detail}`). Yukleme sirasinda VLM devre disi kalabilir."

    if provider == "gemini" and not gemini_ready:
        if ollama_ready:
            return "local", "Gemini VLM auth hazir degil. `local` fallback kullaniliyor."
        return "gemini", "Gemini VLM auth hazir degil. Yukleme sirasinda VLM devre disi kalabilir."

    return provider or "gemini", None


def _settings_widgets(pipeline: RAGPipeline) -> list:
    vlm_cfg = pipeline.vlm_config
    processing_mode_current = _sanitize_select_value(
        _get_runtime_value("processing_mode", getattr(pipeline, "processing_mode", "classic")),
        _PROCESSING_MODE_VALUES,
        getattr(pipeline, "processing_mode", "classic"),
    )
    multimodal_answer_mode_current = _sanitize_select_value(
        _get_runtime_value("multimodal_answer_mode", getattr(pipeline, "multimodal_answer_mode", "auto")),
        _MULTIMODAL_ANSWER_MODE_VALUES,
        getattr(pipeline, "multimodal_answer_mode", "auto"),
    )
    embedding_device_current = _sanitize_select_value(
        _get_runtime_value("embedding_device", pipeline.embedding_device),
        _EMBEDDING_DEVICE_VALUES,
        "auto",
    )
    embedding_choice_current = str(
        _get_runtime_value("embedding_model_choice", pipeline.embedding_model)
    ).strip()
    embedding_values = list(_EMBEDDING_MODEL_PRESETS)
    if pipeline.embedding_model not in embedding_values:
        embedding_values.append(pipeline.embedding_model)
    if embedding_choice_current and embedding_choice_current not in embedding_values:
        embedding_values.append(embedding_choice_current)
    if not embedding_choice_current:
        embedding_choice_current = pipeline.embedding_model

    vlm_mode_current = _sanitize_select_value(
        _get_runtime_value("vlm_mode", vlm_cfg.mode if vlm_cfg else "auto"),
        _VLM_MODE_VALUES,
        vlm_cfg.mode if vlm_cfg else "auto",
    )
    vlm_provider_current = _sanitize_select_value(
        _get_runtime_value("vlm_provider", vlm_cfg.provider if vlm_cfg else "gemini"),
        _VLM_PROVIDER_VALUES,
        vlm_cfg.provider if vlm_cfg else "gemini",
    )
    vlm_pages_current = _clamp_int(
        _get_runtime_value("vlm_max_pages", vlm_cfg.max_pages if vlm_cfg else _VLM_MAX_PAGES_DEFAULT),
        vlm_cfg.max_pages if vlm_cfg else _VLM_MAX_PAGES_DEFAULT,
        0,
        _VLM_MAX_PAGES_LIMIT,
    )

    return [
        Select(
            id="processing_mode",
            label="Processing Mode",
            values=_PROCESSING_MODE_VALUES,
            initial_value=processing_mode_current,
            description="classic: mevcut text-first akış. multimodal: visual page chunk + multimodal retrieval.",
        ),
        Select(
            id="multimodal_answer_mode",
            label="Multimodal Answer Generation",
            values=_MULTIMODAL_ANSWER_MODE_VALUES,
            initial_value=multimodal_answer_mode_current,
            description="Sadece multimodal mode + Gemini icin anlamli. off: text-only. auto: gorsel/layout sorularinda. on: visual evidence varsa her zaman.",
        ),
        Select(
            id="embedding_model",
            label="Embedding Model",
            values=embedding_values,
            initial_value=embedding_choice_current,
            description="Varsayilan: Gemini embedding. auto: lokal e5-base/e5-small secilir.",
        ),
        Select(
            id="embedding_device",
            label="Embedding Device",
            values=_EMBEDDING_DEVICE_VALUES,
            initial_value=embedding_device_current,
        ),
        Select(
            id="vlm_mode",
            label="VLM Mode",
            values=_VLM_MODE_VALUES,
            initial_value=vlm_mode_current,
        ),
        Select(
            id="vlm_provider",
            label="VLM Provider",
            values=_VLM_PROVIDER_VALUES,
            initial_value=vlm_provider_current,
        ),
        Slider(
            id="vlm_max_pages",
            label="VLM Max Pages",
            initial=float(vlm_pages_current),
            min=0,
            max=_VLM_MAX_PAGES_LIMIT,
            step=1,
        ),
    ]


def _apply_runtime_overrides_to_pipeline(pipeline: RAGPipeline) -> None:
    overrides = _runtime_overrides()
    if not overrides:
        return

    device = _sanitize_select_value(
        overrides.get("embedding_device", pipeline.embedding_device),
        _EMBEDDING_DEVICE_VALUES,
        pipeline.embedding_device,
    )
    model_choice = str(overrides.get("embedding_model_choice", pipeline.embedding_model)).strip()
    model_resolved = _resolve_embedding_model_choice(model_choice, device) or pipeline.embedding_model

    vlm_cfg = pipeline.vlm_config
    vlm_mode_default = vlm_cfg.mode if vlm_cfg else "auto"
    vlm_provider_default = vlm_cfg.provider if vlm_cfg else "gemini"
    vlm_pages_default = vlm_cfg.max_pages if vlm_cfg else _VLM_MAX_PAGES_DEFAULT

    pipeline.reconfigure_runtime(
        processing_mode=_sanitize_select_value(
            overrides.get("processing_mode"),
            _PROCESSING_MODE_VALUES,
            getattr(pipeline, "processing_mode", "classic"),
        ),
        multimodal_answer_mode=_sanitize_select_value(
            overrides.get("multimodal_answer_mode"),
            _MULTIMODAL_ANSWER_MODE_VALUES,
            getattr(pipeline, "multimodal_answer_mode", "auto"),
        ),
        embedding_model=model_resolved,
        embedding_device=device,
        vlm_mode=_sanitize_select_value(overrides.get("vlm_mode"), _VLM_MODE_VALUES, vlm_mode_default),
        vlm_provider=_sanitize_select_value(overrides.get("vlm_provider"), _VLM_PROVIDER_VALUES, vlm_provider_default),
        vlm_max_pages=_clamp_int(overrides.get("vlm_max_pages"), vlm_pages_default, 0, _VLM_MAX_PAGES_LIMIT),
    )


def _active_llm_model(pipeline: RAGPipeline | None) -> str:
    if pipeline is None:
        return "-"
    provider = (pipeline.llm_provider or "gemini").strip().lower()
    if provider == "openai":
        return (pipeline.openai_model or "").strip() or "-"
    if provider == "local":
        if pipeline.ollama_config:
            return (pipeline.ollama_config.llm_model or "").strip() or "-"
        return "-"
    if provider == "none":
        return "extractive"
    return (pipeline.gemini_model or "").strip() or "-"


def _apply_chat_profile_to_pipeline(pipeline: RAGPipeline) -> str | None:
    """
    Apply selected Chainlit chat profile to runtime LLM provider/model.
    Returns an optional user-facing warning.
    """
    settings = _get_cached_settings()
    profile_name = (cl.user_session.get("chat_profile") or "").strip()
    provider = _CHAT_PROFILE_TO_PROVIDER.get(profile_name)
    if not provider:
        return None

    # Keep provider-specific models synced with env defaults.
    pipeline.gemini_model = settings.gemini_model
    pipeline.openai_model = settings.openai_model
    pipeline.ollama_config = OllamaConfig(
        base_url=settings.ollama_base_url,
        llm_model=settings.ollama_llm_model,
        vlm_model=settings.ollama_vlm_model,
        timeout=settings.ollama_timeout,
    )

    resolved_provider, warning = _resolve_llm_provider_choice(provider, settings, pipeline.ollama_config)
    pipeline.llm_provider = resolved_provider
    return warning


async def _sync_profile_to_pipeline(pipeline: RAGPipeline) -> None:
    warning = _apply_chat_profile_to_pipeline(pipeline)
    previous = cl.user_session.get(_PROFILE_WARNING_KEY)
    if warning and warning != previous:
        await cl.Message(content=warning).send()
    cl.user_session.set(_PROFILE_WARNING_KEY, warning or "")


def _shorten_for_sidebar(text: str, limit: int = 84) -> str:
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)] + "..."


def _embedding_runtime_label(model_name: str | None, device: str | None) -> str:
    model = (model_name or "").strip().lower()
    dev = (device or "").strip().lower() or "-"
    if model.startswith("gemini-embedding-"):
        return "remote api"
    return dev


def _sidebar_escape(text: str | None) -> str:
    return html.escape((text or "").strip())


def _sidebar_chip(text: str | None, *, accent: bool = False) -> str:
    classes = "docqa-sidebar-chip"
    if accent:
        classes += " accent"
    return f'<span class="{classes}">{_sidebar_escape(text or "-")}</span>'


def _sidebar_row(label: str, values: list[str]) -> str:
    value_html = "".join(values) if values else _sidebar_chip("-")
    return (
        '<div class="docqa-sidebar-row">'
        f'<div class="docqa-sidebar-label">{_sidebar_escape(label)}</div>'
        f'<div class="docqa-sidebar-values">{value_html}</div>'
        "</div>"
    )


def _render_sidebar_panel(
    *,
    mode: str,
    llm_provider: str,
    llm_model: str,
    embedding_model: str,
    embedding_runtime: str,
    vlm_provider: str,
    vlm_mode: str,
    vlm_max_pages: str,
    active_doc: str | None,
    docs: list[str],
) -> str:
    active_doc_html = (
        f'<div class="docqa-sidebar-active">{_sidebar_escape(active_doc)}</div>'
        if active_doc
        else '<div class="docqa-sidebar-empty">Aktif belge yok</div>'
    )
    docs_html = "".join(
        f'<li class="docqa-sidebar-doc">{_sidebar_escape(doc)}</li>'
        for doc in docs[:6]
    )
    if not docs_html:
        docs_html = '<li class="docqa-sidebar-doc empty">Henüz belge yüklenmedi</li>'

    return (
        '<div class="docqa-sidebar-panel">'
        '<div class="docqa-sidebar-section">'
        '<div class="docqa-sidebar-heading">Oturum Durumu</div>'
        f'{_sidebar_row("Mod", [_sidebar_chip(mode, accent=True)])}'
        f'{_sidebar_row("LLM", [_sidebar_chip(llm_provider, accent=True), _sidebar_chip(llm_model)])}'
        f'{_sidebar_row("Embedding", [_sidebar_chip(embedding_model), _sidebar_chip(embedding_runtime)])}'
        f'{_sidebar_row("VLM", [_sidebar_chip(vlm_provider), _sidebar_chip(vlm_mode), _sidebar_chip(f"pages {vlm_max_pages}")])}'
        "</div>"
        '<div class="docqa-sidebar-section">'
        '<div class="docqa-sidebar-heading">Belge Bağlamı</div>'
        '<div class="docqa-sidebar-subhead">Aktif Belge</div>'
        f"{active_doc_html}"
        '<div class="docqa-sidebar-subhead">Yüklenen Belgeler</div>'
        f'<ul class="docqa-sidebar-docs">{docs_html}</ul>'
        "</div>"
        '<div class="docqa-sidebar-section">'
        '<div class="docqa-sidebar-heading">Hızlı Komutlar</div>'
        '<div class="docqa-sidebar-command"><span>Aktif belge seç</span><code>/use &lt;dosya&gt;</code></div>'
        '<div class="docqa-sidebar-command"><span>Belge modu</span><code>/doc</code></div>'
        '<div class="docqa-sidebar-command"><span>Sohbet modu</span><code>/chat</code></div>'
        "</div>"
        "</div>"
    )


def _get_chat_history() -> list[str]:
    raw = cl.user_session.get(_CHAT_HISTORY_KEY)
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if isinstance(item, str)]


def _append_chat_history_user_message(message: str) -> None:
    text = _shorten_for_sidebar(message)
    if not text:
        return
    items = _get_chat_history()
    if not items or items[-1] != text:
        items.append(text)
    if len(items) > _CHAT_HISTORY_MAX:
        items = items[-_CHAT_HISTORY_MAX:]
    cl.user_session.set(_CHAT_HISTORY_KEY, items)


def _looks_like_doc_mode_request(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    for pat in _DOC_MODE_REQUEST_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            return True
    return False


def _looks_like_chat_mode_request(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    for pat in _CHAT_MODE_REQUEST_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            return True
    return False


def _looks_like_doc_switch(query: str, pipeline: RAGPipeline) -> bool:
    """
    Heuristic to auto-switch from chat → doc when the user asks about documents.
    Conservative by design: only triggers if there are loaded documents AND the
    message contains clear document cues or mentions a loaded filename.
    """
    if not pipeline or not pipeline.has_documents:
        return False
    q = (query or "").strip().lower()
    if not q:
        return False

    # Filename mention is a strong signal.
    try:
        for name in pipeline.list_documents():
            if name and name.lower() in q:
                return True
    except Exception:
        pass

    # Otherwise require explicit document cue words.
    for pat in _DOC_CUE_PATTERNS[:2]:
        if re.search(pat, q, flags=re.IGNORECASE):
            return True
    return False


def _looks_like_smalltalk(query: str) -> bool:
    """
    Answer small-talk in chat mode even during doc sessions.
    Document-agnostic: uses only surface-form heuristics and avoids triggering
    when the message contains obvious document cues.
    """
    q = (query or "").strip().lower()
    if not q:
        return False
    for pat in _DOC_CUE_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            return False
    if len(q) <= 80:
        for pat in _SMALLTALK_PATTERNS:
            if re.search(pat, q, flags=re.IGNORECASE):
                return True
        for pat in _PRAISE_PATTERNS:
            if re.search(pat, q, flags=re.IGNORECASE):
                return True
        for pat in _NEGATIVE_FEELING_PATTERNS:
            if re.search(pat, q, flags=re.IGNORECASE):
                return True
    return False


def _smalltalk_style(query: str) -> str:
    """
    Return style hint for chat response:
      - "empathetic" for negative feelings
      - "congratulatory" for praise
      - "neutral" otherwise
    """
    q = (query or "").strip().lower()
    if not q:
        return "neutral"
    for pat in _DOC_CUE_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            return "neutral"

    for pat in _NEGATIVE_FEELING_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            return "empathetic"
    for pat in _PRAISE_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            return "congratulatory"
    return "neutral"


def _get_pipeline() -> RAGPipeline:
    """Get or lazily create the pipeline stored in the user session."""
    pipeline: RAGPipeline | None = cl.user_session.get("pipeline")
    if pipeline is None:
        settings = _get_cached_settings()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)

        # Ollama config (only used when LLM_PROVIDER=local or VLM_PROVIDER=local)
        ollama_cfg = OllamaConfig(
            base_url=settings.ollama_base_url,
            llm_model=settings.ollama_llm_model,
            vlm_model=settings.ollama_vlm_model,
            timeout=settings.ollama_timeout,
        )

        vlm_provider = getattr(settings, "vlm_provider", "gemini")

        pipeline = RAGPipeline(
            embedding_model=settings.embedding_model,
            chroma_dir=settings.chroma_dir,
            gemini_api_key=settings.gemini_api_key,
            gemini_model=settings.gemini_model,
            ocr_config=OCRConfig(
                enabled=getattr(settings, "ocr_enabled", True),
                lang="tur+eng",
                tesseract_cmd=settings.tesseract_cmd,
                tessdata_prefix=settings.tessdata_prefix,
                tesseract_config=getattr(settings, "tesseract_config", None),
            ),
            embedding_device=getattr(settings, "embedding_device", "auto"),
            processing_mode=getattr(settings, "processing_mode", "classic"),
            multimodal_answer_mode=getattr(settings, "multimodal_answer_mode", "auto"),
            vlm_config=VLMConfig(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                mode=settings.vlm_mode,
                max_pages=settings.vlm_max_pages,
                provider=vlm_provider,
                ollama_base_url=settings.ollama_base_url,
                ollama_vlm_model=settings.ollama_vlm_model,
                ollama_timeout=settings.ollama_timeout,
            ),
            llm_provider=settings.llm_provider,
            ollama_config=ollama_cfg,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
        )
        _apply_runtime_overrides_to_pipeline(pipeline)
        cl.user_session.set("pipeline", pipeline)
    return pipeline


async def _process_uploaded_file(file_path: str, file_name: str) -> str:
    """
    Ingest a single uploaded file into the pipeline.
    Returns a status message.
    """
    pipeline = _get_pipeline()
    path = Path(file_path)

    try:
        state = pipeline.add_document(path, display_name=file_name)
        lines = [
            f"**{file_name}** basariyla yuklendi ve indekslendi.",
            f"- Sayfa sayisi: {state.page_count}",
            f"- Chunk sayisi: {len(state.chunks)}",
            f"- Toplam indekslenen chunk: {pipeline.total_chunks}",
        ]
        if state.restored_from_cache:
            lines.append("- Bu belge kalici cache'den geri yuklendi; OCR/VLM/embedding tekrar calismadi.")
        if state.warnings:
            lines.append(f"- Uyarilar: {'; '.join(state.warnings)}")
        return "\n".join(lines)
    except Exception as e:
        return f"**{file_name}** yuklenirken hata olustu: {e}"


# ── Lifecycle hooks ──────────────────────────────────────────────────────────


@cl.set_chat_profiles
async def set_chat_profiles(_current_user, _language):
    default_profile = _default_profile_name()
    return [
        cl.ChatProfile(
            name="Gemini",
            display_name="Gemini",
            markdown_description="Google Gemini (RAG + chat).",
            default=(default_profile == "Gemini"),
        ),
        cl.ChatProfile(
            name="OpenAI",
            display_name="OpenAI",
            markdown_description="OpenAI-compatible Chat Completions ile RAG + chat.",
            default=(default_profile == "OpenAI"),
        ),
        cl.ChatProfile(
            name="Local",
            display_name="Local",
            markdown_description="Ollama local model (RAG + chat).",
            default=(default_profile == "Local"),
        ),
        cl.ChatProfile(
            name="Extractive",
            display_name="Extractive",
            markdown_description="LLM yok, sadece extractive cevap.",
            default=(default_profile == "Extractive"),
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("mode", "doc")
    cl.user_session.set(_CHAT_HISTORY_KEY, [])
    pipeline = _get_pipeline()
    profile_warning = _apply_chat_profile_to_pipeline(pipeline)

    # Track active sessions for optional auto-exit behavior.
    async with _EXIT_LOCK:
        global _ACTIVE_CHAT_SESSIONS
        _ACTIVE_CHAT_SESSIONS += 1
        await _cancel_exit_task()
        if _auto_exit_enabled():
            print(
                f"[auto-exit] enabled, active_sessions={_ACTIVE_CHAT_SESSIONS}",
                flush=True,
            )

    # Intentionally no auto welcome message to avoid initial layout jump in UI.
    if profile_warning:
        await cl.Message(content=profile_warning).send()
    cl.user_session.set(_PROFILE_WARNING_KEY, profile_warning or "")
    await cl.ChatSettings(_settings_widgets(pipeline)).send()
    await _update_documents_sidebar(pipeline)


@cl.on_settings_update
async def on_settings_update(values: dict):
    pipeline: RAGPipeline | None = cl.user_session.get("pipeline")
    if not pipeline:
        pipeline = _get_pipeline()

    current_vlm = pipeline.vlm_config
    current_pages = current_vlm.max_pages if current_vlm else _VLM_MAX_PAGES_DEFAULT

    selected_device = _sanitize_select_value(
        values.get("embedding_device"),
        _EMBEDDING_DEVICE_VALUES,
        pipeline.embedding_device,
    )
    selected_processing_mode = _sanitize_select_value(
        values.get("processing_mode"),
        _PROCESSING_MODE_VALUES,
        getattr(pipeline, "processing_mode", "classic"),
    )
    selected_multimodal_answer_mode = _sanitize_select_value(
        values.get("multimodal_answer_mode"),
        _MULTIMODAL_ANSWER_MODE_VALUES,
        getattr(pipeline, "multimodal_answer_mode", "auto"),
    )
    selected_model_choice = str(values.get("embedding_model") or pipeline.embedding_model).strip()
    if not selected_model_choice:
        selected_model_choice = pipeline.embedding_model
    resolved_model = _resolve_embedding_model_choice(selected_model_choice, selected_device) or pipeline.embedding_model

    selected_vlm_mode = _sanitize_select_value(
        values.get("vlm_mode"),
        _VLM_MODE_VALUES,
        current_vlm.mode if current_vlm else "auto",
    )
    selected_vlm_provider = _sanitize_select_value(
        values.get("vlm_provider"),
        _VLM_PROVIDER_VALUES,
        current_vlm.provider if current_vlm else "gemini",
    )
    selected_vlm_pages = _clamp_int(
        values.get("vlm_max_pages"),
        current_pages,
        0,
        _VLM_MAX_PAGES_LIMIT,
    )

    settings = _get_cached_settings()
    ollama_cfg = pipeline.ollama_config or OllamaConfig(
        base_url=settings.ollama_base_url,
        llm_model=settings.ollama_llm_model,
        vlm_model=settings.ollama_vlm_model,
        timeout=settings.ollama_timeout,
    )
    resolved_vlm_provider, vlm_warning = _resolve_vlm_provider_choice(
        selected_vlm_provider,
        settings,
        ollama_cfg,
    )

    cl.user_session.set(
        _RUNTIME_OVERRIDES_KEY,
        {
            "embedding_model_choice": selected_model_choice,
            "embedding_device": selected_device,
            "processing_mode": selected_processing_mode,
            "multimodal_answer_mode": selected_multimodal_answer_mode,
            "vlm_mode": selected_vlm_mode,
            "vlm_provider": resolved_vlm_provider,
            "vlm_max_pages": selected_vlm_pages,
        },
    )

    result = pipeline.reconfigure_runtime(
        processing_mode=selected_processing_mode,
        multimodal_answer_mode=selected_multimodal_answer_mode,
        embedding_model=resolved_model,
        embedding_device=selected_device,
        vlm_mode=selected_vlm_mode,
        vlm_provider=resolved_vlm_provider,
        vlm_max_pages=selected_vlm_pages,
    )

    if (
        result.get("embedding_changed")
        or result.get("vlm_changed")
        or result.get("processing_mode_changed")
        or result.get("multimodal_answer_mode_changed")
    ):
        messages = [
            "**Runtime ayarlari guncellendi**",
            f"- Processing: `{getattr(pipeline, 'processing_mode', 'classic')}`",
            f"- Multimodal Answer Gen: `{getattr(pipeline, 'multimodal_answer_mode', 'auto')}`",
            f"- Embedding: `{pipeline.embedding_model}` | `{_embedding_runtime_label(pipeline.embedding_model, pipeline.embedding_device)}`",
            f"- VLM: `{resolved_vlm_provider}` | `{selected_vlm_mode}` | max_pages=`{selected_vlm_pages}`",
        ]
        if result.get("index_rebuilt"):
            messages.append("- Mevcut dokumanlar icin embedding indeksi yeniden olusturuldu.")
        elif result.get("embedding_changed"):
            messages.append("- Embedding ayari degisti; indeks yeni ayarlarla hazir.")
        if result.get("vlm_changed"):
            messages.append("- VLM ayarlari bir sonraki dosya yuklemelerinde uygulanir.")
        if result.get("processing_mode_changed"):
            messages.append("- Processing mode bir sonraki belge yuklemelerinde uygulanir. Mevcut belgeleri yeniden yuklemeniz gerekir.")
        if result.get("multimodal_answer_mode_changed"):
            messages.append("- Multimodal answer generation ayari aninda uygulanir; reindex gerekmez.")
        if vlm_warning:
            messages.append(f"- Not: {vlm_warning}")
        await cl.Message(content="\n".join(messages)).send()

    await _update_documents_sidebar(pipeline)


@cl.on_chat_end
async def on_chat_end() -> None:
    # Decrement active sessions and auto-exit if enabled.
    async with _EXIT_LOCK:
        global _ACTIVE_CHAT_SESSIONS
        _ACTIVE_CHAT_SESSIONS = max(0, _ACTIVE_CHAT_SESSIONS - 1)
        if _ACTIVE_CHAT_SESSIONS == 0:
            if _auto_exit_enabled():
                print(
                    f"[auto-exit] last client disconnected, exiting in {_auto_exit_grace_seconds()}s",
                    flush=True,
                )
            await _schedule_auto_exit_if_idle()


def _process_uploaded_file_sync(file_path: str, file_name: str, pipeline: RAGPipeline) -> str:
    """Sync wrapper for file processing (called via make_async)."""
    return _process_uploaded_file_sync_with_progress(file_path, file_name, pipeline, None)


def _process_uploaded_file_sync_with_progress(
    file_path: str,
    file_name: str,
    pipeline: RAGPipeline,
    progress_callback=None,
) -> str:
    """Sync file processing with optional progress callback."""
    path = Path(file_path)

    try:
        state = pipeline.add_document(
            path,
            display_name=file_name,
            progress_callback=progress_callback,
        )
        lines = [
            f"**{file_name}** basariyla yuklendi ve indekslendi.",
            f"- Sayfa sayisi: {state.page_count}",
            f"- Chunk sayisi: {len(state.chunks)}",
            f"- Toplam indekslenen chunk: {pipeline.total_chunks}",
        ]
        if state.restored_from_cache:
            lines.append("- Bu belge kalici cache'den geri yuklendi; OCR/VLM/embedding tekrar calismadi.")
        if state.warnings:
            lines.append(f"- Uyarilar: {'; '.join(state.warnings)}")
        return "\n".join(lines)
    except Exception as e:
        return f"**{file_name}** yuklenirken hata olustu: {e}"


def _extract_uploaded_file_info(elem) -> tuple[str | None, str]:
    """
    Be defensive across Chainlit versions: uploaded elements may come as
    Element objects or plain dicts.
    """
    path = getattr(elem, "path", None)
    name = getattr(elem, "name", None)
    if isinstance(elem, dict):
        path = path or elem.get("path")
        name = name or elem.get("name")
    return path, (name or "dosya")


async def _update_documents_sidebar(pipeline: RAGPipeline | None = None) -> None:
    """
    Render a small "loaded docs + active doc" panel in the left sidebar.
    UI-only helper; does not affect retrieval/generation logic.
    """
    try:
        pipeline = _resolve_sidebar_pipeline(pipeline)

        mode = cl.user_session.get("mode") or "doc"
        docs = pipeline.list_documents() if pipeline else []
        active = pipeline.active_document_name if pipeline else None
        llm_provider = (pipeline.llm_provider if pipeline else "gemini") or "gemini"
        llm_model = _active_llm_model(pipeline)
        embedding_model = pipeline.embedding_model if pipeline else "-"
        embedding_runtime = _embedding_runtime_label(
            pipeline.embedding_model if pipeline else "-",
            pipeline.embedding_device if pipeline else "-",
        )
        vlm_provider = (pipeline.vlm_config.provider if (pipeline and pipeline.vlm_config) else "-")
        vlm_mode = (pipeline.vlm_config.mode if (pipeline and pipeline.vlm_config) else "-")
        vlm_max_pages = (pipeline.vlm_config.max_pages if (pipeline and pipeline.vlm_config) else "-")

        print(
            f"[ui] sidebar update: docs={docs}, active={active}, mode={mode}",
            flush=True,
        )

        content = _render_sidebar_panel(
            mode=mode,
            llm_provider=llm_provider,
            llm_model=llm_model,
            embedding_model=embedding_model,
            embedding_runtime=embedding_runtime,
            vlm_provider=vlm_provider,
            vlm_mode=vlm_mode,
            vlm_max_pages=str(vlm_max_pages),
            active_doc=active,
            docs=docs,
        )
        await cl.ElementSidebar.set_title("Belge Durumu")
        await cl.ElementSidebar.set_elements(
            [
                cl.Text(
                    name="belge-durumu",
                    content=content,
                    display="inline",
                )
            ]
        )
    except Exception as err:
        # Sidebar updates are best-effort and must not break chat flow.
        import traceback
        print(f"[ui] sidebar update failed: {err}", flush=True)
        traceback.print_exc()
        return


def _render_upload_progress(
    file_name: str,
    steps: list[str],
    in_progress: str | None,
    final_state: str | None = None,
    summary: str | None = None,
) -> str:
    if final_state == "done":
        title = f"**{file_name}** - ✅ TAMAMLANDI"
    elif final_state == "error":
        title = f"**{file_name}** - ❌ HATA"
    else:
        title = f"**{file_name}** isleniyor..."
    lines = [title, ""]
    for s in steps:
        if s == in_progress:
            lines.append(f"- [..] {s}")
        else:
            lines.append(f"- [x] {s}")
    if not steps and in_progress:
        lines.append(f"- [..] {in_progress}")
    if summary:
        lines.extend(["", "---", summary])
    return "\n".join(lines)


async def _stream_text_response(text: str, chunk_size: int = 24) -> None:
    """
    UI-only streaming: emit an already-generated response in small chunks.
    Does not change retrieval/generation logic.
    """
    msg = cl.Message(content="")
    await msg.send()
    if not text:
        await msg.update()
        return
    for i in range(0, len(text), chunk_size):
        await msg.stream_token(text[i:i + chunk_size])
    await msg.update()


def _build_qa_response(result, mode: str) -> str:
    answer = result.answer
    return f"{answer}{_build_qa_debug_suffix(result, mode)}"


def _build_qa_debug_suffix(result, mode: str) -> str:
    debug_lines = [
        f"- **Mod**: {mode}",
        f"- **Intent**: {result.intent}",
        f"- **Citation sayisi**: {result.citations_found}",
    ]
    if result.coverage_expected is not None:
        status_emoji = "OK" if result.coverage_ok else "EKSIK"
        debug_lines.append(
            f"- **Kapsam**: beklenen={result.coverage_expected}, "
            f"bulunan={result.coverage_actual}, durum={status_emoji}"
        )
    debug_text = "\n".join(debug_lines)
    return (
        f"\n\n"
        f"---\n"
        f"<details><summary>Debug Bilgisi</summary>\n\n"
        f"{debug_text}\n\n"
        f"</details>"
    )


def _format_standard_error(title: str, err: Exception | str) -> str:
    detail = re.sub(r"\s+", " ", str(err or "")).strip() or "Bilinmeyen hata"
    if len(detail) > 320:
        detail = detail[:320] + "..."
    return (
        f"**{title}**\n"
        f"- Islem tamamlanamadi.\n"
        f"- Detay: `{detail}`\n"
        f"- `Tekrar dene` ile ayni istegi yeniden calistirabilirsin."
    )


async def _send_standard_error(
    title: str,
    err: Exception | str,
    retry_payload: dict | None = None,
) -> None:
    actions = None
    if retry_payload:
        actions = [
            cl.Action(
                name="retry_last",
                payload=retry_payload,
                label="Tekrar dene",
                tooltip="Ayni istegi tekrar calistir",
                icon="refresh-cw",
            )
        ]
    await cl.Message(content=_format_standard_error(title, err), actions=actions).send()


async def _stream_doc_answer_live(
    pipeline: RAGPipeline,
    query: str,
    mode: str,
    thinking_msg: cl.Message | None = None,
):
    token_queue: SimpleQueue[str] = SimpleQueue()
    stream_msg: cl.Message | None = None

    def _on_token(token: str) -> None:
        token_queue.put(token)

    worker = asyncio.create_task(
        cl.make_async(pipeline.ask_stream)(query, _on_token)
    )

    streamed_chars = 0
    thinking_removed = False
    while not worker.done():
        while True:
            try:
                token = token_queue.get_nowait()
            except Empty:
                break
            if token:
                if thinking_msg and not thinking_removed:
                    try:
                        await thinking_msg.remove()
                    except Exception:
                        pass
                    thinking_removed = True
                if stream_msg is None:
                    stream_msg = cl.Message(content="")
                    await stream_msg.send()
                streamed_chars += len(token)
                await stream_msg.stream_token(token)
        await asyncio.sleep(0.03)

    while True:
        try:
            token = token_queue.get_nowait()
        except Empty:
            break
        if token:
            if thinking_msg and not thinking_removed:
                try:
                    await thinking_msg.remove()
                except Exception:
                    pass
                thinking_removed = True
            if stream_msg is None:
                stream_msg = cl.Message(content="")
                await stream_msg.send()
            streamed_chars += len(token)
            await stream_msg.stream_token(token)

    result = await worker
    if thinking_msg and not thinking_removed:
        try:
            await thinking_msg.remove()
        except Exception:
            pass
        thinking_removed = True
    if stream_msg is None:
        stream_msg = cl.Message(content="")
        await stream_msg.send()
    if streamed_chars == 0 and result.answer:
        await stream_msg.stream_token(result.answer)
    await stream_msg.stream_token(_build_qa_debug_suffix(result, mode))
    await stream_msg.update()
    return result


async def _process_uploaded_file_with_progress(file_path: str, file_name: str) -> str:
    pipeline: RAGPipeline | None = cl.user_session.get("pipeline")
    if pipeline is None:
        pipeline = _get_pipeline()

    progress_queue: SimpleQueue[str] = SimpleQueue()
    seen_steps: list[str] = []
    in_progress = "Dosya alindi, islem baslatiliyor..."

    def _on_progress(step: str) -> None:
        progress_queue.put(step)

    progress_msg = cl.Message(content=_render_upload_progress(file_name, seen_steps, in_progress))
    await progress_msg.send()

    worker = asyncio.create_task(
        cl.make_async(_process_uploaded_file_sync_with_progress)(
            file_path,
            file_name,
            pipeline,
            _on_progress,
        )
    )

    while not worker.done():
        updated = False
        while True:
            try:
                step = progress_queue.get_nowait()
            except Empty:
                break
            if not seen_steps or seen_steps[-1] != step:
                seen_steps.append(step)
                in_progress = step
                updated = True
        if updated:
            progress_msg.content = _render_upload_progress(file_name, seen_steps, in_progress)
            await progress_msg.update()
        await asyncio.sleep(0.25)

    while True:
        try:
            step = progress_queue.get_nowait()
        except Empty:
            break
        if not seen_steps or seen_steps[-1] != step:
            seen_steps.append(step)
            in_progress = step

    status = await worker
    status_lower = status.lower()
    is_error = "hata olustu" in status_lower or status_lower.startswith("hata:")
    final_state = "error" if is_error else "done"
    progress_msg.content = _render_upload_progress(
        file_name,
        seen_steps,
        None,
        final_state=final_state,
        summary=status,
    )
    await progress_msg.update()
    cl.user_session.set("pipeline", pipeline)
    return status


@cl.action_callback("retry_last")
async def on_retry_last(action: cl.Action):
    payload = action.payload or {}
    query = (payload.get("query") or "").strip()
    kind = (payload.get("kind") or "ask").strip().lower()
    mode = payload.get("mode") or (cl.user_session.get("mode") or "doc")
    chat_style = payload.get("chat_style") or _smalltalk_style(query)

    if not query:
        await cl.Message(content="Tekrar deneme icin sorgu bulunamadi. Lutfen tekrar yaz.").send()
        return

    try:
        await action.remove()
    except Exception:
        pass

    pipeline: RAGPipeline | None = cl.user_session.get("pipeline")
    if not pipeline:
        pipeline = _get_pipeline()
    await _sync_profile_to_pipeline(pipeline)

    thinking_msg = cl.Message(content="Tekrar deneniyor...")
    await thinking_msg.send()
    try:
        if kind == "chat":
            answer = await cl.make_async(pipeline.chat)(query, chat_style)
            await thinking_msg.remove()
            await _stream_text_response(answer)
            await _update_documents_sidebar(pipeline)
            return

        await _stream_doc_answer_live(pipeline, query, mode, thinking_msg=thinking_msg)
        await _update_documents_sidebar(pipeline)
    except Exception as e:
        await thinking_msg.remove()
        await _send_standard_error(
            "Tekrar deneme sirasinda hata",
            e,
            retry_payload=payload,
        )


@cl.on_message
async def on_message(message: cl.Message):
    pipeline: RAGPipeline | None = cl.user_session.get("pipeline")
    mode: str = cl.user_session.get("mode") or "doc"
    raw_query = (message.content or "").strip()
    hinted_thread_id, _ = _extract_thread_marker(raw_query) if raw_query else (None, "")
    if hinted_thread_id:
        cl.user_session.set("ui_thread_id", hinted_thread_id)
    active_ui_thread_id = (cl.user_session.get("ui_thread_id") or "").strip() or None
    if active_ui_thread_id:
        pipeline_for_thread = _thread_pipeline_get(active_ui_thread_id)
        if pipeline_for_thread is not None and pipeline_for_thread is not pipeline:
            pipeline = pipeline_for_thread
            cl.user_session.set("pipeline", pipeline)

    # Check for file attachments in the message
    if message.elements:
        if not pipeline:
            pipeline = _get_pipeline()
            cl.user_session.set("pipeline", pipeline)
        _thread_pipeline_set(active_ui_thread_id, pipeline)
        handled_any = False
        for elem in message.elements:
            file_path, file_name = _extract_uploaded_file_info(elem)
            if file_path:
                handled_any = True
                await _process_uploaded_file_with_progress(file_path, file_name)
                pipeline = cl.user_session.get("pipeline")
                _thread_pipeline_set(active_ui_thread_id, pipeline)
                await _update_documents_sidebar(pipeline)
        if not handled_any:
            await cl.Message(
                content=(
                    "Yuklenen dosya algılandı ancak dosya yolu okunamadı. "
                    "Lütfen dosyayı tekrar yükleyip bir kısa mesajla birlikte gönderin."
                )
            ).send()

    if not raw_query:
        return
    thread_id, query = _extract_thread_marker(raw_query)
    if thread_id:
        cl.user_session.set("ui_thread_id", thread_id)
    active_ui_thread_id = cl.user_session.get("ui_thread_id")
    if active_ui_thread_id:
        pipeline_for_thread = _thread_pipeline_get(active_ui_thread_id)
        if pipeline_for_thread is not None and pipeline_for_thread is not pipeline:
            pipeline = pipeline_for_thread
            cl.user_session.set("pipeline", pipeline)
    open_m = _OPEN_THREAD_CMD_RE.match(query)
    if open_m:
        forced_thread_id = (open_m.group(1) or "").strip()
        if forced_thread_id:
            cl.user_session.set("ui_thread_id", forced_thread_id)
            active_ui_thread_id = forced_thread_id
        if active_ui_thread_id:
            pipeline_for_thread = _thread_pipeline_get(active_ui_thread_id)
            if pipeline_for_thread is not None:
                pipeline = pipeline_for_thread
                cl.user_session.set("pipeline", pipeline)
        entries = _thread_memory_load((active_ui_thread_id or "").strip())
        if entries:
            for item in entries:
                role = item.get("role", "assistant")
                author = "Kullanici" if role == "user" else "Asistan"
                await cl.Message(content=item.get("content", ""), author=author).send()
        else:
            await cl.Message(content="Bu sohbete ait kayit bulunamadi.").send()
        await _update_documents_sidebar(pipeline)
        return
    if not query:
        return
    chat_style = _smalltalk_style(query)

    # Natural-language mode switches (work in any mode).
    if _looks_like_chat_mode_request(query) or query.strip().lower() in ("/chat", "/sohbet"):
        cl.user_session.set("mode", "chat")
        await cl.Message(content="Sohbet moduna geçildi. Belge soruları için `/doc` yazabilirsin.").send()
        await _update_documents_sidebar(pipeline)
        return
    if _looks_like_doc_mode_request(query) or query.strip().lower() in ("/doc", "/belge"):
        cl.user_session.set("mode", "doc")
        if pipeline and pipeline.has_documents:
            await cl.Message(content="Belge moduna geçildi. Belge sorunu sorabilirsin. (Sohbet için `/chat` yazabilirsin.)").send()
        else:
            await cl.Message(content="Belge moduna geçildi. Devam etmek için lütfen bir PDF/PNG/JPG yükle. (Sohbet için `/chat` yazabilirsin.)").send()
        await _update_documents_sidebar(pipeline)
        return

    if not query.startswith("/"):
        _append_chat_history_user_message(query)
        _thread_memory_add(active_ui_thread_id, "user", query)

    # Auto small-talk: answer conversational messages even in doc mode.
    if _looks_like_smalltalk(query):
        if not pipeline:
            pipeline = _get_pipeline()
        thinking_msg = cl.Message(content="Dusunuyorum...")
        await thinking_msg.send()
        try:
            answer = await cl.make_async(pipeline.chat)(query, chat_style)
        except Exception as e:
            await thinking_msg.remove()
            await _send_standard_error(
                "Sohbet cevabi olusturulamadi",
                e,
                retry_payload={
                    "kind": "chat",
                    "query": query,
                    "chat_style": chat_style,
                    "mode": cl.user_session.get("mode") or mode,
                },
            )
            return
        await thinking_msg.remove()
        await _stream_text_response(answer)
        _thread_memory_add(active_ui_thread_id, "assistant", answer)
        await _update_documents_sidebar(pipeline)
        return

    # Commands (document-agnostic)
    qlow = query.lower()
    if qlow.startswith("/use "):
        if not pipeline:
            pipeline = _get_pipeline()
        _thread_pipeline_set(active_ui_thread_id, pipeline)
        name = query[5:].strip()
        ok = pipeline.set_active_document(name)
        if ok:
            resolved = pipeline.active_document_name or name
            await cl.Message(content=f"Aktif belge ayarlandı: **{resolved}**").send()
        else:
            docs = pipeline.list_documents() if pipeline else []
            await cl.Message(content=f"Belge bulunamadı: **{name}**\nMevcut belgeler: {', '.join(docs) if docs else '(yok)'}").send()
        await _update_documents_sidebar(pipeline)
        return

    # Ensure pipeline exists
    if not pipeline:
        pipeline = _get_pipeline()
    await _sync_profile_to_pipeline(pipeline)
    _thread_pipeline_set(active_ui_thread_id, pipeline)

    # Chat mode does not require documents
    mode = cl.user_session.get("mode") or mode
    if mode == "chat":
        # If user is asking how to return to doc mode, switch and guide.
        if _looks_like_doc_mode_request(query):
            cl.user_session.set("mode", "doc")
            if pipeline.has_documents:
                await cl.Message(
                    content="Belge moduna geçildi. Belge sorunu sorabilirsin. (İstersen `/chat` ile tekrar sohbet moduna dönebilirsin.)"
                ).send()
            else:
                await cl.Message(
                    content="Belge moduna geçildi. Devam etmek için lütfen bir PDF/PNG/JPG yükle. (Sohbet için `/chat` yazabilirsin.)"
                ).send()
            await _update_documents_sidebar(pipeline)
            return

        # Auto-switch to doc if message clearly refers to a loaded document.
        if _looks_like_doc_switch(query, pipeline):
            cl.user_session.set("mode", "doc")
            mode = "doc"
            await _update_documents_sidebar(pipeline)
        else:
            thinking_msg = cl.Message(content="Dusunuyorum...")
            await thinking_msg.send()
            try:
                answer = await cl.make_async(pipeline.chat)(query, chat_style)
            except Exception as e:
                await thinking_msg.remove()
                await _send_standard_error(
                    "Sohbet cevabi olusturulamadi",
                    e,
                    retry_payload={
                        "kind": "chat",
                        "query": query,
                        "chat_style": chat_style,
                        "mode": cl.user_session.get("mode") or mode,
                    },
                )
                return
            await thinking_msg.remove()
            await _stream_text_response(answer)
            _thread_memory_add(active_ui_thread_id, "assistant", answer)
            await _update_documents_sidebar(pipeline)
            return

    mode = cl.user_session.get("mode") or mode
    # Doc mode requires documents
    if not pipeline.has_documents:
        no_doc_msg = "Henuz belge yuklenmedi. Lutfen once bir belge yukleyin. (Sohbet için `/chat` yazabilirsin.)"
        await cl.Message(content=no_doc_msg).send()
        _thread_memory_add(active_ui_thread_id, "assistant", no_doc_msg)
        await _update_documents_sidebar(pipeline)
        return
    if not pipeline.has_index:
        no_index_msg = (
            "Bu oturumda yuklenen belgelerden indeks olusturulamadi (metin cikarimi bos olabilir veya OCR/VLM gerekir).\n\n"
            "- PDF/PNG/JPG’yi tekrar yuklemeyi dene\n"
            "- Taranmis (image-only) PDF ise OCR kurulu oldugundan emin ol (README → OCR)\n"
            "- (Opsiyonel) VLM aciksa VLM_MAX_PAGES limitini kontrol et\n\n"
            "Sohbet için `/chat` yazabilirsin."
        )
        await cl.Message(content=no_index_msg).send()
        _thread_memory_add(active_ui_thread_id, "assistant", no_index_msg)
        await _update_documents_sidebar(pipeline)
        return

    # Show thinking indicator
    thinking_msg = cl.Message(content="Dusunuyorum...")
    await thinking_msg.send()

    # Generate and stream answer (real token stream when provider supports it)
    try:
        result = await _stream_doc_answer_live(pipeline, query, mode, thinking_msg=thinking_msg)
        _thread_memory_add(active_ui_thread_id, "assistant", result.answer)
        await _update_documents_sidebar(pipeline)
    except Exception as e:
        try:
            await thinking_msg.remove()
        except Exception:
            pass
        await _send_standard_error(
            "Belge cevabi olusturulamadi",
            e,
            retry_payload={
                "kind": "ask",
                "query": query,
                "mode": cl.user_session.get("mode") or mode,
            },
        )
        return
