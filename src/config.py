from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
import os


LLMProvider = Literal["none", "openai", "gemini", "local"]
VLMProvider = Literal["gemini", "local"]
VLMMode = Literal["off", "auto", "force"]
VisualChunkLevel = Literal["page", "region"]
EmbeddingDevice = Literal["auto", "cpu", "cuda"]
ProcessingMode = Literal["classic", "multimodal"]
MultimodalAnswerMode = Literal["off", "auto", "on"]


@dataclass(frozen=True)
class Settings:
    llm_provider: LLMProvider
    openai_api_key: str
    openai_model: str
    gemini_api_key: str
    gemini_model: str

    embedding_model: str
    embedding_device: EmbeddingDevice
    processing_mode: ProcessingMode
    multimodal_answer_mode: MultimodalAnswerMode

    data_dir: Path
    chroma_dir: Path

    ocr_enabled: bool
    tesseract_cmd: Optional[str]
    tessdata_prefix: Optional[str]
    tesseract_config: Optional[str]

    # VLM (multimodal extract-only) controls
    vlm_mode: VLMMode
    vlm_max_pages: int
    vlm_provider: VLMProvider
    visual_chunk_level: VisualChunkLevel

    # Ollama (local LLM/VLM) settings
    ollama_base_url: str
    ollama_llm_model: str
    ollama_vlm_model: str
    ollama_timeout: int


def load_settings() -> Settings:
    # Load .env if present (dev-friendly)
    load_dotenv(override=False)

    def _cuda_available() -> bool:
        try:
            import torch  # noqa: WPS433

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    llm_provider: LLMProvider = os.getenv("LLM_PROVIDER", "none").strip().lower()  # type: ignore
    if llm_provider not in ("none", "openai", "gemini", "local"):
        llm_provider = "none"

    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    chroma_dir = Path(os.getenv("CHROMA_DIR", str(data_dir / "chroma")))

    ocr_enabled_raw = (os.getenv("OCR_ENABLED", "1") or "1").strip().lower()
    ocr_enabled = ocr_enabled_raw in ("1", "true", "yes", "y", "on")
    tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip() or None
    tessdata_prefix = os.getenv("TESSDATA_PREFIX", "").strip() or None
    tesseract_config = os.getenv("TESSERACT_CONFIG", "").strip() or None

    # VLM: keep UI behavior by default (force, 25 pages), but allow env override.
    vlm_mode_raw = os.getenv("VLM_MODE", "force").strip().lower()
    vlm_mode: VLMMode = "force"
    if vlm_mode_raw in ("off", "auto", "force"):
        vlm_mode = vlm_mode_raw  # type: ignore[assignment]

    try:
        vlm_max_pages = int(os.getenv("VLM_MAX_PAGES", "25").strip())
    except Exception:
        vlm_max_pages = 25
    # Safety clamp (avoid accidental huge costs)
    vlm_max_pages = max(0, min(200, vlm_max_pages))

    vlm_provider_raw = os.getenv("VLM_PROVIDER", "gemini").strip().lower()
    vlm_provider: VLMProvider = vlm_provider_raw if vlm_provider_raw in ("gemini", "local") else "gemini"  # type: ignore[assignment]
    visual_chunk_level_raw = (os.getenv("VISUAL_CHUNK_LEVEL", "page") or "page").strip().lower()
    visual_chunk_level: VisualChunkLevel = (
        visual_chunk_level_raw  # type: ignore[assignment]
        if visual_chunk_level_raw in ("page", "region")
        else "page"
    )

    # Ollama settings (only used when LLM_PROVIDER=local or VLM_PROVIDER=local)
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    ollama_llm_model = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b").strip()
    ollama_vlm_model = os.getenv("OLLAMA_VLM_MODEL", "llava:7b").strip()
    try:
        ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "120").strip())
    except Exception:
        ollama_timeout = 120

    embedding_device_raw = (os.getenv("EMBEDDING_DEVICE", "auto") or "auto").strip().lower()
    embedding_device: EmbeddingDevice = (
        embedding_device_raw  # type: ignore[assignment]
        if embedding_device_raw in ("auto", "cpu", "cuda")
        else "auto"
    )

    processing_mode_raw = (os.getenv("DOC_PROCESSING_MODE", "classic") or "classic").strip().lower()
    processing_mode: ProcessingMode = (
        processing_mode_raw  # type: ignore[assignment]
        if processing_mode_raw in ("classic", "multimodal")
        else "classic"
    )

    multimodal_answer_mode_raw = (os.getenv("MULTIMODAL_ANSWER_MODE", "auto") or "auto").strip().lower()
    multimodal_answer_mode: MultimodalAnswerMode = (
        multimodal_answer_mode_raw  # type: ignore[assignment]
        if multimodal_answer_mode_raw in ("off", "auto", "on")
        else "auto"
    )

    # Embedding model selection:
    # - Default: Gemini embedding for highest remote quality.
    # - "auto" keeps the previous local behavior:
    #     - CUDA available (and not forced cpu) -> multilingual-e5-base
    #     - otherwise -> multilingual-e5-small
    embedding_model_raw = (os.getenv("EMBEDDING_MODEL", "gemini-embedding-001") or "gemini-embedding-001").strip()
    if embedding_model_raw.lower() == "auto":
        cuda_ok = _cuda_available() and embedding_device != "cpu"
        embedding_model = "intfloat/multilingual-e5-base" if cuda_ok else "intfloat/multilingual-e5-small"
    else:
        embedding_model = embedding_model_raw

    return Settings(
        llm_provider=llm_provider,
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        embedding_model=embedding_model,
        embedding_device=embedding_device,
        processing_mode=processing_mode,
        multimodal_answer_mode=multimodal_answer_mode,
        data_dir=data_dir,
        chroma_dir=chroma_dir,
        ocr_enabled=ocr_enabled,
        tesseract_cmd=tesseract_cmd,
        tessdata_prefix=tessdata_prefix,
        tesseract_config=tesseract_config,
        vlm_mode=vlm_mode,
        vlm_max_pages=vlm_max_pages,
        vlm_provider=vlm_provider,
        visual_chunk_level=visual_chunk_level,
        ollama_base_url=ollama_base_url,
        ollama_llm_model=ollama_llm_model,
        ollama_vlm_model=ollama_vlm_model,
        ollama_timeout=ollama_timeout,
    )
