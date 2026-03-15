from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
import os


LLMProvider = Literal["none", "openai", "gemini", "local"]
VLMProvider = Literal["gemini", "local"]
VLMMode = Literal["off", "auto", "force"]
OCRBackend = Literal["docai", "paddle_vl", "paddle", "tesseract_legacy"]
VisualChunkLevel = Literal["page", "region"]
VisualRegionSource = Literal["heuristic", "detector"]
VisualDetectorBackend = Literal["none", "sidecar", "docai", "docling"]
TableStructureBackend = Literal["off", "auto", "docai", "gemini", "heuristic"]
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
    ocr_backend: OCRBackend
    ocr_lang: str
    ocr_device: str
    paddle_ocr_version: str
    tesseract_cmd: Optional[str]
    tessdata_prefix: Optional[str]
    tesseract_config: Optional[str]
    docai_ocr_processor_id: str
    docai_ocr_processor_version: str

    # VLM (multimodal extract-only) controls
    vlm_mode: VLMMode
    vlm_max_pages: int
    vlm_provider: VLMProvider
    visual_chunk_level: VisualChunkLevel
    visual_region_source: VisualRegionSource
    visual_detector_backend: VisualDetectorBackend
    visual_detector_dir: Path
    docai_project_id: str
    docai_location: str
    docai_layout_processor_id: str
    docai_layout_processor_version: str
    docai_timeout_seconds: int
    docling_python_bin: str
    docling_layout_model: str
    docling_artifacts_path: Optional[Path]
    docling_device: str
    table_structure_enabled: bool
    table_structure_backend: TableStructureBackend
    table_structure_min_confidence: float
    table_structure_gemini_model: str
    docai_table_processor_id: str
    docai_table_processor_version: str

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
    ocr_backend_raw = (os.getenv("OCR_BACKEND", "docai") or "docai").strip().lower()
    ocr_backend: OCRBackend = (
        ocr_backend_raw  # type: ignore[assignment]
        if ocr_backend_raw in ("docai", "paddle_vl", "paddle", "tesseract_legacy")
        else "docai"
    )
    ocr_lang = (os.getenv("OCR_LANG", "tur+eng") or "tur+eng").strip()
    ocr_device_raw = (os.getenv("OCR_DEVICE", "auto") or "auto").strip().lower()
    ocr_device = ocr_device_raw if ocr_device_raw in ("auto", "cpu", "cuda") else "auto"
    paddle_ocr_version = (os.getenv("PADDLE_OCR_VERSION", "PP-OCRv5") or "PP-OCRv5").strip()
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
    visual_region_source_raw = (os.getenv("VISUAL_REGION_SOURCE", "heuristic") or "heuristic").strip().lower()
    visual_region_source: VisualRegionSource = (
        visual_region_source_raw  # type: ignore[assignment]
        if visual_region_source_raw in ("heuristic", "detector")
        else "heuristic"
    )
    visual_detector_backend_raw = (os.getenv("VISUAL_DETECTOR_BACKEND", "none") or "none").strip().lower()
    visual_detector_backend: VisualDetectorBackend = (
        visual_detector_backend_raw  # type: ignore[assignment]
        if visual_detector_backend_raw in ("none", "sidecar", "docai", "docling")
        else "none"
    )
    visual_detector_dir = Path(
        os.getenv("VISUAL_DETECTOR_DIR", str(data_dir / "layout_sidecars"))
    )
    docai_project_id = (os.getenv("DOCAI_PROJECT_ID", "") or os.getenv("VERTEX_PROJECT_ID", "") or os.getenv("GOOGLE_CLOUD_PROJECT", "")).strip()
    docai_location = (os.getenv("DOCAI_LOCATION", "") or "us").strip()
    docai_ocr_processor_id = (os.getenv("DOCAI_OCR_PROCESSOR_ID", "") or "").strip()
    docai_ocr_processor_version = (os.getenv("DOCAI_OCR_PROCESSOR_VERSION", "") or "").strip()
    docai_layout_processor_id = (os.getenv("DOCAI_LAYOUT_PROCESSOR_ID", "") or "").strip()
    docai_layout_processor_version = (
        os.getenv("DOCAI_LAYOUT_PROCESSOR_VERSION", "pretrained-layout-parser-v1.6-pro-2025-12-01") or ""
    ).strip()
    try:
        docai_timeout_seconds = int((os.getenv("DOCAI_TIMEOUT_SECONDS", "120") or "120").strip())
    except Exception:
        docai_timeout_seconds = 120
    docai_timeout_seconds = max(10, min(600, docai_timeout_seconds))
    docling_python_bin = (os.getenv("DOCLING_PYTHON_BIN", "") or "").strip()
    docling_layout_model = (os.getenv("DOCLING_LAYOUT_MODEL", "docling-layout-heron-101") or "docling-layout-heron-101").strip()
    docling_artifacts_raw = (os.getenv("DOCLING_ARTIFACTS_PATH", "") or "").strip()
    docling_artifacts_path = Path(docling_artifacts_raw) if docling_artifacts_raw else None
    docling_device_raw = (os.getenv("DOCLING_DEVICE", "auto") or "auto").strip().lower()
    docling_device = docling_device_raw if docling_device_raw in ("auto", "cpu", "cuda", "mps", "xpu") else "auto"
    table_structure_enabled_raw = (os.getenv("TABLE_STRUCTURE_ENABLED", "0") or "0").strip().lower()
    table_structure_enabled = table_structure_enabled_raw in ("1", "true", "yes", "y", "on")
    table_structure_backend_raw = (os.getenv("TABLE_STRUCTURE_BACKEND", "auto") or "auto").strip().lower()
    table_structure_backend: TableStructureBackend = (
        table_structure_backend_raw  # type: ignore[assignment]
        if table_structure_backend_raw in ("off", "auto", "docai", "gemini", "heuristic")
        else "auto"
    )
    try:
        table_structure_min_confidence = float((os.getenv("TABLE_STRUCTURE_MIN_CONFIDENCE", "0.55") or "0.55").strip())
    except Exception:
        table_structure_min_confidence = 0.55
    table_structure_min_confidence = max(0.0, min(1.0, table_structure_min_confidence))
    table_structure_gemini_model = (
        os.getenv("TABLE_STRUCTURE_GEMINI_MODEL", "") or os.getenv("GEMINI_MODEL", "gemini-2.0-flash") or "gemini-2.0-flash"
    ).strip()
    docai_table_processor_id = (os.getenv("DOCAI_TABLE_PROCESSOR_ID", "") or "").strip()
    docai_table_processor_version = (os.getenv("DOCAI_TABLE_PROCESSOR_VERSION", "") or "").strip()

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
        ocr_backend=ocr_backend,
        ocr_lang=ocr_lang,
        ocr_device=ocr_device,
        paddle_ocr_version=paddle_ocr_version,
        tesseract_cmd=tesseract_cmd,
        tessdata_prefix=tessdata_prefix,
        tesseract_config=tesseract_config,
        docai_ocr_processor_id=docai_ocr_processor_id,
        docai_ocr_processor_version=docai_ocr_processor_version,
        vlm_mode=vlm_mode,
        vlm_max_pages=vlm_max_pages,
        vlm_provider=vlm_provider,
        visual_chunk_level=visual_chunk_level,
        visual_region_source=visual_region_source,
        visual_detector_backend=visual_detector_backend,
        visual_detector_dir=visual_detector_dir,
        docai_project_id=docai_project_id,
        docai_location=docai_location,
        docai_layout_processor_id=docai_layout_processor_id,
        docai_layout_processor_version=docai_layout_processor_version,
        docai_timeout_seconds=docai_timeout_seconds,
        docling_python_bin=docling_python_bin,
        docling_layout_model=docling_layout_model,
        docling_artifacts_path=docling_artifacts_path,
        docling_device=docling_device,
        table_structure_enabled=table_structure_enabled,
        table_structure_backend=table_structure_backend,
        table_structure_min_confidence=table_structure_min_confidence,
        table_structure_gemini_model=table_structure_gemini_model,
        docai_table_processor_id=docai_table_processor_id,
        docai_table_processor_version=docai_table_processor_version,
        ollama_base_url=ollama_base_url,
        ollama_llm_model=ollama_llm_model,
        ollama_vlm_model=ollama_vlm_model,
        ollama_timeout=ollama_timeout,
    )
