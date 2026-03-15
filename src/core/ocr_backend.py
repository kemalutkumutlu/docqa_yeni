from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Optional

from PIL import Image


@dataclass(frozen=True)
class OCRConfig:
    enabled: bool = True
    backend: str = "tesseract_legacy"
    lang: str = "tur+eng"
    device: str = "auto"
    paddle_ocr_version: str = "PP-OCRv5"
    tesseract_cmd: Optional[str] = None
    tessdata_prefix: Optional[str] = None
    tesseract_config: Optional[str] = None
    docai_project_id: str = ""
    docai_location: str = "us"
    docai_processor_id: str = ""
    docai_processor_version: str = ""
    docai_timeout_seconds: int = 120


@dataclass(frozen=True)
class OCRResult:
    text: str = ""
    source: str = "ocr"
    warnings: list[str] = field(default_factory=list)


def _find_local_libgomp_dir() -> Optional[Path]:
    vendor_root = Path(__file__).resolve().parents[2] / ".vendor" / "libgomp"
    if not vendor_root.exists():
        return None
    for candidate in sorted(vendor_root.glob("**/libgomp.so*")):
        if candidate.is_file():
            return candidate.parent
    return None


def configure_ocr_backend(cfg: OCRConfig) -> None:
    if (cfg.backend or "").strip().lower() != "tesseract_legacy":
        return
    if not cfg.tesseract_cmd:
        if cfg.tessdata_prefix:
            os.environ["TESSDATA_PREFIX"] = cfg.tessdata_prefix
        return
    try:
        import pytesseract  # noqa: WPS433

        pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd
        if cfg.tessdata_prefix:
            os.environ["TESSDATA_PREFIX"] = cfg.tessdata_prefix
    except Exception:
        return


def ocr_image_text(
    img: Image.Image,
    *,
    cfg: OCRConfig,
    document_name: str,
    page_number: int,
    image_kind: str,
) -> OCRResult:
    backend = (cfg.backend or "tesseract_legacy").strip().lower()
    if backend == "docai":
        return _ocr_with_docai_or_fallback(
            img,
            cfg=cfg,
            document_name=document_name,
            page_number=page_number,
            image_kind=image_kind,
        )
    if backend in ("paddle_vl", "paddle"):
        return _ocr_with_paddle_or_fallback(
            img,
            cfg=cfg,
            backend=backend,
            document_name=document_name,
            page_number=page_number,
            image_kind=image_kind,
        )
    return _ocr_with_tesseract(img, cfg=cfg, image_kind=image_kind)


def _ocr_with_tesseract(img: Image.Image, *, cfg: OCRConfig, image_kind: str) -> OCRResult:
    import pytesseract  # noqa: WPS433

    config = (cfg.tesseract_config or "").strip()
    text = pytesseract.image_to_string(img, lang=cfg.lang, config=config) or ""
    source = "image_ocr" if image_kind == "image" else "ocr"
    return OCRResult(text=text, source=source, warnings=[])


def _ocr_with_docai_or_fallback(
    img: Image.Image,
    *,
    cfg: OCRConfig,
    document_name: str,
    page_number: int,
    image_kind: str,
) -> OCRResult:
    try:
        text = _ocr_with_docai(
            img,
            cfg=cfg,
            document_name=document_name,
            page_number=page_number,
        )
        source = "docai_image_ocr" if image_kind == "image" else "docai_ocr"
        return OCRResult(text=text, source=source, warnings=[])
    except Exception as exc:
        fallback = _try_legacy_fallback(
            img,
            cfg=cfg,
            image_kind=image_kind,
            warning=(
                f"Document AI OCR kullanilamadi ({document_name} page {page_number}): {exc}. "
                "Tesseract legacy fallback denendi."
            ),
        )
        if fallback is not None:
            return fallback
        source = "docai_image_ocr" if image_kind == "image" else "docai_ocr"
        return OCRResult(text="", source=source, warnings=[str(exc)])


def _ocr_with_paddle_or_fallback(
    img: Image.Image,
    *,
    cfg: OCRConfig,
    backend: str,
    document_name: str,
    page_number: int,
    image_kind: str,
) -> OCRResult:
    try:
        text = _ocr_with_paddle(
            img,
            cfg=cfg,
            backend=backend,
        )
        source = "paddle_vl_ocr" if backend == "paddle_vl" else "paddle_ocr"
        return OCRResult(text=text, source=source, warnings=[])
    except Exception as exc:
        label = "PaddleOCR-VL-1.5" if backend == "paddle_vl" else "PaddleOCR"
        if backend == "paddle_vl":
            light_result, light_exc = _try_light_paddle_fallback(
                img,
                cfg=cfg,
                image_kind=image_kind,
                warning=(
                    f"{label} kullanilamadi ({document_name} page {page_number}): {exc}. "
                    "Standart PaddleOCR fallback denendi."
                ),
            )
            if light_result is not None:
                return light_result
            detail = f"{exc}"
            if light_exc is not None:
                detail += f" | PaddleOCR fallback da basarisiz: {light_exc}"
            fallback = _try_legacy_fallback(
                img,
                cfg=cfg,
                image_kind=image_kind,
                warning=(
                    f"{label} kullanilamadi ({document_name} page {page_number}): {detail}. "
                    "Tesseract legacy fallback denendi."
                ),
            )
            if fallback is not None:
                return fallback
            return OCRResult(text="", source="paddle_vl_ocr", warnings=[detail])
        fallback = _try_legacy_fallback(
            img,
            cfg=cfg,
            image_kind=image_kind,
            warning=(
                f"{label} kullanilamadi ({document_name} page {page_number}): {exc}. "
                "Tesseract legacy fallback denendi."
            ),
        )
        if fallback is not None:
            return fallback
        source = "paddle_vl_ocr" if backend == "paddle_vl" else "paddle_ocr"
        return OCRResult(text="", source=source, warnings=[str(exc)])


def _try_light_paddle_fallback(
    img: Image.Image,
    *,
    cfg: OCRConfig,
    image_kind: str,
    warning: str,
) -> tuple[OCRResult | None, Exception | None]:
    try:
        text = _ocr_with_paddle(img, cfg=cfg, backend="paddle")
        return OCRResult(
            text=text,
            source="paddle_ocr",
            warnings=[warning],
        ), None
    except Exception as exc:
        return None, exc


def _try_legacy_fallback(
    img: Image.Image,
    *,
    cfg: OCRConfig,
    image_kind: str,
    warning: str,
) -> OCRResult | None:
    legacy_cfg = OCRConfig(
        enabled=cfg.enabled,
        backend="tesseract_legacy",
        lang=cfg.lang,
        device=cfg.device,
        paddle_ocr_version=cfg.paddle_ocr_version,
        tesseract_cmd=cfg.tesseract_cmd,
        tessdata_prefix=cfg.tessdata_prefix,
        tesseract_config=cfg.tesseract_config,
    )
    try:
        configure_ocr_backend(legacy_cfg)
        result = _ocr_with_tesseract(img, cfg=legacy_cfg, image_kind=image_kind)
        return OCRResult(text=result.text, source=result.source, warnings=[warning])
    except Exception as exc:
        return OCRResult(text="", source=("image_ocr" if image_kind == "image" else "ocr"), warnings=[warning, f"Tesseract legacy fallback da basarisiz: {exc}"])


def _resolve_docai_project_id(explicit: str) -> str:
    if explicit:
        return explicit
    for name in ("DOCAI_PROJECT_ID", "VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCP_PROJECT"):
        value = (os.getenv(name, "") or "").strip()
        if value:
            return value
    try:
        import google.auth

        _, project_id = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if project_id:
            return project_id
    except Exception:
        pass
    raise ValueError("Document AI OCR icin project id bulunamadi. DOCAI_PROJECT_ID ayarlayin.")


def _ocr_with_paddle(
    img: Image.Image,
    *,
    cfg: OCRConfig,
    backend: str,
) -> str:
    runner = Path(__file__).resolve().parents[2] / "scripts" / "paddle_ocr_runner.py"
    if not runner.exists():
        raise RuntimeError(f"Paddle OCR runner bulunamadi: {runner}")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        img.save(temp_path, format="PNG")
        cmd = [
            sys.executable,
            str(runner),
            "--image",
            str(temp_path),
            "--backend",
            backend,
            "--lang",
            cfg.lang,
            "--device",
            cfg.device,
            "--ocr-version",
            cfg.paddle_ocr_version or "PP-OCRv5",
        ]
        env = {**os.environ, "PYTHONNOUSERSITE": "1"}
        env.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        libgomp_dir = _find_local_libgomp_dir()
        if libgomp_dir is not None:
            current_ld_path = env.get("LD_LIBRARY_PATH", "").strip()
            env["LD_LIBRARY_PATH"] = (
                f"{libgomp_dir}{os.pathsep}{current_ld_path}" if current_ld_path else str(libgomp_dir)
            )
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            timeout=180,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            detail_lines = [line.strip() for line in (stderr.splitlines() + stdout.splitlines()) if line.strip()]
            detail = detail_lines[-1] if detail_lines else ""
            raise RuntimeError(
                (f"Paddle OCR runner basarisiz cikti (code={proc.returncode})" + (f": {detail}" if detail else ""))
            )
        try:
            payload = json.loads((proc.stdout or "").strip() or "{}")
        except Exception as exc:
            raise RuntimeError(f"Paddle OCR runner ciktisi parse edilemedi: {exc}") from exc
        if not payload.get("ok"):
            raise RuntimeError(str(payload.get("error", "Paddle OCR runner basarisiz")))
        text = str(payload.get("text", "") or "").strip()
        if not text:
            raise RuntimeError("Paddle OCR bos sonuc dondurdu")
        return text
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _ocr_with_docai(
    img: Image.Image,
    *,
    cfg: OCRConfig,
    document_name: str,
    page_number: int,
) -> str:
    if not (cfg.docai_processor_id or "").strip():
        raise ValueError("DOCAI_OCR_PROCESSOR_ID ayarlanmamis")
    try:
        from google.api_core.client_options import ClientOptions
        from google.cloud import documentai
    except Exception as exc:
        raise RuntimeError(f"google-cloud-documentai import edilemedi: {exc}") from exc

    project_id = _resolve_docai_project_id((cfg.docai_project_id or "").strip())
    location = (cfg.docai_location or "us").strip()
    processor_id = (cfg.docai_processor_id or "").strip()
    processor_version = (cfg.docai_processor_version or "").strip()
    endpoint = f"{location}-documentai.googleapis.com"
    client = documentai.DocumentProcessorServiceClient(client_options=ClientOptions(api_endpoint=endpoint))
    name = (
        client.processor_version_path(project_id, location, processor_id, processor_version)
        if processor_version
        else client.processor_path(project_id, location, processor_id)
    )

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=buffer.getvalue(), mime_type="image/png"),
    )
    result = client.process_document(request=request, timeout=max(10, min(600, int(cfg.docai_timeout_seconds or 120))))
    text = (getattr(getattr(result, "document", None), "text", "") or "").strip()
    if not text:
        raise RuntimeError(f"Document AI OCR bos sonuc dondurdu ({document_name} page {page_number})")
    return text
