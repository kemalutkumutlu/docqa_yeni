from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Protocol, Sequence

import google.auth
from PIL import Image


@dataclass(frozen=True)
class DetectorRegion:
    bbox_left: int
    bbox_top: int
    bbox_right: int
    bbox_bottom: int
    label: str = ""
    crop_type: str = "detector_region"
    confidence: float = 0.0
    summary_text: str = ""


@dataclass(frozen=True)
class DetectorResult:
    regions: list[DetectorRegion] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class LayoutDetector(Protocol):
    name: str

    def detect(
        self,
        image: Image.Image,
        *,
        page_number: int,
        summary_text: str,
        document_name: str,
    ) -> DetectorResult:
        ...


class UnavailableLayoutDetector:
    name = "unavailable"

    def detect(
        self,
        image: Image.Image,
        *,
        page_number: int,
        summary_text: str,
        document_name: str,
    ) -> DetectorResult:
        del image, page_number, summary_text, document_name
        return DetectorResult(
            regions=[],
            warnings=[
                "VISUAL_REGION_SOURCE=detector secili, ancak bbox detector henuz entegre degil; heuristic fallback kullanildi."
            ],
        )


def resolve_sidecar_path(detector_dir: Path | None, document_name: str) -> Path | None:
    if detector_dir is None:
        return None
    base = Path(document_name).name
    stem = Path(base).stem
    candidates = [
        detector_dir / f"{base}.regions.json",
        detector_dir / f"{stem}.regions.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


class SidecarLayoutDetector:
    name = "sidecar"

    def __init__(self, detector_dir: Path | None) -> None:
        self.detector_dir = detector_dir

    @staticmethod
    def _region_from_row(row: dict) -> DetectorRegion | None:
        bbox = row.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            left, top, right, bottom = bbox
        else:
            left = row.get("bbox_left", 0)
            top = row.get("bbox_top", 0)
            right = row.get("bbox_right", 0)
            bottom = row.get("bbox_bottom", 0)
        try:
            left_i = int(left)
            top_i = int(top)
            right_i = int(right)
            bottom_i = int(bottom)
        except Exception:
            return None
        if right_i <= left_i or bottom_i <= top_i:
            return None
        return DetectorRegion(
            bbox_left=left_i,
            bbox_top=top_i,
            bbox_right=right_i,
            bbox_bottom=bottom_i,
            label=str(row.get("label", "") or "").strip(),
            crop_type=str(row.get("crop_type", "detector_region") or "detector_region").strip(),
            confidence=max(0.0, min(1.0, float(row.get("confidence", 0.0) or 0.0))),
            summary_text=str(row.get("summary_text", "") or "").strip(),
        )

    def detect(
        self,
        image: Image.Image,
        *,
        page_number: int,
        summary_text: str,
        document_name: str,
    ) -> DetectorResult:
        del image, summary_text
        sidecar_path = resolve_sidecar_path(self.detector_dir, document_name)
        if sidecar_path is None:
            return DetectorResult(
                regions=[],
                warnings=["VISUAL_DETECTOR_BACKEND=sidecar secili, ancak detector dizini ayarlanmamis."],
            )
        if not sidecar_path.exists():
            return DetectorResult(
                regions=[],
                warnings=[f"Detector sidecar bulunamadi: {sidecar_path.name}; heuristic fallback kullanildi."],
            )
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return DetectorResult(
                regions=[],
                warnings=[f"Detector sidecar okunamadi ({sidecar_path.name}): {exc}; heuristic fallback kullanildi."],
            )

        rows: list[dict] = []
        if isinstance(payload, dict) and isinstance(payload.get("pages"), dict):
            rows = payload.get("pages", {}).get(str(page_number), []) or []
        elif isinstance(payload, dict) and isinstance(payload.get("regions"), list):
            rows = [row for row in payload.get("regions", []) if int(row.get("page", 0) or 0) == page_number]
        elif isinstance(payload, list):
            rows = [row for row in payload if isinstance(row, dict) and int(row.get("page", 0) or 0) == page_number]

        regions = [region for row in rows if isinstance(row, dict) for region in [self._region_from_row(row)] if region is not None]
        if not regions:
            return DetectorResult(
                regions=[],
                warnings=[f"Detector sidecar icinde page={page_number} icin bbox bulunamadi; heuristic fallback kullanildi."],
            )
        return DetectorResult(regions=regions, warnings=[])


def _extract_text(text: str, text_anchor) -> str:
    if not text_anchor or not getattr(text_anchor, "text_segments", None):
        return ""
    parts: list[str] = []
    for segment in text_anchor.text_segments:
        start = int(getattr(segment, "start_index", 0) or 0)
        end = int(getattr(segment, "end_index", 0) or 0)
        if end > start:
            parts.append(text[start:end])
    return " ".join(part.strip() for part in parts if part and part.strip()).strip()


def _bbox_from_layout(layout, *, page_width: int, page_height: int, image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    if not layout:
        return None
    poly = getattr(layout, "bounding_poly", None)
    if not poly:
        return None
    normalized = getattr(poly, "normalized_vertices", None) or []
    if normalized:
        xs = [float(vertex.x or 0.0) * image_width for vertex in normalized]
        ys = [float(vertex.y or 0.0) * image_height for vertex in normalized]
    else:
        vertices = getattr(poly, "vertices", None) or []
        if not vertices:
            return None
        scale_x = image_width / max(1, page_width)
        scale_y = image_height / max(1, page_height)
        xs = [float(getattr(vertex, "x", 0) or 0) * scale_x for vertex in vertices]
        ys = [float(getattr(vertex, "y", 0) or 0) * scale_y for vertex in vertices]
    left = max(0, int(min(xs)))
    top = max(0, int(min(ys)))
    right = min(image_width, int(max(xs)))
    bottom = min(image_height, int(max(ys)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


class DocAILayoutDetector:
    name = "docai"

    def __init__(
        self,
        *,
        project_id: str,
        location: str,
        processor_id: str,
        processor_version: str,
        timeout_seconds: int,
    ) -> None:
        self.project_id = (project_id or "").strip()
        self.location = (location or "us").strip()
        self.processor_id = (processor_id or "").strip()
        self.processor_version = (processor_version or "").strip()
        self.timeout_seconds = max(10, min(600, int(timeout_seconds or 120)))

    @staticmethod
    def _resolve_project_id(explicit: str) -> str:
        if explicit:
            return explicit
        for name in ("VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCP_PROJECT"):
            value = (os.getenv(name, "") or "").strip()
            if value:
                return value
        _, project_id = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if project_id:
            return project_id
        raise ValueError("Document AI icin project id bulunamadi. DOCAI_PROJECT_ID veya GOOGLE_CLOUD_PROJECT ayarlayin.")

    @staticmethod
    def _candidate_regions(page, full_text: str, *, image_width: int, image_height: int) -> list[DetectorRegion]:
        page_dim = getattr(page, "dimension", None)
        page_width = int(getattr(page_dim, "width", image_width) or image_width)
        page_height = int(getattr(page_dim, "height", image_height) or image_height)
        rows: list[DetectorRegion] = []

        for table in getattr(page, "tables", None) or []:
            bbox = _bbox_from_layout(getattr(table, "layout", None), page_width=page_width, page_height=page_height, image_width=image_width, image_height=image_height)
            if not bbox:
                continue
            summary = _extract_text(full_text, getattr(getattr(table, "layout", None), "text_anchor", None))
            rows.append(
                DetectorRegion(
                    bbox_left=bbox[0],
                    bbox_top=bbox[1],
                    bbox_right=bbox[2],
                    bbox_bottom=bbox[3],
                    label="table",
                    crop_type="docai_table",
                    confidence=float(getattr(getattr(table, "layout", None), "confidence", 0.0) or 0.0),
                    summary_text=summary,
                )
            )

        for block in getattr(page, "blocks", None) or []:
            bbox = _bbox_from_layout(getattr(block, "layout", None), page_width=page_width, page_height=page_height, image_width=image_width, image_height=image_height)
            if not bbox:
                continue
            summary = _extract_text(full_text, getattr(getattr(block, "layout", None), "text_anchor", None))
            if not summary:
                continue
            rows.append(
                DetectorRegion(
                    bbox_left=bbox[0],
                    bbox_top=bbox[1],
                    bbox_right=bbox[2],
                    bbox_bottom=bbox[3],
                    label="block",
                    crop_type="docai_block",
                    confidence=float(getattr(getattr(block, "layout", None), "confidence", 0.0) or 0.0),
                    summary_text=summary,
                )
            )

        rows.sort(key=lambda row: (row.bbox_top, row.bbox_left))
        return rows[:10]

    def detect(
        self,
        image: Image.Image,
        *,
        page_number: int,
        summary_text: str,
        document_name: str,
    ) -> DetectorResult:
        del page_number, summary_text, document_name
        if not self.processor_id:
            return DetectorResult(
                regions=[],
                warnings=["VISUAL_DETECTOR_BACKEND=docai secili, ancak DOCAI_LAYOUT_PROCESSOR_ID tanimli degil; heuristic fallback kullanildi."],
            )
        try:
            from google.api_core.client_options import ClientOptions
            from google.cloud import documentai
        except Exception as exc:
            return DetectorResult(
                regions=[],
                warnings=[f"google-cloud-documentai kurulu degil veya import edilemedi ({exc}); heuristic fallback kullanildi."],
            )

        try:
            project_id = self._resolve_project_id(self.project_id)
        except Exception as exc:
            return DetectorResult(regions=[], warnings=[f"Document AI project id cozulmedi ({exc}); heuristic fallback kullanildi."])

        try:
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="PNG")
            content = image_bytes.getvalue()
            client = documentai.DocumentProcessorServiceClient(
                client_options=ClientOptions(api_endpoint=f"{self.location}-documentai.googleapis.com")
            )
            if self.processor_version:
                name = client.processor_version_path(project_id, self.location, self.processor_id, self.processor_version)
            else:
                name = client.processor_path(project_id, self.location, self.processor_id)
            raw_document = documentai.RawDocument(content=content, mime_type="image/png")
            request = documentai.ProcessRequest(name=name, raw_document=raw_document)
            result = client.process_document(request=request, timeout=self.timeout_seconds)
            document = result.document
            pages = getattr(document, "pages", None) or []
            if not pages:
                return DetectorResult(regions=[], warnings=["Document AI sonucunda sayfa layout verisi donmedi; heuristic fallback kullanildi."])
            regions = self._candidate_regions(
                pages[0],
                getattr(document, "text", "") or "",
                image_width=image.width,
                image_height=image.height,
            )
            if not regions:
                return DetectorResult(regions=[], warnings=["Document AI bbox bolgesi uretemedi; heuristic fallback kullanildi."])
            return DetectorResult(regions=regions, warnings=[])
        except Exception as exc:
            return DetectorResult(
                regions=[],
                warnings=[f"Document AI layout parser cagrisi basarisiz ({exc}); heuristic fallback kullanildi."],
            )


class DoclingLayoutDetector:
    name = "docling"

    def __init__(
        self,
        *,
        python_bin: str,
        model_name: str,
        artifacts_path: Path | None,
        device: str,
    ) -> None:
        self.python_bin = (python_bin or "").strip() or sys.executable
        self.model_name = (model_name or "docling-layout-heron-101").strip() or "docling-layout-heron-101"
        self.artifacts_path = artifacts_path
        self.device = (device or "auto").strip().lower() or "auto"

    @staticmethod
    def _runner_path() -> Path:
        return Path(__file__).resolve().parents[2] / "scripts" / "docling_layout_runner.py"

    @staticmethod
    def _region_from_row(row: dict) -> DetectorRegion | None:
        try:
            left = int(row.get("bbox_left", 0) or 0)
            top = int(row.get("bbox_top", 0) or 0)
            right = int(row.get("bbox_right", 0) or 0)
            bottom = int(row.get("bbox_bottom", 0) or 0)
        except Exception:
            return None
        if right <= left or bottom <= top:
            return None
        return DetectorRegion(
            bbox_left=left,
            bbox_top=top,
            bbox_right=right,
            bbox_bottom=bottom,
            label=str(row.get("label", "") or "").strip(),
            crop_type=str(row.get("crop_type", "docling_region") or "docling_region").strip(),
            confidence=max(0.0, min(1.0, float(row.get("confidence", 0.0) or 0.0))),
            summary_text=str(row.get("summary_text", "") or "").strip(),
        )

    @staticmethod
    def _build_subprocess_env(*, python_bin: str, model_name: str, device: str, artifacts_path: Path | None) -> dict[str, str]:
        env: dict[str, str] = {}
        pass_through = (
            "HOME",
            "USER",
            "LOGNAME",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "TERM",
            "TMPDIR",
            "TMP",
            "TEMP",
            "SSL_CERT_FILE",
            "REQUESTS_CA_BUNDLE",
            "CURL_CA_BUNDLE",
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "NO_PROXY",
            "ALL_PROXY",
            "XDG_CACHE_HOME",
            "HF_HOME",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "CUDA_VISIBLE_DEVICES",
            "LD_LIBRARY_PATH",
        )
        for env_name in pass_through:
            value = os.environ.get(env_name)
            if value:
                env[env_name] = value
        python_dir = str(Path(python_bin).resolve().parent)
        env["PATH"] = python_dir + os.pathsep + os.environ.get("PATH", "")
        env["PYTHONNOUSERSITE"] = "1"
        env["DOCLING_LAYOUT_MODEL"] = model_name
        env["DOCLING_DEVICE"] = device
        if artifacts_path is not None:
            env["DOCLING_ARTIFACTS_PATH"] = str(artifacts_path)
        return env

    def detect(
        self,
        image: Image.Image,
        *,
        page_number: int,
        summary_text: str,
        document_name: str,
    ) -> DetectorResult:
        del page_number, summary_text
        runner_path = self._runner_path()
        if not runner_path.exists():
            return DetectorResult(
                regions=[],
                warnings=[f"Docling layout runner bulunamadi ({runner_path.name}); heuristic fallback kullanildi."],
            )

        with tempfile.TemporaryDirectory(prefix="docling_layout_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "page.png"
            output_path = tmp_path / "regions.json"
            try:
                image.save(image_path, format="PNG")
            except Exception as exc:
                return DetectorResult(
                    regions=[],
                    warnings=[f"Docling layout icin gecici image yazilamadi ({exc}); heuristic fallback kullanildi."],
                )

            env = self._build_subprocess_env(
                python_bin=self.python_bin,
                model_name=self.model_name,
                device=self.device,
                artifacts_path=self.artifacts_path,
            )

            cmd = [
                self.python_bin,
                str(runner_path),
                "--image",
                str(image_path),
                "--output",
                str(output_path),
                "--document-name",
                Path(document_name).name,
            ]
            try:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=False,
                    env=env,
                )
            except FileNotFoundError:
                return DetectorResult(
                    regions=[],
                    warnings=[f"Docling python bulunamadi ({self.python_bin}); heuristic fallback kullanildi."],
                )
            except Exception as exc:
                return DetectorResult(
                    regions=[],
                    warnings=[f"Docling layout subprocess baslatilamadi ({exc}); heuristic fallback kullanildi."],
                )

            stderr = (completed.stderr or "").strip()
            if completed.returncode != 0:
                detail = stderr.splitlines()[-1] if stderr else f"code={completed.returncode}"
                return DetectorResult(
                    regions=[],
                    warnings=[f"Docling layout runner basarisiz ({detail}); heuristic fallback kullanildi."],
                )
            if not output_path.exists():
                return DetectorResult(
                    regions=[],
                    warnings=["Docling layout runner cikti uretmedi; heuristic fallback kullanildi."],
                )
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except Exception as exc:
                return DetectorResult(
                    regions=[],
                    warnings=[f"Docling layout JSON okunamadi ({exc}); heuristic fallback kullanildi."],
                )

        rows = payload.get("regions", []) if isinstance(payload, dict) else []
        regions = [region for row in rows if isinstance(row, dict) for region in [self._region_from_row(row)] if region is not None]
        if not regions:
            return DetectorResult(
                regions=[],
                warnings=["Docling layout bbox bolgesi uretemedi; heuristic fallback kullanildi."],
            )
        warnings = [str(item).strip() for item in (payload.get("warnings", []) if isinstance(payload, dict) else []) if str(item).strip()]
        return DetectorResult(regions=regions, warnings=warnings)


def get_layout_detector(
    name: str,
    *,
    backend: str = "none",
    detector_dir: Path | None = None,
    docai_project_id: str = "",
    docai_location: str = "us",
    docai_processor_id: str = "",
    docai_processor_version: str = "",
    docai_timeout_seconds: int = 120,
    docling_python_bin: str = "",
    docling_layout_model: str = "docling-layout-heron-101",
    docling_artifacts_path: Path | None = None,
    docling_device: str = "auto",
) -> LayoutDetector:
    requested = (name or "").strip().lower()
    if requested == "detector":
        backend_name = (backend or "").strip().lower()
        if backend_name == "sidecar":
            return SidecarLayoutDetector(detector_dir)
        if backend_name == "docai":
            return DocAILayoutDetector(
                project_id=docai_project_id,
                location=docai_location,
                processor_id=docai_processor_id,
                processor_version=docai_processor_version,
                timeout_seconds=docai_timeout_seconds,
            )
        if backend_name == "docling":
            return DoclingLayoutDetector(
                python_bin=docling_python_bin,
                model_name=docling_layout_model,
                artifacts_path=docling_artifacts_path,
                device=docling_device,
            )
        return UnavailableLayoutDetector()
    return UnavailableLayoutDetector()


def detector_regions_available(regions: Sequence[DetectorRegion]) -> bool:
    return any(
        int(region.bbox_right) > int(region.bbox_left) and int(region.bbox_bottom) > int(region.bbox_top)
        for region in regions
    )
