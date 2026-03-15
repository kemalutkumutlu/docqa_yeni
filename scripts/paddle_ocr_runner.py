from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def _patch_opencv_headless_metadata_alias() -> None:
    original_version = importlib_metadata.version

    def _version(name: str) -> str:
        if name == "opencv-contrib-python":
            try:
                return original_version(name)
            except importlib_metadata.PackageNotFoundError:
                return original_version("opencv-contrib-python-headless")
        return original_version(name)

    importlib_metadata.version = _version


_patch_opencv_headless_metadata_alias()
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _resolve_paddle_device(device: str) -> str | None:
    dev = (device or "auto").strip().lower()
    if dev == "cpu":
        return "cpu"
    if dev == "cuda":
        return "gpu:0"
    return None


def _resolve_paddle_lang(lang: str) -> str:
    value = (lang or "").strip().lower()
    if value in ("tur+eng", "eng+tur", "tur", "tr"):
        return "tr"
    if value in ("eng", "en"):
        return "en"
    return "en"


def _collect_strings(value) -> list[str]:
    out: list[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        clean = value.strip()
        if clean:
            out.append(clean)
        return out
    if isinstance(value, dict):
        preferred = [
            "markdown",
            "markdown_text",
            "text",
            "content",
            "rec_texts",
            "overall_ocr_res",
            "res",
        ]
        for key in preferred:
            if key in value:
                out.extend(_collect_strings(value.get(key)))
        for key, item in value.items():
            if key in preferred:
                continue
            out.extend(_collect_strings(item))
        return out
    if isinstance(value, (list, tuple)):
        for item in value:
            out.extend(_collect_strings(item))
        return out
    if hasattr(value, "res"):
        out.extend(_collect_strings(getattr(value, "res")))
    if hasattr(value, "json"):
        out.extend(_collect_strings(getattr(value, "json")))
    if hasattr(value, "markdown"):
        out.extend(_collect_strings(getattr(value, "markdown")))
    if hasattr(value, "rec_texts"):
        out.extend(_collect_strings(getattr(value, "rec_texts")))
    return out


def _unique_join(lines: list[str]) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        clean = " ".join((line or "").split()).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return "\n".join(out).strip()


def _run_paddle(image_path: Path, *, lang: str, device: str, ocr_version: str) -> str:
    from paddleocr import PaddleOCR

    kwargs = {
        "lang": _resolve_paddle_lang(lang),
        "ocr_version": ocr_version or "PP-OCRv5",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    resolved_device = _resolve_paddle_device(device)
    if resolved_device:
        kwargs["device"] = resolved_device
    pipeline = PaddleOCR(**kwargs)
    img = np.array(Image.open(image_path).convert("RGB"))
    results = pipeline.predict(img)
    text = _unique_join(_collect_strings(results))
    if not text:
        raise RuntimeError("PaddleOCR bos sonuc dondurdu")
    return text


def _run_paddle_vl(image_path: Path, *, device: str) -> str:
    from paddleocr import PaddleOCRVL

    kwargs = {}
    resolved_device = _resolve_paddle_device(device)
    if resolved_device:
        kwargs["device"] = resolved_device
    pipeline = PaddleOCRVL(**kwargs)
    img = np.array(Image.open(image_path).convert("RGB"))
    results = pipeline.predict(img)
    text = _unique_join(_collect_strings(results))
    if not text:
        raise RuntimeError("PaddleOCR-VL bos sonuc dondurdu")
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--backend", required=True, choices=["paddle", "paddle_vl"])
    parser.add_argument("--lang", default="tur+eng")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ocr-version", default="PP-OCRv5")
    args = parser.parse_args()

    image_path = Path(args.image)
    try:
        if args.backend == "paddle_vl":
            text = _run_paddle_vl(image_path, device=args.device)
        else:
            text = _run_paddle(image_path, lang=args.lang, device=args.device, ocr_version=args.ocr_version)
        print(json.dumps({"ok": True, "text": text}, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
