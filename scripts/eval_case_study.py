from __future__ import annotations

"""
Case_Study_20260205.pdf strict acceptance eval.

Goal: a deterministic, repeatable gate for "90+" quality without tuning logic
to the document content (the eval itself is doc-specific; the system remains
document-agnostic).

Run:
  python scripts/eval_case_study.py
  python scripts/eval_case_study.py --pdf Case_Study_20260205.pdf
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def _setup_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _contains_any(hay: str, needles: list[str]) -> bool:
    h = (hay or "").lower()
    return any(n.lower() in h for n in needles)


def main(argv: Optional[list[str]] = None) -> int:
    _setup_utf8()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="Case_Study_20260205.pdf", help="Path to Case Study PDF")
    args = ap.parse_args(argv)

    from src.config import load_settings
    from src.core.ingestion import OCRConfig
    from src.core.pipeline import RAGPipeline
    from src.core.vlm_extract import VLMConfig

    settings = load_settings()
    if settings.llm_provider != "gemini" or not settings.gemini_api_key:
        return _fail("LLM_PROVIDER=gemini and GEMINI_API_KEY must be set in .env")

    pdf = Path(args.pdf)
    if not pdf.exists():
        return _fail(f"PDF not found: {pdf}")

    pipe = RAGPipeline(
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
        # Match UI default: quality-first VLM is allowed, but ingestion selects best text by dual-quality.
        vlm_config=VLMConfig(api_key=settings.gemini_api_key, model=settings.gemini_model, mode="force"),
    )

    st = pipe.add_document(pdf, display_name=pdf.name)
    if len(st.chunks) < 12:
        return _fail(f"Ingestion unstable: chunks={len(st.chunks)} (<12)")
    _ok(f"ingestion chunks={len(st.chunks)} pages={st.page_count}")

    # P0.1 / P0.2 / P1.x style gates
    tests = [
        # (query, expect_intent, expect_min_items, must_contain_any, must_not_contain_any)
        ("fonksiyonel gereksinimler nelerdir", "section_list", 5, [], ["kapsam uyarısı"]),
        ("teslimatlar nelerdir", "section_list", 5, [], ["kapsam uyarısı"]),
        ("teslim süresi nedir", "normal_qa", None, ["7 gün", "7gun"], []),
        ("projenin amacı nedir", "normal_qa", None, [], []),
        ("araba kaç beygir", "normal_qa", None, ["belgede bu bilgi bulunamadı"], []),
    ]

    for q, intent, min_items, must_any, must_not_any in tests:
        res = pipe.ask(q)
        ans = (res.answer or "").strip()
        low = ans.lower()

        if res.intent != intent:
            return _fail(f"intent mismatch for {q!r}: got={res.intent} expected={intent}")

        # Must not hallucinate for irrelevant query in doc mode.
        if q == "araba kaç beygir":
            if ans != "Belgede bu bilgi bulunamadı.":
                return _fail(f"negative test failed: answer={ans!r}")
            _ok(f"negative query -> not found ({q!r})")
            continue

        if must_any and not _contains_any(ans, must_any):
            return _fail(f"missing required phrase(s) for {q!r}: need one of {must_any}, got={ans[:220]!r}")

        if must_not_any and _contains_any(ans, must_not_any):
            return _fail(f"contains forbidden phrase(s) for {q!r}: {must_not_any}")

        if ans == "Belgede bu bilgi bulunamadı.":
            return _fail(f"unexpected not-found for {q!r}")

        if res.citations_found <= 0:
            return _fail(f"missing citations for {q!r}")

        if min_items is not None:
            # Deterministic section_list should pass coverage gates.
            if res.coverage_ok is False:
                return _fail(
                    f"coverage failed for {q!r}: expected={res.coverage_expected} got={res.coverage_actual}"
                )
            if res.coverage_actual is not None and res.coverage_actual < min_items:
                return _fail(f"too few items for {q!r}: got={res.coverage_actual} need>={min_items}")

        _ok(f"query={q!r}: intent={res.intent} citations={res.citations_found}")

    print("CASE STUDY EVAL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

