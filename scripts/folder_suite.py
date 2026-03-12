from __future__ import annotations

"""
Folder suite: run the pipeline over every PDF in a folder and (optionally) generate answers.

Why:
- You can drop many PDFs under test_data/ and run one command.
- With RAG_LOG=1 enabled, every Q/A is persisted to JSONL logs (per session + per doc).

Examples:
  # Retrieval-only (no LLM calls), fastest:
  python scripts/folder_suite.py --dir test_data --mode retrieval

  # Full ask() (requires Gemini key):
  python scripts/folder_suite.py --dir test_data --mode ask

  # Use a custom query list file (one query per line):
  python scripts/folder_suite.py --dir test_data --mode retrieval --queries scripts/queries.txt
"""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Iterable


def _setup_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def _load_queries(path: Path | None) -> list[str]:
    if path is None:
        # Minimal, document-agnostic default set.
        return [
            "Bu dokümanın konusu nedir?",
            "Özet çıkar.",
            "Bu dokümanda geçen önemli başlıklar nelerdir?",
            "Bu dokümanda geçen sayılar/tarih aralıkları nelerdir?",
        ]
    raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    qs = [ln.strip() for ln in raw if ln.strip() and not ln.strip().startswith("#")]
    return qs


def _iter_pdfs(folder: Path) -> list[Path]:
    # Use filesystem glob (not ripgrep) so gitignore doesn't affect this.
    pdfs = sorted(folder.rglob("*.pdf"), key=lambda p: p.name.lower())
    return pdfs


def main() -> int:
    _setup_utf8()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="test_data", help="Folder containing PDFs")
    ap.add_argument(
        "--isolate",
        type=int,
        default=1,
        help="1: index each PDF in isolation (faster, safer). 0: load all PDFs into one session.",
    )
    ap.add_argument(
        "--max_pdfs",
        type=int,
        default=0,
        help="If >0, only process the first N PDFs (useful for quick sanity checks).",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="retrieval",
        choices=["retrieval", "ask"],
        help="retrieval: no LLM calls; ask: generate answers (requires Gemini key)",
    )
    ap.add_argument("--queries", type=str, default="", help="Path to queries.txt (one per line)")
    args = ap.parse_args()

    from src.config import load_settings
    from src.core.ingestion import OCRConfig
    from src.core.pipeline import RAGPipeline
    from src.core.vlm_extract import VLMConfig

    settings = load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    folder = Path(args.dir)
    if not folder.exists() or not folder.is_dir():
        print(f"[FAIL] folder not found: {folder}")
        return 2

    pdfs = _iter_pdfs(folder)
    if not pdfs:
        print(f"[FAIL] no PDFs found under: {folder}")
        return 2
    if args.max_pdfs and args.max_pdfs > 0:
        pdfs = pdfs[: int(args.max_pdfs)]

    if args.mode == "ask":
        if settings.llm_provider != "gemini" or not settings.gemini_api_key:
            print("[FAIL] ask mode requires LLM_PROVIDER=gemini and GEMINI_API_KEY in .env")
            return 2

    qpath = Path(args.queries) if args.queries.strip() else None
    queries = _load_queries(qpath)
    if not queries:
        print("[FAIL] no queries loaded")
        return 2

    def _make_pipe(chroma_dir: Path) -> RAGPipeline:
        return RAGPipeline(
            embedding_model=settings.embedding_model,
            chroma_dir=chroma_dir,
            gemini_api_key=settings.gemini_api_key,
            gemini_model=settings.gemini_model,
            ocr_config=OCRConfig(
                enabled=getattr(settings, "ocr_enabled", True),
                lang="tur+eng",
                tesseract_cmd=settings.tesseract_cmd,
                tessdata_prefix=settings.tessdata_prefix,
                tesseract_config=getattr(settings, "tesseract_config", None),
            ),
            vlm_config=VLMConfig(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                mode=settings.vlm_mode,
                max_pages=settings.vlm_max_pages,
            ),
        )

    # In isolate mode we create a new pipeline per PDF with a temporary Chroma dir
    # (prevents O(N^2) re-embedding as you add more PDFs).
    pipe_shared = None if args.isolate else _make_pipe(settings.chroma_dir)
    sid = pipe_shared.session_id if pipe_shared else "per-pdf"
    print(
        f"[OK] session_id={sid} mode={args.mode} pdfs={len(pdfs)} queries={len(queries)} isolate={bool(args.isolate)}",
        flush=True,
    )

    failures: list[str] = []
    for p in pdfs:
        pipe = pipe_shared
        tmp_cm = None
        if pipe is None:
            tmp_cm = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
            tmp = Path(tmp_cm.name)
            pipe = _make_pipe(tmp / "chroma")

        try:
            st = pipe.add_document(p, display_name=p.name)
            print(
                f"\n[OK] indexed {st.file_name}: pages={st.page_count} chunks={len(st.chunks)} session={pipe.session_id}",
                flush=True,
            )
            if st.warnings:
                print(f"  warnings: {'; '.join(st.warnings)[:220]}", flush=True)
        except Exception as e:
            msg = f"index failed for {p.name}: {e}"
            print(f"[FAIL] {msg}", flush=True)
            failures.append(msg)
            if tmp_cm is not None:
                tmp_cm.cleanup()
            continue

        for q in queries:
            try:
                if args.mode == "retrieval":
                    ret = pipe.get_retrieval(q)
                    print(
                        f"  [OK] retrieval q={q!r}: intent={ret.intent} evidences={len(ret.evidences)}",
                        flush=True,
                    )
                else:
                    res = pipe.ask(q)
                    # Print a compact preview; full content is in JSONL logs if enabled.
                    prev = (res.answer or "").replace("\n", " ")[:140]
                    print(
                        f"  [OK] ask q={q!r}: intent={res.intent} citations={res.citations_found} preview={prev!r}",
                        flush=True,
                    )
            except Exception as e:
                msg = f"query failed for {p.name} q={q!r}: {e}"
                print(f"  [FAIL] {msg}", flush=True)
                failures.append(msg)

        if tmp_cm is not None:
            tmp_cm.cleanup()

    if failures:
        print(f"\n[FAIL] failures={len(failures)} (showing first 5)", flush=True)
        for m in failures[:5]:
            print(" - " + m, flush=True)
        return 1

    print("\nFOLDER SUITE PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

