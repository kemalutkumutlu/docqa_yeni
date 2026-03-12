from __future__ import annotations

"""
Retrieval quality evaluation (LLM-free).

Loads eval_questions.json, runs retrieval pipeline on the target PDF,
and reports intent accuracy, hit@k, and evidence recall.

Run:
  python scripts/eval_retrieval.py --pdf Case_Study_20260205.pdf
  python scripts/eval_retrieval.py --pdf Case_Study_20260205.pdf --json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional


def _setup_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def main(argv: Optional[list[str]] = None) -> int:
    _setup_utf8()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    ap = argparse.ArgumentParser(description="Retrieval quality evaluation (LLM-free)")
    ap.add_argument("--pdf", default="Case_Study_20260205.pdf", help="Path to target PDF")
    ap.add_argument(
        "--questions",
        default=str(Path(__file__).resolve().parents[1] / "test_data" / "eval_questions.json"),
        help="Path to eval questions JSON",
    )
    ap.add_argument("--json", action="store_true", help="Output results as JSON")
    ap.add_argument("--top-k", type=int, default=8, help="Top-k for retrieval (default: 8)")
    args = ap.parse_args(argv)

    # ── Load questions ────────────────────────────────────────────────────────
    q_path = Path(args.questions)
    if not q_path.exists():
        print(f"[ERROR] Questions file not found: {q_path}")
        return 1

    with q_path.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    if not questions:
        print("[ERROR] Empty questions file")
        return 1

    # ── Build pipeline ────────────────────────────────────────────────────────
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        return 1

    from src.config import load_settings
    from src.core.ingestion import OCRConfig
    from src.core.pipeline import RAGPipeline
    from src.core.vlm_extract import VLMConfig

    settings = load_settings()

    pipe = RAGPipeline(
        embedding_model=settings.embedding_model,
        chroma_dir=settings.chroma_dir,
        gemini_api_key=settings.gemini_api_key or "",
        gemini_model=settings.gemini_model,
        ocr_config=OCRConfig(
            enabled=getattr(settings, "ocr_enabled", True),
            lang="tur+eng",
            tesseract_cmd=settings.tesseract_cmd,
            tessdata_prefix=settings.tessdata_prefix,
            tesseract_config=getattr(settings, "tesseract_config", None),
        ),
        vlm_config=VLMConfig(
            api_key=settings.gemini_api_key or "",
            model=settings.gemini_model,
            mode=settings.vlm_mode,
        ),
    )

    print(f"Indexing {pdf_path.name} ...")
    t0 = time.perf_counter()
    pipe.add_document(pdf_path, display_name=pdf_path.name)
    index_time = time.perf_counter() - t0
    print(f"Indexed in {index_time:.1f}s  (chunks={pipe.total_chunks})\n")

    # ── Run evaluation ────────────────────────────────────────────────────────
    results = []
    intent_correct = 0
    heading_hits = 0
    heading_total = 0
    section_hits = 0
    section_total = 0
    evidence_met = 0

    for q in questions:
        query = q["query"]
        expected_intent = q["expected_intent"]
        expected_section = q.get("expected_section_key")
        expected_heading = q.get("expected_heading_contains")
        min_evidence = q.get("min_evidence_count", 0)

        t1 = time.perf_counter()
        ret = pipe.get_retrieval(query)
        latency_ms = (time.perf_counter() - t1) * 1000

        # Intent accuracy
        intent_ok = ret.intent == expected_intent
        if intent_ok:
            intent_correct += 1

        # Section key hit (for section_list queries)
        section_ok = None
        if expected_section:
            section_total += 1
            matched_sections = set()
            for ev in ret.evidences:
                sid = ev.section_id or ""
                # section_id is like "doc_a:2" or just "2"
                key_part = sid.rsplit(":", 1)[-1] if ":" in sid else sid
                matched_sections.add(key_part)
            section_ok = expected_section in matched_sections
            if section_ok:
                section_hits += 1

        # Heading contains hit
        heading_ok = None
        if expected_heading:
            heading_total += 1
            all_headings = " ".join(ev.heading_path for ev in ret.evidences)
            heading_ok = expected_heading.lower() in all_headings.lower()
            if heading_ok:
                heading_hits += 1

        # Evidence count
        evidence_ok = len(ret.evidences) >= min_evidence
        if evidence_ok:
            evidence_met += 1

        row = {
            "query": query,
            "intent_expected": expected_intent,
            "intent_actual": ret.intent,
            "intent_ok": intent_ok,
            "evidence_count": len(ret.evidences),
            "min_evidence": min_evidence,
            "evidence_ok": evidence_ok,
            "section_ok": section_ok,
            "heading_ok": heading_ok,
            "latency_ms": round(latency_ms, 1),
        }
        results.append(row)

    # ── Report ────────────────────────────────────────────────────────────────
    total = len(questions)

    if args.json:
        report = {
            "total_questions": total,
            "intent_accuracy": round(intent_correct / total, 3) if total else 0,
            "heading_hit_rate": round(heading_hits / heading_total, 3) if heading_total else None,
            "section_hit_rate": round(section_hits / section_total, 3) if section_total else None,
            "evidence_met_rate": round(evidence_met / total, 3) if total else 0,
            "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / total, 1) if total else 0,
            "details": results,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        # Pretty table
        print("=" * 90)
        print(f"{'Query':<50} {'Intent':^8} {'Evid':^5} {'Sec':^5} {'Head':^5} {'ms':>6}")
        print("-" * 90)
        for r in results:
            intent_sym = "✓" if r["intent_ok"] else "✗"
            evid_sym = "✓" if r["evidence_ok"] else "✗"
            sec_sym = "✓" if r["section_ok"] is True else ("✗" if r["section_ok"] is False else "·")
            head_sym = "✓" if r["heading_ok"] is True else ("✗" if r["heading_ok"] is False else "·")
            q_short = r["query"][:48]
            print(f"{q_short:<50} {intent_sym:^8} {evid_sym:^5} {sec_sym:^5} {head_sym:^5} {r['latency_ms']:>6.0f}")
        print("=" * 90)

        # Summary
        print(f"\n  Intent Accuracy : {intent_correct}/{total} ({intent_correct/total*100:.0f}%)")
        if heading_total:
            print(f"  Heading Hit     : {heading_hits}/{heading_total} ({heading_hits/heading_total*100:.0f}%)")
        if section_total:
            print(f"  Section Hit     : {section_hits}/{section_total} ({section_hits/section_total*100:.0f}%)")
        print(f"  Evidence Met    : {evidence_met}/{total} ({evidence_met/total*100:.0f}%)")
        avg_lat = sum(r["latency_ms"] for r in results) / total if total else 0
        print(f"  Avg Latency     : {avg_lat:.0f} ms")
        print()

        # Final verdict
        all_ok = intent_correct == total and evidence_met == total
        if heading_total:
            all_ok = all_ok and heading_hits == heading_total
        if section_total:
            all_ok = all_ok and section_hits == section_total

        if all_ok:
            print("RETRIEVAL EVAL PASSED")
        else:
            print("RETRIEVAL EVAL: some checks failed (see above)")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
