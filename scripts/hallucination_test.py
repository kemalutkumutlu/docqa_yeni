from __future__ import annotations

"""
Hallucination & Faithfulness Test Suite

Measures:
  - Hallucination rate: % of out-of-scope questions where the system fabricates an answer
  - False-negative rate: % of in-scope questions incorrectly answered "not found"
  - Citation compliance: % of in-scope answers that include proper citations
  - Answer latency: avg response time per query

Run:
  python scripts/hallucination_test.py --pdf test_data/Case_Study_20260205.pdf
  python scripts/hallucination_test.py --pdf test_data/Case_Study_20260205.pdf --json
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


# ── Test cases ─────────────────────────────────────────────────────────────
# Each test case: (query, is_in_document, expected_keywords_if_positive)
# is_in_document = True -> answer should NOT be "Belgede bu bilgi bulunamadi."
# is_in_document = False -> answer MUST be exactly "Belgede bu bilgi bulunamadi."

_TEST_CASES_CASE_STUDY = [
    # ── POSITIVE: answers exist in Case_Study_20260205.pdf ──
    ("teslim suresi nedir", True, ["7 gun", "7 gün"]),
    ("projenin amaci nedir", True, ["belge analiz", "soru-cevap", "soru cevap"]),
    ("fonksiyonel gereksinimler nelerdir", True, ["belge yukleme", "belge yükleme"]),
    ("teslimatlar nelerdir", True, ["devlog", "testing", "readme"]),
    ("beklenen calisma suresi kac saat", True, ["25", "35"]),
    ("teslim yontemi nedir", True, ["github"]),
    ("demo video ne kadar surmeli", True, ["3", "5"]),
    ("pozisyon bilgisi nedir", True, ["mid-senior", "ai/ml", "yazilim", "yazılım"]),
    ("LLM araclari kullanmak serbest mi", True, ["serbest"]),
    ("teknik mulakatta ne bekleniyor", True, ["sunma", "demo", "tartis"]),
    # ── NEGATIVE: answers do NOT exist in this document ──
    ("araba kac beygir", False, []),
    ("turkiye nin baskenti neresidir", False, []),
    ("python programlama dilinin yaraticisi kimdir", False, []),
    ("dunya nufusu kac", False, []),
    ("yapay zeka ne zaman icat edildi", False, []),
    ("bu projenin butcesi ne kadar", False, []),
    # NOTE: "hangi veritabani kullaniliyor" was removed because the Case Study
    # document discusses building a RAG system (mentions ChromaDB / vector DB),
    # making it a borderline case — the LLM sometimes legitimately answers from
    # the technical-approach section.  Replaced with a clearly out-of-scope query.
    ("mars gezegeninin yuzey sicakligi kac derece", False, []),
    ("API endpoint leri nelerdir", False, []),
    ("kullanici kayit islemi nasil yapilir", False, []),
    ("sunucu gereksinimleri nelerdir", False, []),
    ("bu belgedeki grafikleri acikla", False, []),
    ("projenin gelir modeli nedir", False, []),
    ("musteri memnuniyeti orani kactir", False, []),
    ("agustos 2025 satis rakamlari nelerdir", False, []),
    ("rakip analizi sonuclari nelerdir", False, []),
]


def main(argv: Optional[list[str]] = None) -> int:
    _setup_utf8()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    ap = argparse.ArgumentParser(description="Hallucination & Faithfulness Test Suite")
    ap.add_argument("--pdf", default="test_data/Case_Study_20260205.pdf", help="Path to target PDF")
    ap.add_argument("--json", action="store_true", help="Output results as JSON")
    ap.add_argument("--runs", type=int, default=1, help="Number of runs per query (for flakiness detection)")
    args = ap.parse_args(argv)

    from src.config import load_settings
    from src.core.ingestion import OCRConfig
    from src.core.pipeline import RAGPipeline
    from src.core.vlm_extract import VLMConfig

    settings = load_settings()
    if settings.llm_provider != "gemini" or not settings.gemini_api_key:
        print("[FAIL] LLM_PROVIDER=gemini and GEMINI_API_KEY must be set in .env")
        return 1

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"[FAIL] PDF not found: {pdf}")
        return 1

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
        vlm_config=VLMConfig(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            mode=settings.vlm_mode,
        ),
    )

    print(f"Indexing {pdf.name} ...")
    t0 = time.perf_counter()
    st = pipe.add_document(pdf, display_name=pdf.name)
    index_time = time.perf_counter() - t0
    print(f"Indexed in {index_time:.1f}s  (chunks={len(st.chunks)} pages={st.page_count})\n")

    NOT_FOUND = "Belgede bu bilgi bulunamadi."
    NOT_FOUND_EXACT = "Belgede bu bilgi bulunamadı."

    # ── Run tests ──────────────────────────────────────────────────────────
    results = []

    # Counters
    positive_total = 0
    positive_correct = 0  # answered with content (not "not found")
    positive_with_citations = 0
    positive_keyword_hit = 0

    negative_total = 0
    negative_correct = 0  # answered with "not found"
    negative_hallucinated = 0  # answered with fabricated content

    latencies_positive = []
    latencies_negative = []

    for run_idx in range(args.runs):
        for query, is_in_doc, expected_keywords in _TEST_CASES_CASE_STUDY:
            t1 = time.perf_counter()
            try:
                res = pipe.ask(query)
                ans = (res.answer or "").strip()
                latency_ms = (time.perf_counter() - t1) * 1000
            except Exception as e:
                ans = f"ERROR: {e}"
                latency_ms = (time.perf_counter() - t1) * 1000

            # Normalize for comparison
            ans_lower = ans.lower().replace("ı", "i").replace("ö", "o").replace("ü", "u").replace("ç", "c").replace("ş", "s").replace("ğ", "g")
            is_not_found = (ans == NOT_FOUND_EXACT) or ("belgede bu bilgi bulunamadi" in ans_lower and len(ans) < 80)

            row = {
                "run": run_idx,
                "query": query,
                "is_in_doc": is_in_doc,
                "answer_preview": ans[:200],
                "is_not_found_response": is_not_found,
                "citations": res.citations_found if hasattr(res, "citations_found") else 0,
                "latency_ms": round(latency_ms, 1),
                "correct": False,
                "keyword_hit": None,
            }

            if is_in_doc:
                positive_total += 1
                latencies_positive.append(latency_ms)
                if not is_not_found:
                    positive_correct += 1
                    row["correct"] = True
                    if hasattr(res, "citations_found") and res.citations_found > 0:
                        positive_with_citations += 1
                    # Check keywords
                    if expected_keywords:
                        kw_hit = any(kw.lower() in ans_lower for kw in expected_keywords)
                        row["keyword_hit"] = kw_hit
                        if kw_hit:
                            positive_keyword_hit += 1
                    else:
                        row["keyword_hit"] = True
                        positive_keyword_hit += 1
                else:
                    row["correct"] = False
                    row["keyword_hit"] = False
            else:
                negative_total += 1
                latencies_negative.append(latency_ms)
                if is_not_found:
                    negative_correct += 1
                    row["correct"] = True
                else:
                    negative_hallucinated += 1
                    row["correct"] = False

            results.append(row)

    # ── Calculate metrics ──────────────────────────────────────────────────
    hallucination_rate = negative_hallucinated / negative_total * 100 if negative_total else 0
    faithfulness_rate = negative_correct / negative_total * 100 if negative_total else 0
    false_negative_rate = (positive_total - positive_correct) / positive_total * 100 if positive_total else 0
    true_positive_rate = positive_correct / positive_total * 100 if positive_total else 0
    citation_rate = positive_with_citations / positive_correct * 100 if positive_correct else 0
    keyword_accuracy = positive_keyword_hit / positive_total * 100 if positive_total else 0
    avg_latency_pos = sum(latencies_positive) / len(latencies_positive) if latencies_positive else 0
    avg_latency_neg = sum(latencies_negative) / len(latencies_negative) if latencies_negative else 0
    avg_latency_all = (sum(latencies_positive) + sum(latencies_negative)) / (len(latencies_positive) + len(latencies_negative)) if (latencies_positive or latencies_negative) else 0

    metrics = {
        "total_queries": len(_TEST_CASES_CASE_STUDY) * args.runs,
        "positive_total": positive_total,
        "positive_correct": positive_correct,
        "negative_total": negative_total,
        "negative_correct": negative_correct,
        "negative_hallucinated": negative_hallucinated,
        "hallucination_rate_pct": round(hallucination_rate, 1),
        "faithfulness_rate_pct": round(faithfulness_rate, 1),
        "true_positive_rate_pct": round(true_positive_rate, 1),
        "false_negative_rate_pct": round(false_negative_rate, 1),
        "citation_compliance_pct": round(citation_rate, 1),
        "keyword_accuracy_pct": round(keyword_accuracy, 1),
        "avg_latency_positive_ms": round(avg_latency_pos, 0),
        "avg_latency_negative_ms": round(avg_latency_neg, 0),
        "avg_latency_all_ms": round(avg_latency_all, 0),
    }

    if args.json:
        report = {**metrics, "details": results}
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        # ── Pretty output ──────────────────────────────────────────────────
        print("=" * 95)
        print(f"{'Query':<55} {'Type':^6} {'OK':^4} {'Cit':^4} {'KW':^4} {'ms':>6}")
        print("-" * 95)
        for r in results:
            typ = "POS" if r["is_in_doc"] else "NEG"
            ok_sym = "+" if r["correct"] else "X"
            cit_sym = str(r["citations"]) if r["citations"] else "-"
            kw_sym = "+" if r["keyword_hit"] is True else ("X" if r["keyword_hit"] is False else "-")
            q_short = r["query"][:53]
            print(f"{q_short:<55} {typ:^6} {ok_sym:^4} {cit_sym:^4} {kw_sym:^4} {r['latency_ms']:>6.0f}")
        print("=" * 95)

        print(f"\n{'='*50}")
        print(f" HALLUCINATION & FAITHFULNESS REPORT")
        print(f"{'='*50}")
        print(f"  Positive (in-doc) queries     : {positive_total}")
        print(f"    Correctly answered           : {positive_correct}/{positive_total} ({true_positive_rate:.0f}%)")
        print(f"    False negatives (missed)     : {positive_total - positive_correct}/{positive_total} ({false_negative_rate:.0f}%)")
        print(f"    Citation compliance          : {positive_with_citations}/{positive_correct} ({citation_rate:.0f}%)")
        print(f"    Keyword accuracy             : {positive_keyword_hit}/{positive_total} ({keyword_accuracy:.0f}%)")
        print(f"")
        print(f"  Negative (out-of-scope) queries: {negative_total}")
        print(f"    Correctly refused            : {negative_correct}/{negative_total} ({faithfulness_rate:.0f}%)")
        print(f"    Hallucinated (FAIL)          : {negative_hallucinated}/{negative_total} ({hallucination_rate:.0f}%)")
        print(f"")
        print(f"  Latency:")
        print(f"    Avg (positive)               : {avg_latency_pos:.0f} ms")
        print(f"    Avg (negative)               : {avg_latency_neg:.0f} ms")
        print(f"    Avg (all)                    : {avg_latency_all:.0f} ms")
        print(f"{'='*50}")

        # Verdict
        all_ok = (
            hallucination_rate == 0
            and false_negative_rate <= 20  # allow up to 20% flakiness for LLM
            and citation_rate >= 80
        )
        if all_ok:
            print("\nHALLUCINATION TEST PASSED")
        else:
            issues = []
            if hallucination_rate > 0:
                issues.append(f"hallucination_rate={hallucination_rate:.0f}%")
            if false_negative_rate > 20:
                issues.append(f"false_negative_rate={false_negative_rate:.0f}%")
            if citation_rate < 80:
                issues.append(f"citation_rate={citation_rate:.0f}%")
            print(f"\nHALLUCINATION TEST: issues found ({', '.join(issues)})")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
