"""
Phase 5 — LLM answer generation with strict guardrails.

Responsibilities:
  1. Build a context window from Evidence list
  2. Strict system prompt: no hallucination, mandatory citations
  3. Section-list mode: instruct LLM to list every item + coverage post-check
  4. Fallback: "Belgede bu bilgi bulunamadı." if context is empty / insufficient
"""
from __future__ import annotations

import os
import re
import time
import json
from pathlib import Path
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, List, Optional

from google import genai
from google.genai import types

from .gemini_client import build_gemini_client, gemini_model_candidates, is_model_not_found_error
from .retrieval import CoverageInfo, Evidence, QueryIntent, RetrievalResult


def _openai_retryable(e: Exception) -> bool:
    msg = str(e)
    return any(
        s in msg
        for s in (
            "503",
            "502",
            "504",
            "429",
            "500",
            "timed out",
            "Timeout",
            "Connection reset",
        )
    )


def _openai_chat_url() -> str:
    base_url = (os.getenv("OPENAI_BASE_URL", "") or "").strip().rstrip("/")
    if not base_url:
        return "https://api.openai.com/v1/chat/completions"
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _openai_chat_completion(
    api_key: str,
    model: str,
    system_instruction: str,
    user_contents: str,
    temperature: float,
    max_tokens: int = 4096,
) -> str:
    """
    Minimal OpenAI Chat Completions call via stdlib urllib (no extra dependency).
    """
    url = _openai_chat_url()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_contents},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    last_err: Optional[Exception] = None
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        except Exception as e:
            last_err = e
            if attempt >= 4 or not _openai_retryable(e):
                raise
            time.sleep(min(12.0, 1.5 * (2 ** (attempt - 1))))
    raise last_err  # type: ignore[misc]


def _openai_chat_completion_stream(
    api_key: str,
    model: str,
    system_instruction: str,
    user_contents: str,
    temperature: float,
    max_tokens: int = 4096,
    on_token: Optional[Callable[[str], None]] = None,
) -> str:
    """
    OpenAI streaming call (SSE data lines) with token callback.
    Returns full accumulated answer text.
    """
    url = _openai_chat_url()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_contents},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    def _emit(tok: str) -> None:
        if on_token and tok:
            try:
                on_token(tok)
            except Exception:
                pass

    text_parts: list[str] = []
    last_err: Optional[Exception] = None
    for attempt in range(1, 5):
        emitted_any = bool(text_parts)
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    evt = json.loads(data)
                    tok = (((evt.get("choices") or [{}])[0].get("delta") or {}).get("content") or "")
                    if tok:
                        text_parts.append(tok)
                        _emit(tok)
                        emitted_any = True
                return "".join(text_parts).strip()
        except Exception as e:
            last_err = e
            if attempt >= 4 or emitted_any or not _openai_retryable(e):
                raise
            time.sleep(min(12.0, 1.5 * (2 ** (attempt - 1))))

    raise last_err  # type: ignore[misc]


# ── System prompts ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT_BASE = """\
Sen bir belge analiz asistanısın. Sana verilen BAĞLAM parçalarını kullanarak \
kullanıcının sorusunu yanıtla.

KESİN KURALLAR — bunlara uymazsan cevap geçersiz sayılır:
1. SADECE verilen BAĞLAM'daki bilgileri kullan. Bağlamda olmayan hiçbir bilgiyi \
   ekleme, tahmin etme veya yorumlama.
2. Eğer sorunun cevabı bağlamda yoksa veya yetersizse, tam olarak şu cümleyi yaz: \
   "Belgede bu bilgi bulunamadı."
3. Her bilgi cümlesinin sonuna kaynak referansı ekle: [DosyaAdı - Sayfa X]
4. Türkçe cevap ver (kullanıcı İngilizce sorarsa İngilizce).
5. Cevabı düzgün formatlayarak ver (madde işaretleri, numaralı liste vb.).
"""

_SECTION_LIST_ADDENDUM = """\
UYARI: Bu bir "liste/bölüm çıkarma" sorusudur. Bağlamdaki ilgili bölümün \
ALTINDAKİ TÜM maddeleri, satırları veya alt başlıkları eksiksiz olarak listele. \
Hiçbirini atlama. Eğer bağlamda {expected} adet madde varsa, cevabında da en az \
{expected} adet madde olmalıdır.
"""

_CHAT_SYSTEM_PROMPT = """\
Sen yardımcı bir asistansın.

Kurallar:
- Normal sohbet edebilirsin (selamlaşma, hal hatır, genel sorular).
- Bu modda "belge içeriğine dayanarak" iddia üretme; belge soruları için kullanıcıdan belge moduna geçmesini iste.
- Gereksiz yere kaynak/citation yazma.
"""


def _chat_style_addendum(chat_style: str) -> str:
    style = (chat_style or "").strip().lower()
    if style == "empathetic":
        return (
            "\n\nTON KILAVUZU:\n"
            "- Kullanıcı olumsuz/üzgün bir duygu paylaşıyor.\n"
            "- Kısa bir empati ifadesiyle başla (örn. 'Üzgünüm, zor bir gün gibi görünüyor.').\n"
            "- Yargılamadan, sakin ve destekleyici bir dille devam et.\n"
        )
    if style == "congratulatory":
        return (
            "\n\nTON KILAVUZU:\n"
            "- Kullanıcı övgü/tebrik içerikli bir ifade kullandı.\n"
            "- Kısa bir teşekkür veya tebrik karşılığı ver.\n"
            "- Samimi ama kısa kal; abartılı ifadelerden kaçın.\n"
        )
    return ""


# ── Language selection (lightweight, document-agnostic) ───────────────────────

_TR_CHARS = set("çğıöşüÇĞİÖŞÜ")
_EN_CUES = {
    "what",
    "why",
    "who",
    "when",
    "where",
    "how",
    "list",
    "enumerate",
    "summarize",
    "requirements",
    "deliverables",
    "project",
    "document",
    "pdf",
    "section",
    "page",
    "about",
}
_TR_CUES = {
    "nedir",
    "nelerdir",
    "listele",
    "sırala",
    "sirala",
    "kaç",
    "kac",
    "belge",
    "doküman",
    "dokuman",
    "sayfa",
    "bölüm",
    "bolum",
    "madde",
    "teslimat",
    "gereksinim",
}


def _preferred_language(query: str) -> str:
    """
    Return "tr" or "en" based on lightweight cues.
    We keep this conservative: default to Turkish unless the query clearly looks English.
    """
    q = (query or "").strip()
    if not q:
        return "tr"
    if any(ch in _TR_CHARS for ch in q):
        return "tr"

    low = q.lower()
    # Turkish cue words (ASCII-only Turkish writing included)
    if any(w in low for w in _TR_CUES):
        return "tr"
    # English question cues
    if any(w in low for w in _EN_CUES):
        return "en"
    # If it's mostly ASCII and contains typical English spacing, lean English.
    if re.search(r"\b(what|how|why|when|where|who)\b", low):
        return "en"
    return "tr"


def _language_addendum(query: str) -> str:
    lang = _preferred_language(query)
    if lang == "en":
        return "\n\nCEVAP DILI: English. Answer strictly in English.\n"
    return "\n\nCEVAP DILI: Türkçe. Yanıtı kesinlikle Türkçe ver.\n"


# ── Context builder ──────────────────────────────────────────────────────────

def _build_context(evidences: List[Evidence]) -> str:
    """
    Assemble evidence chunks into a single context string for the LLM.
    Prefer parent chunks (full sections) to avoid redundancy with children.
    """
    # Deduplicate: if a parent exists for a section, skip its children
    parent_sections = {ev.section_id for ev in evidences if ev.kind == "parent"}
    blocks: list[str] = []
    seen_sections: set[str] = set()

    for ev in evidences:
        # Skip child chunks if parent is already included
        if ev.kind == "child" and ev.section_id in parent_sections:
            continue

        key = (ev.section_id, ev.kind)
        if key in seen_sections:
            continue
        seen_sections.add(key)

        header = f"[{ev.heading_path} | Sayfa {ev.page_start}"
        if ev.page_end != ev.page_start:
            header += f"-{ev.page_end}"
        header += "]"

        blocks.append(f"{header}\n{ev.text}")

    return "\n\n---\n\n".join(blocks)


def _visual_parts(evidences: List[Evidence], limit: int = 2) -> list[types.Part]:
    parts: list[types.Part] = []
    seen_paths: set[str] = set()
    for ev in evidences:
        if ev.modality != "visual" or not ev.image_path or ev.image_path in seen_paths:
            continue
        try:
            img_bytes = Path(ev.image_path).read_bytes()
        except Exception:
            continue
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
        seen_paths.add(ev.image_path)
        if len(parts) >= limit:
            break
    return parts


def _build_gemini_user_contents(user_message: str, evidences: List[Evidence]) -> str | list[types.Part]:
    image_parts = _visual_parts(evidences)
    if not image_parts:
        return user_message
    return [types.Part.from_text(text=user_message)] + image_parts


# ── Deterministic section-list rendering (doc-agnostic) ───────────────────────

def _extract_file_name_from_heading_path(heading_path: str) -> str:
    """
    heading_path is built as: "<file_name> / <heading> / <subheading> ..."
    """
    hp = (heading_path or "").strip()
    if " / " in hp:
        return hp.split(" / ", 1)[0].strip() or "Belge"
    return hp or "Belge"


def _strip_leading_heading_line(text: str) -> list[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return []
    # Parent chunks usually start with the section title.
    return lines[1:] if len(lines) > 1 else []


def _extract_numbered_or_bulleted(lines: list[str]) -> list[str]:
    out: list[str] = []
    for ln in lines:
        if re.match(r"^(\d+[\.\)]\s+|[a-zA-Z][\.\)]\s+|[-•*]\s+)", ln):
            out.append(ln)
    return out


def _extract_indexed_table_rows(lines: list[str]) -> list[str]:
    """
    Handle PDF table extraction patterns like:
      1
      LABEL
      description...
      2
      LABEL2
      description...
    """
    items: list[str] = []
    i = 0
    while i < len(lines):
        if re.match(r"^\d{1,3}$", lines[i].strip()):
            j = i + 1
            if j >= len(lines):
                break
            label = lines[j].strip()
            if re.match(r"^\d{1,3}$", label) or not (2 <= len(label) <= 140):
                i += 1
                continue
            k = j + 1
            desc_parts: list[str] = []
            while k < len(lines) and not re.match(r"^\d{1,3}$", lines[k].strip()):
                desc_parts.append(lines[k].strip())
                k += 1
            desc = " ".join([p for p in desc_parts if p]).strip()
            if desc:
                items.append(f"{label}: {desc}")
            else:
                items.append(label)
            i = k
            continue
        i += 1
    return items


def _extract_label_desc_pairs(lines: list[str]) -> list[str]:
    def _tok(s: str) -> int:
        return len([t for t in re.split(r"\s+", s.strip()) if t])

    def _looks_like_label(s: str) -> bool:
        if not (3 <= len(s) <= 60):
            return False
        if s.endswith((".", "!", "?", ":", ";")):
            return False
        if re.match(r"^\d+$", s):
            return False
        return _tok(s) <= 7

    def _looks_like_desc(s: str) -> bool:
        if len(s) >= 80:
            return True
        if _tok(s) >= 10:
            return True
        if any(p in s for p in (".", ";", ":")) and _tok(s) >= 6:
            return True
        return False

    # Drop a likely two-column table header if detected.
    # Pattern: two short "label-like" lines followed by another label and a description.
    # This is doc-agnostic (no vocabulary) and prevents headers from becoming items.
    if len(lines) >= 4:
        if _looks_like_label(lines[0]) and _looks_like_label(lines[1]) and _looks_like_label(lines[2]) and (
            _looks_like_desc(lines[3]) or len(lines[3]) >= 40
        ):
            lines = lines[2:]

    items: list[str] = []
    i = 0
    while i < len(lines) - 1:
        label = lines[i].strip()
        if not _looks_like_label(label):
            i += 1
            continue

        # Collect multi-line descriptions until the next label-like line starts a new row.
        j = i + 1
        desc_parts: list[str] = []
        while j < len(lines):
            s = lines[j].strip()
            if not s:
                j += 1
                continue
            if _looks_like_label(s) and desc_parts:
                break
            desc_parts.append(s)
            j += 1

        desc = " ".join([p for p in desc_parts if p]).strip()
        if desc and (_looks_like_desc(desc) or len(desc_parts) >= 1):
            # Skip probable table header pairs like "ColumnA: ColumnB"
            # where both sides look like short labels, not descriptions.
            if len(desc_parts) == 1 and _looks_like_label(desc_parts[0]) and not _looks_like_desc(desc_parts[0]):
                i = j
                continue
            items.append(f"{label}: {desc}")
            i = j
        else:
            i += 1

    return items


def _extract_subheadings(lines: list[str]) -> list[str]:
    out: list[str] = []
    for ln in lines:
        if re.match(r"^[A-Z0-9]+\.\d+", ln) or re.match(r"^\d+\.\d+", ln):
            out.append(ln)
    return out


def _extract_section_list_items(section_text: str) -> list[str]:
    """
    Document-agnostic extraction of list/table rows from a section's parent text.
    """
    lines = _strip_leading_heading_line(section_text)
    if not lines:
        return []

    # 1) Bullets / numbered list
    numbered = _extract_numbered_or_bulleted(lines)
    if len(numbered) >= 2:
        return numbered

    # 2) Indexed table rows (most common for PDF tables)
    indexed = _extract_indexed_table_rows(lines)
    if len(indexed) >= 3:
        return indexed

    # 3) Label/description pairs
    pairs = _extract_label_desc_pairs(lines)
    if len(pairs) >= 3:
        return pairs

    # 4) Subheadings inside the section
    subs = _extract_subheadings(lines)
    if len(subs) >= 3:
        return subs

    # No safe extraction: return empty to fall back to LLM.
    return []


def _render_deterministic_section_list(retrieval: RetrievalResult) -> Optional[str]:
    """
    If we can confidently extract items from the parent section chunk, render
    them deterministically with citations (no LLM).
    """
    if retrieval.intent != "section_list" or not retrieval.evidences or not retrieval.coverage:
        return None

    target_sid = retrieval.coverage.section_id
    parent_ev: Optional[Evidence] = None
    for ev in retrieval.evidences:
        if ev.kind == "parent" and ev.section_id == target_sid:
            parent_ev = ev
            break
    if parent_ev is None:
        # fallback: any parent
        for ev in retrieval.evidences:
            if ev.kind == "parent":
                parent_ev = ev
                break
    if parent_ev is None:
        return None

    items = _extract_section_list_items(parent_ev.text)
    if not items:
        return None

    # Only use deterministic rendering if it meets (or exceeds) the structural expected count.
    expected = retrieval.coverage.expected_items if retrieval.coverage else None
    if expected is not None and len(items) < expected:
        return None

    file_name = _extract_file_name_from_heading_path(parent_ev.heading_path)
    if parent_ev.page_start and parent_ev.page_end and parent_ev.page_end != parent_ev.page_start:
        cite = f"[{file_name} - Sayfa {parent_ev.page_start}-{parent_ev.page_end}]"
    else:
        cite = f"[{file_name} - Sayfa {parent_ev.page_start or 1}]"

    # Render as numbered list; keep items as-is (extract-only; no translation).
    lines_out: list[str] = []
    for idx, it in enumerate(items, start=1):
        t = (it or "").strip()
        if not t:
            continue
        lines_out.append(f"{idx}. {t} {cite}")

    return "\n".join(lines_out).strip() or None


# ── Coverage post-validation ─────────────────────────────────────────────────

def _count_answer_items(answer: str) -> int:
    """
    Count bullet / numbered items in the LLM's answer.
    """
    count = 0
    for line in answer.splitlines():
        line = line.strip()
        if re.match(r"^(\d+[\.\)]\s|[-•*]\s|[a-zA-Z][\.\)]\s)", line):
            count += 1
        # Also count simple "Label: ..." lines (common for table-to-list answers)
        elif re.match(r"^[^:\n]{2,80}:\s+.+", line):
            count += 1
    return count


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    citations_found: int
    coverage_expected: Optional[int]
    coverage_actual: Optional[int]
    coverage_ok: Optional[bool]  # None if not a section-list query
    intent: QueryIntent
    context_preview: str  # first N chars of context (for debug)


def generate_chat_answer(
    query: str,
    gemini_api_key: str,
    gemini_model: str = "gemini-2.0-flash",
    chat_style: str = "neutral",
) -> str:
    """
    Chat-only generation (no retrieval, no citations).
    """
    def _retryable(e: Exception) -> bool:
        msg = str(e)
        return any(
            s in msg
            for s in (
                "503",
                "UNAVAILABLE",
                "429",
                "RESOURCE_EXHAUSTED",
                "500",
                "INTERNAL",
                "timed out",
                "Timeout",
            )
        )

    last_err: Optional[Exception] = None
    for attempt in range(1, 5):
        try:
            last_model_error: Optional[Exception] = None
            for model_name in gemini_model_candidates(gemini_model):
                try:
                    client = build_gemini_client(gemini_api_key, model_name=model_name)
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=f"SORU: {query}",
                        config=types.GenerateContentConfig(
                            system_instruction=_CHAT_SYSTEM_PROMPT + _chat_style_addendum(chat_style) + _language_addendum(query),
                            temperature=0.4,
                            max_output_tokens=1024,
                        ),
                    )
                    return (resp.text or "").strip() or "Anlayamadım, tekrar eder misin?"
                except Exception as model_exc:
                    last_model_error = model_exc
                    if not is_model_not_found_error(model_exc):
                        raise
            if last_model_error is not None:
                raise last_model_error
        except Exception as e:
            last_err = e
            if attempt >= 4 or not _retryable(e):
                raise
            time.sleep(min(12.0, 1.5 * (2 ** (attempt - 1))))

    # Should never reach here
    raise last_err  # type: ignore[misc]


def generate_chat_answer_openai(
    query: str,
    openai_api_key: str,
    openai_model: str = "gpt-4o-mini",
    chat_style: str = "neutral",
) -> str:
    """
    Chat-only generation via OpenAI Chat Completions.
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, 5):
        try:
            return (
                _openai_chat_completion(
                    api_key=openai_api_key,
                    model=openai_model,
                    system_instruction=_CHAT_SYSTEM_PROMPT + _chat_style_addendum(chat_style) + _language_addendum(query),
                    user_contents=f"SORU: {query}",
                    temperature=0.4,
                    max_tokens=1024,
                )
                or "Anlayamadım, tekrar eder misin?"
            )
        except Exception as e:
            last_err = e
            if attempt >= 4 or not _openai_retryable(e):
                raise
            time.sleep(min(12.0, 1.5 * (2 ** (attempt - 1))))

    raise last_err  # type: ignore[misc]


# ── Main generation function ─────────────────────────────────────────────────

def generate_answer(
    retrieval: RetrievalResult,
    query: str,
    gemini_api_key: str,
    gemini_model: str = "gemini-2.0-flash",
) -> GenerationResult:
    """
    Given retrieval results + user query, call Gemini and return a
    guarded, cited answer.
    """
    # Edge case: no evidence
    if not retrieval.evidences:
        return GenerationResult(
            answer="Belgede bu bilgi bulunamadı.",
            citations_found=0,
            coverage_expected=None,
            coverage_actual=None,
            coverage_ok=None,
            intent=retrieval.intent,
            context_preview="",
        )

    # Deterministic path for section_list (prevents missing items / hallucination).
    # Only triggers when we have coverage info (i.e., a parent section chunk).
    deterministic = _render_deterministic_section_list(retrieval)
    if deterministic:
        citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", deterministic)) + len(
            re.findall(r"\[[^\]]*?/\s*\d+\s*\]", deterministic)
        )
        expected = retrieval.coverage.expected_items if retrieval.coverage else None
        actual = _count_answer_items(deterministic) if expected is not None else None
        ok = (actual >= expected) if (expected is not None and actual is not None) else None
        return GenerationResult(
            answer=deterministic,
            citations_found=citations_found,
            coverage_expected=expected,
            coverage_actual=actual,
            coverage_ok=ok,
            intent=retrieval.intent,
            context_preview="",  # deterministic path doesn't need to expose context
        )

    context = _build_context(retrieval.evidences)

    # Build system prompt
    system = _SYSTEM_PROMPT_BASE + _language_addendum(query)
    coverage_expected: Optional[int] = None
    if retrieval.intent == "section_list" and retrieval.coverage:
        coverage_expected = retrieval.coverage.expected_items
        system += _SECTION_LIST_ADDENDUM.format(expected=coverage_expected)

    # Build user message with context
    user_message = (
        f"BAĞLAM:\n{context}\n\n"
        f"---\n\n"
        f"SORU: {query}"
    )

    gemini_contents = _build_gemini_user_contents(user_message, retrieval.evidences)

    def _call(
        system_instruction: str,
        user_contents: str | list[types.Part],
        temperature: float,
        max_tokens: int = 4096,
    ) -> str:
        def _retryable(e: Exception) -> bool:
            msg = str(e)
            return any(
                s in msg
                for s in (
                    "503",
                    "UNAVAILABLE",
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "500",
                    "INTERNAL",
                    "timed out",
                    "Timeout",
                )
            )

        last_err: Optional[Exception] = None
        for attempt in range(1, 5):
            try:
                last_model_error: Optional[Exception] = None
                for model_name in gemini_model_candidates(gemini_model):
                    try:
                        client = build_gemini_client(gemini_api_key, model_name=model_name)
                        response = client.models.generate_content(
                            model=model_name,
                            contents=user_contents,
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction,
                                temperature=temperature,
                                max_output_tokens=max_tokens,
                            ),
                        )
                        return response.text or ""
                    except Exception as model_exc:
                        last_model_error = model_exc
                        if not is_model_not_found_error(model_exc):
                            raise
                if last_model_error is not None:
                    raise last_model_error
            except Exception as e:
                last_err = e
                if attempt >= 4 or not _retryable(e):
                    raise
                time.sleep(min(12.0, 1.5 * (2 ** (attempt - 1))))

        raise last_err  # type: ignore[misc]

    # Call Gemini
    answer = _call(system, gemini_contents, temperature=0.1) or "Belgede bu bilgi bulunamadı."

    # Count citations in the answer
    # Accept a few common citation renderings:
    # - [File - Sayfa 1]
    # - [File / Sayfa 1]
    # - [File | Sayfa 1]
    # - [File / 1]
    citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", answer)) + len(
        re.findall(r"\[[^\]]*?/\s*\d+\s*\]", answer)
    )

    # If citations are missing, do one strict retry to enforce formatting.
    if citations_found == 0 and retrieval.evidences and answer.strip() != "Belgede bu bilgi bulunamadı.":
        system_retry = (
            system
            + "\n\nFORMAT DÜZELTME MODU:\n"
            + "- Sadece cevabı yeniden yaz.\n"
            + "- Her cümle/madde sonunda mutlaka [DosyaAdı - Sayfa X] kaynak formatı olsun.\n"
            + "- Kaynaksız hiçbir cümle yazma.\n"
            + "- İçerik ekleme/çıkarma yapma; sadece formatı düzelt.\n"
        )
        answer_retry = _call(system_retry, gemini_contents, temperature=0.0).strip()
        if answer_retry:
            answer = answer_retry
            citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", answer)) + len(
                re.findall(r"\[[^\]]*?/\s*\d+\s*\]", answer)
            )

    # Coverage post-check
    coverage_actual: Optional[int] = None
    coverage_ok: Optional[bool] = None
    if coverage_expected is not None:
        coverage_actual = _count_answer_items(answer)
        coverage_ok = coverage_actual >= coverage_expected

        # If coverage failed, do one strict retry to force completeness (quality-first).
        if not coverage_ok and retrieval.evidences and answer.strip() != "Belgede bu bilgi bulunamadı.":
            system_retry2 = (
                system
                + "\n\nKAPSAM DÜZELTME MODU:\n"
                + f"- Bağlamda {coverage_expected} madde tespit edildi.\n"
                + f"- Cevabında EN AZ {coverage_expected} madde/satır olmalı.\n"
                + "- Her maddeyi ayrı satırda ver.\n"
                + "- Özetleme yapma; bağlamdaki öğeleri tek tek dök.\n"
                + "- Her satırın sonunda kaynak formatı olsun: [DosyaAdı - Sayfa X]\n"
            )
            answer_retry2 = _call(system_retry2, gemini_contents, temperature=0.0).strip()
            if answer_retry2:
                answer = answer_retry2
                citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", answer)) + len(
                    re.findall(r"\[[^\]]*?/\s*\d+\s*\]", answer)
                )
                coverage_actual = _count_answer_items(answer)
                coverage_ok = coverage_actual >= coverage_expected

        # If coverage failed, append a warning to the answer
        if not coverage_ok:
            answer += (
                f"\n\n⚠️ **Kapsam Uyarısı**: Bağlamda bu bölümde {coverage_expected} "
                f"madde tespit edildi, ancak cevapta {coverage_actual} madde var. "
                f"Lütfen cevabı kontrol edin."
            )

    return GenerationResult(
        answer=answer,
        citations_found=citations_found,
        coverage_expected=coverage_expected,
        coverage_actual=coverage_actual,
        coverage_ok=coverage_ok,
        intent=retrieval.intent,
        context_preview=context[:500],
    )


def generate_answer_openai(
    retrieval: RetrievalResult,
    query: str,
    openai_api_key: str,
    openai_model: str = "gpt-4o-mini",
) -> GenerationResult:
    """
    OpenAI variant of generate_answer() with the same guardrails/retries.
    """
    if not retrieval.evidences:
        return GenerationResult(
            answer="Belgede bu bilgi bulunamadı.",
            citations_found=0,
            coverage_expected=None,
            coverage_actual=None,
            coverage_ok=None,
            intent=retrieval.intent,
            context_preview="",
        )

    deterministic = _render_deterministic_section_list(retrieval)
    if deterministic:
        citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", deterministic)) + len(
            re.findall(r"\[[^\]]*?/\s*\d+\s*\]", deterministic)
        )
        expected = retrieval.coverage.expected_items if retrieval.coverage else None
        actual = _count_answer_items(deterministic) if expected is not None else None
        ok = (actual >= expected) if (expected is not None and actual is not None) else None
        return GenerationResult(
            answer=deterministic,
            citations_found=citations_found,
            coverage_expected=expected,
            coverage_actual=actual,
            coverage_ok=ok,
            intent=retrieval.intent,
            context_preview="",
        )

    context = _build_context(retrieval.evidences)
    system = _SYSTEM_PROMPT_BASE + _language_addendum(query)
    coverage_expected: Optional[int] = None
    if retrieval.intent == "section_list" and retrieval.coverage:
        coverage_expected = retrieval.coverage.expected_items
        system += _SECTION_LIST_ADDENDUM.format(expected=coverage_expected)

    user_message = (
        f"BAĞLAM:\n{context}\n\n"
        f"---\n\n"
        f"SORU: {query}"
    )

    answer = _openai_chat_completion(
        api_key=openai_api_key,
        model=openai_model,
        system_instruction=system,
        user_contents=user_message,
        temperature=0.1,
        max_tokens=4096,
    ) or "Belgede bu bilgi bulunamadı."

    citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", answer)) + len(
        re.findall(r"\[[^\]]*?/\s*\d+\s*\]", answer)
    )

    if citations_found == 0 and retrieval.evidences and answer.strip() != "Belgede bu bilgi bulunamadı.":
        system_retry = (
            system
            + "\n\nFORMAT DÜZELTME MODU:\n"
            + "- Sadece cevabı yeniden yaz.\n"
            + "- Her cümle/madde sonunda mutlaka [DosyaAdı - Sayfa X] kaynak formatı olsun.\n"
            + "- Kaynaksız hiçbir cümle yazma.\n"
            + "- İçerik ekleme/çıkarma yapma; sadece formatı düzelt.\n"
        )
        answer_retry = _openai_chat_completion(
            api_key=openai_api_key,
            model=openai_model,
            system_instruction=system_retry,
            user_contents=user_message,
            temperature=0.0,
            max_tokens=4096,
        ).strip()
        if answer_retry:
            answer = answer_retry
            citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", answer)) + len(
                re.findall(r"\[[^\]]*?/\s*\d+\s*\]", answer)
            )

    coverage_actual: Optional[int] = None
    coverage_ok: Optional[bool] = None
    if coverage_expected is not None:
        coverage_actual = _count_answer_items(answer)
        coverage_ok = coverage_actual >= coverage_expected

        if not coverage_ok and retrieval.evidences and answer.strip() != "Belgede bu bilgi bulunamadı.":
            system_retry2 = (
                system
                + "\n\nKAPSAM DÜZELTME MODU:\n"
                + f"- Bağlamda {coverage_expected} madde tespit edildi.\n"
                + f"- Cevabında EN AZ {coverage_expected} madde/satır olmalı.\n"
                + "- Her maddeyi ayrı satırda ver.\n"
                + "- Özetleme yapma; bağlamdaki öğeleri tek tek dök.\n"
                + "- Her satırın sonunda kaynak formatı olsun: [DosyaAdı - Sayfa X]\n"
            )
            answer_retry2 = _openai_chat_completion(
                api_key=openai_api_key,
                model=openai_model,
                system_instruction=system_retry2,
                user_contents=user_message,
                temperature=0.0,
                max_tokens=4096,
            ).strip()
            if answer_retry2:
                answer = answer_retry2
                citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", answer)) + len(
                    re.findall(r"\[[^\]]*?/\s*\d+\s*\]", answer)
                )
                coverage_actual = _count_answer_items(answer)
                coverage_ok = coverage_actual >= coverage_expected

        if not coverage_ok:
            answer += (
                f"\n\n⚠️ **Kapsam Uyarısı**: Bağlamda bu bölümde {coverage_expected} "
                f"madde tespit edildi, ancak cevapta {coverage_actual} madde var. "
                f"Lütfen cevabı kontrol edin."
            )

    return GenerationResult(
        answer=answer,
        citations_found=citations_found,
        coverage_expected=coverage_expected,
        coverage_actual=coverage_actual,
        coverage_ok=coverage_ok,
        intent=retrieval.intent,
        context_preview=context[:500],
    )


# ── Streaming generation (token callback) ───────────────────────────────────

def generate_answer_stream(
    retrieval: RetrievalResult,
    query: str,
    gemini_api_key: str,
    gemini_model: str = "gemini-2.0-flash",
    on_token: Optional[Callable[[str], None]] = None,
) -> GenerationResult:
    """
    Streaming variant of generate_answer() for UI token-by-token rendering.

    Architectural note:
      - The standard non-streaming generate_answer() remains the source of truth.
      - This path keeps the same retrieval/context/prompt logic, but intentionally
        skips post-generation rewrite retries (citation/coverage retry), because
        once tokens are emitted to the UI they cannot be retracted safely.
    """
    def _emit(text: str) -> None:
        if on_token and text:
            try:
                on_token(text)
            except Exception:
                pass

    if not retrieval.evidences:
        answer = "Belgede bu bilgi bulunamadı."
        _emit(answer)
        return GenerationResult(
            answer=answer,
            citations_found=0,
            coverage_expected=None,
            coverage_actual=None,
            coverage_ok=None,
            intent=retrieval.intent,
            context_preview="",
        )

    deterministic = _render_deterministic_section_list(retrieval)
    if deterministic:
        _emit(deterministic)
        citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", deterministic)) + len(
            re.findall(r"\[[^\]]*?/\s*\d+\s*\]", deterministic)
        )
        expected = retrieval.coverage.expected_items if retrieval.coverage else None
        actual = _count_answer_items(deterministic) if expected is not None else None
        ok = (actual >= expected) if (expected is not None and actual is not None) else None
        return GenerationResult(
            answer=deterministic,
            citations_found=citations_found,
            coverage_expected=expected,
            coverage_actual=actual,
            coverage_ok=ok,
            intent=retrieval.intent,
            context_preview="",
        )

    context = _build_context(retrieval.evidences)
    system = _SYSTEM_PROMPT_BASE + _language_addendum(query)
    coverage_expected: Optional[int] = None
    if retrieval.intent == "section_list" and retrieval.coverage:
        coverage_expected = retrieval.coverage.expected_items
        system += _SECTION_LIST_ADDENDUM.format(expected=coverage_expected)

    user_message = (
        f"BAĞLAM:\n{context}\n\n"
        f"---\n\n"
        f"SORU: {query}"
    )
    gemini_contents = _build_gemini_user_contents(user_message, retrieval.evidences)

    chunks: list[str] = []
    last_model_error: Optional[Exception] = None
    for model_name in gemini_model_candidates(gemini_model):
        try:
            client = build_gemini_client(gemini_api_key, model_name=model_name)
            for event in client.models.generate_content_stream(
                model=model_name,
                contents=gemini_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.1,
                    max_output_tokens=4096,
                ),
            ):
                token = (event.text or "")
                if token:
                    chunks.append(token)
                    _emit(token)
            break
        except Exception as model_exc:
            last_model_error = model_exc
            chunks = []
            if not is_model_not_found_error(model_exc):
                raise
    if not chunks and last_model_error is not None:
        raise last_model_error

    answer = "".join(chunks).strip() or "Belgede bu bilgi bulunamadı."

    citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", answer)) + len(
        re.findall(r"\[[^\]]*?/\s*\d+\s*\]", answer)
    )

    coverage_actual: Optional[int] = None
    coverage_ok: Optional[bool] = None
    if coverage_expected is not None:
        coverage_actual = _count_answer_items(answer)
        coverage_ok = coverage_actual >= coverage_expected
        if coverage_ok is False:
            warning = (
                f"\n\n⚠️ **Kapsam Uyarısı**: Bağlamda bu bölümde {coverage_expected} "
                f"madde tespit edildi, ancak cevapta {coverage_actual} madde var. "
                f"Lütfen cevabı kontrol edin."
            )
            answer += warning
            _emit(warning)

    return GenerationResult(
        answer=answer,
        citations_found=citations_found,
        coverage_expected=coverage_expected,
        coverage_actual=coverage_actual,
        coverage_ok=coverage_ok,
        intent=retrieval.intent,
        context_preview=context[:500],
    )


def generate_answer_openai_stream(
    retrieval: RetrievalResult,
    query: str,
    openai_api_key: str,
    openai_model: str = "gpt-4o-mini",
    on_token: Optional[Callable[[str], None]] = None,
) -> GenerationResult:
    """
    Streaming OpenAI variant for token-by-token UI rendering.
    """
    def _emit(text: str) -> None:
        if on_token and text:
            try:
                on_token(text)
            except Exception:
                pass

    if not retrieval.evidences:
        answer = "Belgede bu bilgi bulunamadı."
        _emit(answer)
        return GenerationResult(
            answer=answer,
            citations_found=0,
            coverage_expected=None,
            coverage_actual=None,
            coverage_ok=None,
            intent=retrieval.intent,
            context_preview="",
        )

    deterministic = _render_deterministic_section_list(retrieval)
    if deterministic:
        _emit(deterministic)
        citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", deterministic)) + len(
            re.findall(r"\[[^\]]*?/\s*\d+\s*\]", deterministic)
        )
        expected = retrieval.coverage.expected_items if retrieval.coverage else None
        actual = _count_answer_items(deterministic) if expected is not None else None
        ok = (actual >= expected) if (expected is not None and actual is not None) else None
        return GenerationResult(
            answer=deterministic,
            citations_found=citations_found,
            coverage_expected=expected,
            coverage_actual=actual,
            coverage_ok=ok,
            intent=retrieval.intent,
            context_preview="",
        )

    context = _build_context(retrieval.evidences)
    system = _SYSTEM_PROMPT_BASE + _language_addendum(query)
    coverage_expected: Optional[int] = None
    if retrieval.intent == "section_list" and retrieval.coverage:
        coverage_expected = retrieval.coverage.expected_items
        system += _SECTION_LIST_ADDENDUM.format(expected=coverage_expected)

    user_message = (
        f"BAĞLAM:\n{context}\n\n"
        f"---\n\n"
        f"SORU: {query}"
    )

    answer = _openai_chat_completion_stream(
        api_key=openai_api_key,
        model=openai_model,
        system_instruction=system,
        user_contents=user_message,
        temperature=0.1,
        max_tokens=4096,
        on_token=_emit,
    ) or "Belgede bu bilgi bulunamadı."

    citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", answer)) + len(
        re.findall(r"\[[^\]]*?/\s*\d+\s*\]", answer)
    )

    coverage_actual: Optional[int] = None
    coverage_ok: Optional[bool] = None
    if coverage_expected is not None:
        coverage_actual = _count_answer_items(answer)
        coverage_ok = coverage_actual >= coverage_expected
        if coverage_ok is False:
            warning = (
                f"\n\n⚠️ **Kapsam Uyarısı**: Bağlamda bu bölümde {coverage_expected} "
                f"madde tespit edildi, ancak cevapta {coverage_actual} madde var. "
                f"Lütfen cevabı kontrol edin."
            )
            answer += warning
            _emit(warning)

    return GenerationResult(
        answer=answer,
        citations_found=citations_found,
        coverage_expected=coverage_expected,
        coverage_actual=coverage_actual,
        coverage_ok=coverage_ok,
        intent=retrieval.intent,
        context_preview=context[:500],
    )


# ── Local LLM generation (Ollama) ──────────────────────────────────────────

def generate_chat_answer_local(
    query: str,
    ollama_cfg: "OllamaConfig",
    chat_style: str = "neutral",
) -> str:
    """
    Chat-only generation via local Ollama (no retrieval, no citations).
    """
    from .local_llm import OllamaConfig, ollama_chat  # noqa: F811

    system = _CHAT_SYSTEM_PROMPT + _chat_style_addendum(chat_style) + _language_addendum(query)
    result = ollama_chat(cfg=ollama_cfg, system=system, user_message=f"SORU: {query}", temperature=0.4, max_tokens=1024)
    return result or "Anlayamadım, tekrar eder misin?"


def generate_answer_local(
    retrieval: RetrievalResult,
    query: str,
    ollama_cfg: "OllamaConfig",
) -> GenerationResult:
    """
    Given retrieval results + user query, call local Ollama LLM and return a
    guarded, cited answer.  Same guardrails / prompts as the Gemini path.
    """
    from .local_llm import OllamaConfig, ollama_chat  # noqa: F811

    # Edge case: no evidence
    if not retrieval.evidences:
        return GenerationResult(
            answer="Belgede bu bilgi bulunamadı.",
            citations_found=0,
            coverage_expected=None,
            coverage_actual=None,
            coverage_ok=None,
            intent=retrieval.intent,
            context_preview="",
        )

    # Deterministic path (shared, no LLM call needed).
    deterministic = _render_deterministic_section_list(retrieval)
    if deterministic:
        citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", deterministic)) + len(
            re.findall(r"\[[^\]]*?/\s*\d+\s*\]", deterministic)
        )
        expected = retrieval.coverage.expected_items if retrieval.coverage else None
        actual = _count_answer_items(deterministic) if expected is not None else None
        ok = (actual >= expected) if (expected is not None and actual is not None) else None
        return GenerationResult(
            answer=deterministic,
            citations_found=citations_found,
            coverage_expected=expected,
            coverage_actual=actual,
            coverage_ok=ok,
            intent=retrieval.intent,
            context_preview="",
        )

    context = _build_context(retrieval.evidences)

    # Build system prompt (SAME as Gemini path)
    system = _SYSTEM_PROMPT_BASE + _language_addendum(query)
    coverage_expected: Optional[int] = None
    if retrieval.intent == "section_list" and retrieval.coverage:
        coverage_expected = retrieval.coverage.expected_items
        system += _SECTION_LIST_ADDENDUM.format(expected=coverage_expected)

    user_message = (
        f"BAĞLAM:\n{context}\n\n"
        f"---\n\n"
        f"SORU: {query}"
    )

    def _call_local(sys: str, msg: str, temp: float, max_tok: int = 4096) -> str:
        return ollama_chat(cfg=ollama_cfg, system=sys, user_message=msg, temperature=temp, max_tokens=max_tok)

    answer = _call_local(system, user_message, 0.1) or "Belgede bu bilgi bulunamadı."

    # Citation counting (same logic as Gemini path)
    citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", answer)) + len(
        re.findall(r"\[[^\]]*?/\s*\d+\s*\]", answer)
    )

    # Citation retry
    if citations_found == 0 and retrieval.evidences and answer.strip() != "Belgede bu bilgi bulunamadı.":
        system_retry = (
            system
            + "\n\nFORMAT DÜZELTME MODU:\n"
            + "- Sadece cevabı yeniden yaz.\n"
            + "- Her cümle/madde sonunda mutlaka [DosyaAdı - Sayfa X] kaynak formatı olsun.\n"
            + "- Kaynaksız hiçbir cümle yazma.\n"
            + "- İçerik ekleme/çıkarma yapma; sadece formatı düzelt.\n"
        )
        answer_retry = _call_local(system_retry, user_message, 0.0).strip()
        if answer_retry:
            answer = answer_retry
            citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", answer)) + len(
                re.findall(r"\[[^\]]*?/\s*\d+\s*\]", answer)
            )

    # Coverage post-check
    coverage_actual: Optional[int] = None
    coverage_ok: Optional[bool] = None
    if coverage_expected is not None:
        coverage_actual = _count_answer_items(answer)
        coverage_ok = coverage_actual >= coverage_expected

        if not coverage_ok and retrieval.evidences and answer.strip() != "Belgede bu bilgi bulunamadı.":
            system_retry2 = (
                system
                + "\n\nKAPSAM DÜZELTME MODU:\n"
                + f"- Bağlamda {coverage_expected} madde tespit edildi.\n"
                + f"- Cevabında EN AZ {coverage_expected} madde/satır olmalı.\n"
                + "- Her maddeyi ayrı satırda ver.\n"
                + "- Özetleme yapma; bağlamdaki öğeleri tek tek dök.\n"
                + "- Her satırın sonunda kaynak formatı olsun: [DosyaAdı - Sayfa X]\n"
            )
            answer_retry2 = _call_local(system_retry2, user_message, 0.0).strip()
            if answer_retry2:
                answer = answer_retry2
                citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", answer)) + len(
                    re.findall(r"\[[^\]]*?/\s*\d+\s*\]", answer)
                )
                coverage_actual = _count_answer_items(answer)
                coverage_ok = coverage_actual >= coverage_expected

        if not coverage_ok:
            answer += (
                f"\n\n⚠️ **Kapsam Uyarısı**: Bağlamda bu bölümde {coverage_expected} "
                f"madde tespit edildi, ancak cevapta {coverage_actual} madde var. "
                f"Lütfen cevabı kontrol edin."
            )

    return GenerationResult(
        answer=answer,
        citations_found=citations_found,
        coverage_expected=coverage_expected,
        coverage_actual=coverage_actual,
        coverage_ok=coverage_ok,
        intent=retrieval.intent,
        context_preview=context[:500],
    )


def generate_answer_local_stream(
    retrieval: RetrievalResult,
    query: str,
    ollama_cfg: "OllamaConfig",
    on_token: Optional[Callable[[str], None]] = None,
) -> GenerationResult:
    """
    Streaming variant of generate_answer_local() for UI token-by-token rendering.

    Keeps the same retrieval/context/prompt logic, but skips post-generation
    rewrite retries for the same reason as generate_answer_stream().
    """
    from .local_llm import OllamaConfig, ollama_chat_stream  # noqa: F811

    def _emit(text: str) -> None:
        if on_token and text:
            try:
                on_token(text)
            except Exception:
                pass

    if not retrieval.evidences:
        answer = "Belgede bu bilgi bulunamadı."
        _emit(answer)
        return GenerationResult(
            answer=answer,
            citations_found=0,
            coverage_expected=None,
            coverage_actual=None,
            coverage_ok=None,
            intent=retrieval.intent,
            context_preview="",
        )

    deterministic = _render_deterministic_section_list(retrieval)
    if deterministic:
        _emit(deterministic)
        citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", deterministic)) + len(
            re.findall(r"\[[^\]]*?/\s*\d+\s*\]", deterministic)
        )
        expected = retrieval.coverage.expected_items if retrieval.coverage else None
        actual = _count_answer_items(deterministic) if expected is not None else None
        ok = (actual >= expected) if (expected is not None and actual is not None) else None
        return GenerationResult(
            answer=deterministic,
            citations_found=citations_found,
            coverage_expected=expected,
            coverage_actual=actual,
            coverage_ok=ok,
            intent=retrieval.intent,
            context_preview="",
        )

    context = _build_context(retrieval.evidences)
    system = _SYSTEM_PROMPT_BASE + _language_addendum(query)
    coverage_expected: Optional[int] = None
    if retrieval.intent == "section_list" and retrieval.coverage:
        coverage_expected = retrieval.coverage.expected_items
        system += _SECTION_LIST_ADDENDUM.format(expected=coverage_expected)

    user_message = (
        f"BAĞLAM:\n{context}\n\n"
        f"---\n\n"
        f"SORU: {query}"
    )

    answer = ollama_chat_stream(
        cfg=ollama_cfg,
        system=system,
        user_message=user_message,
        temperature=0.1,
        max_tokens=4096,
        on_token=_emit,
    ) or "Belgede bu bilgi bulunamadı."

    citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", answer)) + len(
        re.findall(r"\[[^\]]*?/\s*\d+\s*\]", answer)
    )

    coverage_actual: Optional[int] = None
    coverage_ok: Optional[bool] = None
    if coverage_expected is not None:
        coverage_actual = _count_answer_items(answer)
        coverage_ok = coverage_actual >= coverage_expected
        if coverage_ok is False:
            warning = (
                f"\n\n⚠️ **Kapsam Uyarısı**: Bağlamda bu bölümde {coverage_expected} "
                f"madde tespit edildi, ancak cevapta {coverage_actual} madde var. "
                f"Lütfen cevabı kontrol edin."
            )
            answer += warning
            _emit(warning)

    return GenerationResult(
        answer=answer,
        citations_found=citations_found,
        coverage_expected=coverage_expected,
        coverage_actual=coverage_actual,
        coverage_ok=coverage_ok,
        intent=retrieval.intent,
        context_preview=context[:500],
    )


# ── Local / extractive generation (LLM-free) ──────────────────────────────

def generate_extractive_answer(
    retrieval: RetrievalResult,
    query: str,
) -> GenerationResult:
    """
    Generate an answer WITHOUT any LLM call.

    - section_list intent → deterministic rendering (same as the LLM path)
    - normal_qa intent   → return top evidence snippets verbatim with citations

    This allows the system to work when LLM_PROVIDER=none.
    """
    if not retrieval.evidences:
        return GenerationResult(
            answer="Belgede bu bilgi bulunamadı.",
            citations_found=0,
            coverage_expected=None,
            coverage_actual=None,
            coverage_ok=None,
            intent=retrieval.intent,
            context_preview="",
        )

    # Deterministic section list (shared with the LLM path).
    deterministic = _render_deterministic_section_list(retrieval)
    if deterministic:
        citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", deterministic)) + len(
            re.findall(r"\[[^\]]*?/\s*\d+\s*\]", deterministic)
        )
        expected = retrieval.coverage.expected_items if retrieval.coverage else None
        actual = _count_answer_items(deterministic) if expected is not None else None
        ok = (actual >= expected) if (expected is not None and actual is not None) else None
        return GenerationResult(
            answer=deterministic,
            citations_found=citations_found,
            coverage_expected=expected,
            coverage_actual=actual,
            coverage_ok=ok,
            intent=retrieval.intent,
            context_preview="",
        )

    # Extractive fallback: top evidence snippets with citations.
    lines: list[str] = []
    seen_pages: set[str] = set()
    for ev in retrieval.evidences[:5]:  # top 5 evidence chunks
        file_name = _extract_file_name_from_heading_path(ev.heading_path)
        if ev.page_start and ev.page_end and ev.page_end != ev.page_start:
            cite = f"[{file_name} - Sayfa {ev.page_start}-{ev.page_end}]"
        elif ev.page_start:
            cite = f"[{file_name} - Sayfa {ev.page_start}]"
        else:
            cite = f"[{file_name}]"
        # Deduplicate same page content
        key = f"{file_name}:{ev.page_start}:{ev.page_end}:{ev.text[:80]}"
        if key in seen_pages:
            continue
        seen_pages.add(key)

        snippet = ev.text.strip()
        if len(snippet) > 800:
            snippet = snippet[:800] + "…"
        lines.append(f"{snippet}\n{cite}")

    answer = "\n\n---\n\n".join(lines)
    citations_found = len(lines)

    return GenerationResult(
        answer=answer,
        citations_found=citations_found,
        coverage_expected=None,
        coverage_actual=None,
        coverage_ok=None,
        intent=retrieval.intent,
        context_preview="",
    )
