"""
Phase 5 — LLM answer generation with strict guardrails.

Responsibilities:
  1. Build a context window from Evidence list
  2. Strict system prompt: no hallucination, mandatory citations
  3. Section-list mode: instruct LLM to list every item + coverage post-check
  4. Fallback: "Belgede bu bilgi bulunamadı." if context is empty / insufficient
"""
from __future__ import annotations

import logging
import os
import re
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

import httpx

from .retrieval import CoverageInfo, Evidence, QueryIntent, RetrievalResult


def _google_genai_types():
    from google.genai import types

    return types


def _gemini_helpers():
    from .gemini_client import (
        build_gemini_client,
        gemini_model_candidates,
        is_model_not_found_error,
        is_retryable_api_error,
    )

    return build_gemini_client, gemini_model_candidates, is_model_not_found_error, is_retryable_api_error


_RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def _retry_sleep_seconds(attempt: int) -> float:
    return min(12.0, 1.5 * (2 ** (attempt - 1)))


def _openai_retryable(e: Exception) -> bool:
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code in _RETRYABLE_HTTP_STATUS_CODES
    if isinstance(e, (httpx.TimeoutException, httpx.TransportError, OSError, TimeoutError)):
        return True
    return False


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
    user_contents: Any,
    temperature: float,
    max_tokens: int = 4096,
) -> str:
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
    last_err: Optional[Exception] = None
    for attempt in range(1, 5):
        try:
            with httpx.Client(timeout=120, follow_redirects=True) as client:
                resp = client.post(
                    url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                )
                resp.raise_for_status()
                body = resp.json()
            return (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        except Exception as e:
            last_err = e
            if attempt >= 4 or not _openai_retryable(e):
                raise
            time.sleep(_retry_sleep_seconds(attempt))
    raise last_err  # type: ignore[misc]


def _openai_chat_completion_stream(
    api_key: str,
    model: str,
    system_instruction: str,
    user_contents: Any,
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
        try:
            with httpx.stream(
                "POST",
                url,
                json=payload,
                timeout=120,
                follow_redirects=True,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            ) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines():
                    line = (raw or "").strip()
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
            time.sleep(_retry_sleep_seconds(attempt))

    raise last_err  # type: ignore[misc]



from .prompts import (
    SYSTEM_PROMPT_BASE,
    SECTION_LIST_ADDENDUM,
    MULTI_SECTION_ADDENDUM,
    CHAT_SYSTEM_PROMPT,
    chat_style_addendum,
)

_COMPLETE_ENDINGS = (".", "!", "?", '."', '!"', '?"')
_INCOMPLETE_ENDINGS = (",", " ve", " ile", " veya", " çünkü", " ama", " fakat", " Ancak", " Ayrıca")

def _response_looks_incomplete(text: str) -> bool:
    """
    Heuristic guard against provider answers that end mid-sentence.
    Deliberately conservative to avoid unnecessary continuation calls.
    """
    body = (text or "").strip()
    if not body:
        return False
    if body.endswith(_COMPLETE_ENDINGS):
        return False
    if body.endswith(_INCOMPLETE_ENDINGS):
        return True
    if body.count("```") % 2 == 1:
        return True
    if body.count("**") % 2 == 1:
        return True
    if body.count("(") > body.count(")"):
        return True
    if body.count("[") > body.count("]"):
        return True

    words = body.split()
    if len(words) < 16:
        return False

    lines = [ln.rstrip() for ln in body.splitlines() if ln.strip()]
    if lines:
        last_line = lines[-1].strip()
        if re.match(r"^[-*•]\s+\S*$", last_line):
            return True
        if re.match(r"^\d+[\.\)]\s+\S+$", last_line):
            return True
        if re.match(r"^[^:\n]{2,60}:\s*$", last_line):
            return True
        if re.search(r"\[[^\]]+\]$", last_line):
            return False

    last_char = body[-1]
    if last_char.isalnum() and len(body) >= 200:
        tail_words = re.findall(r"\w+", lines[-1] if lines else body[-120:])
        if len(tail_words) >= 4:
            return True
    if body.endswith("...") and len(body) >= 40:
        return True
    return False


def _merge_continuation(base: str, addition: str) -> str:
    base_clean = (base or "").rstrip()
    extra_clean = (addition or "").strip()
    if not extra_clean:
        return base_clean

    if extra_clean in base_clean[-240:]:
        return base_clean

    max_overlap = min(len(base_clean), len(extra_clean), 160)
    overlap = 0
    for size in range(max_overlap, 15, -1):
        if base_clean[-size:] == extra_clean[:size]:
            overlap = size
            break
    if overlap:
        extra_clean = extra_clean[overlap:].lstrip()

    if not extra_clean:
        return base_clean
    if not base_clean:
        return extra_clean
    if base_clean.endswith(("\n", " ")):
        return f"{base_clean}{extra_clean}"
    return f"{base_clean} {extra_clean}"


def _continuation_suffix(base: str, completed: str) -> str:
    base_clean = base or ""
    full_clean = completed or ""
    if full_clean.startswith(base_clean):
        return full_clean[len(base_clean):]
    prefix = 0
    limit = min(len(base_clean), len(full_clean))
    while prefix < limit and base_clean[prefix] == full_clean[prefix]:
        prefix += 1
    return full_clean[prefix:]


def _continue_prompt(query: str, partial_answer: str) -> str:
    tail = partial_answer[-120:].strip()
    return (
        f"SORU: {query}\n\n"
        f"ONCEKI_YANIT:\n{partial_answer}\n\n"
        f"KESILEN_SON_BOLUM:\n{tail}\n\n"
        "Görev:\n"
        "- Önceki yanıt yarım kaldı.\n"
        "- Aynı yanıtı kaldığı yerden devam ettir.\n"
        "- Baştan yazma, tekrar etme, özetleme yapma.\n"
        "- Sadece eksik kalan devam kısmını ver.\n"
        "- Yanıtı doğal ve tamamlanmış şekilde bitir.\n"
        "- Sonu eksik bir ifade, tarih, sezon, sayı veya özel isim ile bitirme.\n"
    )


def _complete_if_incomplete(
    initial_text: str,
    *,
    query: str,
    continue_fn: Callable[[str], str],
    max_rounds: int = 2,
) -> str:
    text = (initial_text or "").strip()
    if not text:
        return text

    for _ in range(max_rounds):
        if not _response_looks_incomplete(text):
            break
        extra = (continue_fn(_continue_prompt(query, text)) or "").strip()
        if not extra:
            break
        merged = _merge_continuation(text, extra)
        if merged == text:
            break
        text = merged
    return text


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

def _verify_answer_grounding(answer: str, context: str) -> str:
    """
    Post-generation verification: check if the answer is grounded in context.

    Heuristic checks:
    1. If answer contains specific numbers/names not found in context → flag
    2. If answer is very long but context is very short → suspicious
    3. If answer says "belgede" + positive claim but context is empty → override

    Returns the (possibly modified) answer.
    """
    answer_stripped = answer.strip()
    _NOT_FOUND = "Belgede bu bilgi bulunamadı."

    # If context is empty/tiny but answer is substantive → force not-found
    if len(context.strip()) < 20 and answer_stripped and answer_stripped != _NOT_FOUND:
        return _NOT_FOUND

    # If answer is much longer than context (ratio > 3x) and context is short → suspicious
    if context.strip() and len(answer_stripped) > 3 * len(context.strip()) and len(context.strip()) < 200:
        # Don't override if the answer looks like a well-cited response
        citation_count = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", answer_stripped))
        if citation_count == 0:
            return _NOT_FOUND

    return answer


def _grounding_check(evidences: List[Evidence], *, min_avg_score: float = -1.0) -> List[Evidence]:
    """
    Pre-generation grounding check: verify that retrieved evidence is
    actually relevant before sending to the LLM.

    If the average score of non-enrichment evidence is below *min_avg_score*,
    return an empty list — causing the LLM to respond with
    "Belgede bu bilgi bulunamadı." instead of hallucinating from
    weakly-related context.

    Section-complete fetches (score=1.0) are always trusted.
    """
    if not evidences:
        return evidences

    # Resolve threshold: -1 sentinel means "read from env/config default"
    if min_avg_score < 0:
        import os
        try:
            min_avg_score = float(os.getenv("GROUNDING_MIN_AVG_SCORE", "0.15"))
        except (ValueError, TypeError):
            min_avg_score = 0.15

    # Section-complete fetches always pass grounding
    if any(ev.score == 1.0 for ev in evidences):
        return evidences

    # Compute average score of real evidence
    # Exclude enrichment parent chunks (score < 0 sentinel) from avg calculation
    real_scores = [ev.score for ev in evidences if ev.score > 0]
    if not real_scores:
        return evidences

    avg = sum(real_scores) / len(real_scores)
    if avg < min_avg_score:
        return []  # empty context → LLM says "Belgede bu bilgi bulunamadı."

    return evidences


def _estimate_tokens(text: str) -> int:
    """
    Fast token estimation without external dependencies.

    Heuristic: ~1 token per 3.5 chars for Turkish/mixed text.
    This is conservative (overestimates) to avoid exceeding limits.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 3.5))


# Maximum context tokens to send to the LLM.
# Configurable via CONTEXT_MAX_TOKENS env var (default: 100_000).
def _get_context_max_tokens() -> int:
    import os
    try:
        return int(os.getenv("CONTEXT_MAX_TOKENS", "100000"))
    except (ValueError, TypeError):
        return 100_000


def _build_context(evidences: List[Evidence]) -> str:
    """
    Assemble evidence chunks into a single context string for the LLM.
    Prefer parent chunks (full sections) to avoid redundancy with children.
    Enforces a token budget to prevent context window overflow.
    """
    # Grounding check: reject weakly-related evidence before context building
    evidences = _grounding_check(evidences)

    if not evidences:
        return ""

    max_tokens = _get_context_max_tokens()

    # Deduplicate: if a parent exists for a section, skip its children
    parent_sections = {ev.section_id for ev in evidences if ev.kind == "parent"}
    blocks: list[str] = []
    seen_sections: set[str] = set()
    total_tokens = 0
    separator_tokens = _estimate_tokens("\n\n---\n\n")

    for ev in evidences:
        # Skip child chunks if parent is already included
        if ev.kind == "child" and ev.section_id in parent_sections:
            continue

        key = (ev.section_id, ev.kind)
        if key in seen_sections:
            continue
        seen_sections.add(key)

        header = _format_evidence_header(ev)
        block = f"{header}\n{ev.text}"
        block_tokens = _estimate_tokens(block)

        # Check if adding this block would exceed the budget
        sep_cost = separator_tokens if blocks else 0
        if total_tokens + block_tokens + sep_cost > max_tokens and blocks:
            # Budget exhausted — stop adding more evidence
            break

        blocks.append(block)
        total_tokens += block_tokens + sep_cost

    return "\n\n---\n\n".join(blocks)


def _evidence_region_suffix(ev: Evidence) -> str:
    region = (getattr(ev, "region_label", "") or "").strip()
    if not region:
        return ""
    return f", Region {region}"


def _evidence_region_details(ev: Evidence) -> str:
    parts: list[str] = []
    region_id = (getattr(ev, "region_id", "") or "").strip()
    crop_type = (getattr(ev, "crop_type", "") or "").strip()
    region_summary = (getattr(ev, "region_summary", "") or "").strip()
    proposal_source = (getattr(ev, "proposal_source", "") or "").strip()
    proposal_confidence = float(getattr(ev, "proposal_confidence", 0.0) or 0.0)
    if region_id:
        parts.append(f"id={region_id}")
    if crop_type and crop_type != "page":
        parts.append(f"type={crop_type}")
    if proposal_source:
        parts.append(f"source={proposal_source}")
    if proposal_confidence > 0:
        parts.append(f"conf={proposal_confidence:.2f}")
    if region_summary:
        parts.append(f"summary={region_summary}")
    return " | ".join(parts)


def _format_evidence_header(ev: Evidence) -> str:
    header = f"[{ev.heading_path} | Sayfa {ev.page_start}"
    if ev.page_end != ev.page_start:
        header += f"-{ev.page_end}"
    if ev.region_label:
        heading_low = (ev.heading_path or "").strip().lower()
        region_low = ev.region_label.strip().lower()
        if f"region {region_low}" not in heading_low:
            header += _evidence_region_suffix(ev)
    details = _evidence_region_details(ev)
    if details:
        header += f" | {details}"
    header += "]"
    return header


def _format_citation(file_name: str, ev: Evidence) -> str:
    cite = f"[{file_name}"
    if ev.page_start and ev.page_end and ev.page_end != ev.page_start:
        cite += f" - Sayfa {ev.page_start}-{ev.page_end}"
    elif ev.page_start:
        cite += f" - Sayfa {ev.page_start}"
    cite += _evidence_region_suffix(ev)
    region_id = (getattr(ev, "region_id", "") or "").strip()
    crop_type = (getattr(ev, "crop_type", "") or "").strip()
    if region_id:
        cite += f", {region_id}"
    if crop_type and crop_type != "page":
        cite += f", {crop_type}"
    cite += "]"
    return cite


def _visual_parts(evidences: List[Evidence], limit: int = 2) -> list[Any]:
    types = _google_genai_types()
    parts: list[Any] = []
    seen_paths: set[str] = set()
    for ev in evidences:
        if ev.modality != "visual" or not ev.image_path or ev.image_path in seen_paths:
            continue
        try:
            img_bytes = Path(ev.image_path).read_bytes()
        except Exception:
            logger.debug("Failed to read visual evidence image: %s", ev.image_path, exc_info=True)
            continue
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
        seen_paths.add(ev.image_path)
        if len(parts) >= limit:
            break
    return parts


_VISUAL_QUERY_PATTERNS = [
    re.compile(r"\b(tablo|table|satır|satir|row|sütun|sutun|column|hücre|hucre|cell)\b", re.IGNORECASE),
    re.compile(r"\b(şekil|sekil|figure|diagram|diyagram|grafik|chart|caption)\b", re.IGNORECASE),
    re.compile(r"\b(form|alan|field|kutu|box|checkbox|label|değer|deger|value)\b", re.IGNORECASE),
    re.compile(r"\b(görsel|gorsel|resim|image|layout|yerleşim|yerlesim|sayfadaki|page)\b", re.IGNORECASE),
]


def _query_prefers_visual_reasoning(query: str) -> bool:
    text = (query or "").strip()
    if not text:
        return False
    return any(pat.search(text) for pat in _VISUAL_QUERY_PATTERNS)


def _should_use_multimodal_answer_generation(
    query: str,
    evidences: List[Evidence],
    multimodal_answer_mode: str,
) -> bool:
    mode = (multimodal_answer_mode or "auto").strip().lower()
    has_visual = any(ev.modality == "visual" and ev.image_path for ev in evidences)
    if not has_visual or mode == "off":
        return False
    if mode == "on":
        return True
    if not any(ev.modality == "text" for ev in evidences):
        return True
    return _query_prefers_visual_reasoning(query)


def _build_gemini_user_contents(
    user_message: str,
    query: str,
    evidences: List[Evidence],
    multimodal_answer_mode: str = "auto",
) -> str | list[Any]:
    types = _google_genai_types()
    if not _should_use_multimodal_answer_generation(query, evidences, multimodal_answer_mode):
        return user_message
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
    cite = _format_citation(file_name, parent_ev)

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
    evidence_summary: list[str] = field(default_factory=list)


def generate_chat_answer(
    query: str,
    gemini_api_key: str,
    gemini_model: str = "gemini-2.0-flash",
    gemini_fallback_model: str = "",
    chat_style: str = "neutral",
) -> str:
    """
    Chat-only generation (no retrieval, no citations).
    """
    system = CHAT_SYSTEM_PROMPT + chat_style_addendum(chat_style) + _language_addendum(query)
    (
        build_gemini_client,
        gemini_model_candidates,
        is_model_not_found_error,
        is_retryable_api_error,
    ) = _gemini_helpers()

    def _call_chat(user_contents: str, *, temperature: float = 0.4, max_tokens: int = 4096) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, 5):
            try:
                last_model_error: Optional[Exception] = None
                for model_name in gemini_model_candidates(gemini_model, fallback_model=gemini_fallback_model):
                    try:
                        client = build_gemini_client(gemini_api_key, model_name=model_name)
                        types = _google_genai_types()
                        resp = client.models.generate_content(
                            model=model_name,
                            contents=user_contents,
                            config=types.GenerateContentConfig(
                                system_instruction=system,
                                temperature=temperature,
                                max_output_tokens=max_tokens,
                            ),
                        )
                        return (resp.text or "").strip()
                    except Exception as model_exc:
                        last_model_error = model_exc
                        if not is_model_not_found_error(model_exc):
                            raise
                if last_model_error is not None:
                    raise last_model_error
            except Exception as e:
                last_err = e
                if attempt >= 4 or not is_retryable_api_error(e):
                    raise
                time.sleep(_retry_sleep_seconds(attempt))
        raise last_err  # type: ignore[misc]

    answer = _call_chat(f"SORU: {query}") or "Anlayamadım, tekrar eder misin?"
    answer = _complete_if_incomplete(
        answer,
        query=query,
        continue_fn=lambda prompt: _call_chat(prompt, temperature=0.2, max_tokens=4096),
    )
    return answer or "Anlayamadım, tekrar eder misin?"


def generate_chat_answer_openai(
    query: str,
    openai_api_key: str,
    openai_model: str = "gpt-4o-mini",
    chat_style: str = "neutral",
) -> str:
    """
    Chat-only generation via OpenAI Chat Completions.
    """
    system = CHAT_SYSTEM_PROMPT + chat_style_addendum(chat_style) + _language_addendum(query)

    def _call_chat(user_contents: str, *, temperature: float = 0.4, max_tokens: int = 4096) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, 5):
            try:
                return (
                    _openai_chat_completion(
                        api_key=openai_api_key,
                        model=openai_model,
                        system_instruction=system,
                        user_contents=user_contents,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    or ""
                )
            except Exception as e:
                last_err = e
                if attempt >= 4 or not _openai_retryable(e):
                    raise
                time.sleep(_retry_sleep_seconds(attempt))
        raise last_err  # type: ignore[misc]

    answer = _call_chat(f"SORU: {query}") or "Anlayamadım, tekrar eder misin?"
    answer = _complete_if_incomplete(
        answer,
        query=query,
        continue_fn=lambda prompt: _call_chat(prompt, temperature=0.2, max_tokens=4096),
    )
    return answer or "Anlayamadım, tekrar eder misin?"


# ── Main generation function ─────────────────────────────────────────────────

def generate_answer(
    retrieval: RetrievalResult,
    query: str,
    gemini_api_key: str,
    gemini_model: str = "gemini-2.0-flash",
    gemini_fallback_model: str = "",
    multimodal_answer_mode: str = "auto",
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
    system = SYSTEM_PROMPT_BASE + _language_addendum(query)
    coverage_expected: Optional[int] = None
    if retrieval.intent == "section_list" and retrieval.coverage:
        coverage_expected = retrieval.coverage.expected_items
        system += SECTION_LIST_ADDENDUM.format(expected=coverage_expected)
    elif retrieval.intent == "multi_section":
        system += MULTI_SECTION_ADDENDUM

    # Build user message with context
    user_message = (
        f"BAĞLAM:\n{context}\n\n"
        f"---\n\n"
        f"SORU: {query}"
    )

    gemini_contents = _build_gemini_user_contents(
        user_message,
        query,
        retrieval.evidences,
        multimodal_answer_mode=multimodal_answer_mode,
    )
    (
        build_gemini_client,
        gemini_model_candidates,
        is_model_not_found_error,
        is_retryable_api_error,
    ) = _gemini_helpers()

    def _call(
        system_instruction: str,
        user_contents: str | list[Any],
        temperature: float,
        max_tokens: int = 4096,
    ) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, 5):
            try:
                last_model_error: Optional[Exception] = None
                for model_name in gemini_model_candidates(gemini_model, fallback_model=gemini_fallback_model):
                    try:
                        client = build_gemini_client(gemini_api_key, model_name=model_name)
                        types = _google_genai_types()
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
                if attempt >= 4 or not is_retryable_api_error(e):
                    raise
                time.sleep(_retry_sleep_seconds(attempt))

        raise last_err  # type: ignore[misc]

    # Call Gemini
    answer = _call(system, gemini_contents, temperature=0.1) or "Belgede bu bilgi bulunamadı."
    if answer.strip() != "Belgede bu bilgi bulunamadı.":
        answer = _complete_if_incomplete(
            answer,
            query=query,
            continue_fn=lambda prompt: _call(system, prompt, temperature=0.0, max_tokens=1024),
        )

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

    # Post-generation grounding verification
    answer = _verify_answer_grounding(answer, context)

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
    system = SYSTEM_PROMPT_BASE + _language_addendum(query)
    coverage_expected: Optional[int] = None
    if retrieval.intent == "section_list" and retrieval.coverage:
        coverage_expected = retrieval.coverage.expected_items
        system += SECTION_LIST_ADDENDUM.format(expected=coverage_expected)
    elif retrieval.intent == "multi_section":
        system += MULTI_SECTION_ADDENDUM

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
    if answer.strip() != "Belgede bu bilgi bulunamadı.":
        answer = _complete_if_incomplete(
            answer,
            query=query,
            continue_fn=lambda prompt: _openai_chat_completion(
                api_key=openai_api_key,
                model=openai_model,
                system_instruction=system,
                user_contents=prompt,
                temperature=0.0,
                max_tokens=1024,
            ),
        )

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
    gemini_fallback_model: str = "",
    multimodal_answer_mode: str = "auto",
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
    system = SYSTEM_PROMPT_BASE + _language_addendum(query)
    coverage_expected: Optional[int] = None
    if retrieval.intent == "section_list" and retrieval.coverage:
        coverage_expected = retrieval.coverage.expected_items
        system += SECTION_LIST_ADDENDUM.format(expected=coverage_expected)
    elif retrieval.intent == "multi_section":
        system += MULTI_SECTION_ADDENDUM

    user_message = (
        f"BAĞLAM:\n{context}\n\n"
        f"---\n\n"
        f"SORU: {query}"
    )
    gemini_contents = _build_gemini_user_contents(
        user_message,
        query,
        retrieval.evidences,
        multimodal_answer_mode=multimodal_answer_mode,
    )
    (
        build_gemini_client,
        gemini_model_candidates,
        is_model_not_found_error,
        is_retryable_api_error,
    ) = _gemini_helpers()

    chunks: list[str] = []
    last_err: Optional[Exception] = None
    for attempt in range(1, 5):
        try:
            last_model_error: Optional[Exception] = None
            chunks = []
            for model_name in gemini_model_candidates(gemini_model, fallback_model=gemini_fallback_model):
                try:
                    client = build_gemini_client(gemini_api_key, model_name=model_name)
                    types = _google_genai_types()
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
            break
        except Exception as e:
            last_err = e
            if attempt >= 4 or chunks or not is_retryable_api_error(e):
                raise
            time.sleep(_retry_sleep_seconds(attempt))
    if not chunks and last_err is not None:
        raise last_err

    answer = "".join(chunks).strip() or "Belgede bu bilgi bulunamadı."
    if answer.strip() != "Belgede bu bilgi bulunamadı.":
        def _continue_call(prompt: str) -> str:
            last_err: Optional[Exception] = None
            for attempt in range(1, 5):
                try:
                    last_model_error_inner: Optional[Exception] = None
                    for model_name in gemini_model_candidates(gemini_model, fallback_model=gemini_fallback_model):
                        try:
                            client = build_gemini_client(gemini_api_key, model_name=model_name)
                            types = _google_genai_types()
                            resp = client.models.generate_content(
                                model=model_name,
                                contents=prompt,
                                config=types.GenerateContentConfig(
                                    system_instruction=system,
                                    temperature=0.0,
                                    max_output_tokens=1024,
                                ),
                            )
                            return (resp.text or "").strip()
                        except Exception as model_exc:
                            last_model_error_inner = model_exc
                            if not is_model_not_found_error(model_exc):
                                raise
                    if last_model_error_inner is not None:
                        raise last_model_error_inner
                except Exception as e:
                    last_err = e
                    if attempt >= 4 or not is_retryable_api_error(e):
                        raise
                    time.sleep(_retry_sleep_seconds(attempt))
            raise last_err  # type: ignore[misc]

        completed = _complete_if_incomplete(
            answer,
            query=query,
            continue_fn=_continue_call,
        )
        suffix = _continuation_suffix(answer, completed)
        if suffix:
            _emit(suffix)
            answer = completed

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
    system = SYSTEM_PROMPT_BASE + _language_addendum(query)
    coverage_expected: Optional[int] = None
    if retrieval.intent == "section_list" and retrieval.coverage:
        coverage_expected = retrieval.coverage.expected_items
        system += SECTION_LIST_ADDENDUM.format(expected=coverage_expected)
    elif retrieval.intent == "multi_section":
        system += MULTI_SECTION_ADDENDUM

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
    if answer.strip() != "Belgede bu bilgi bulunamadı.":
        completed = _complete_if_incomplete(
            answer,
            query=query,
            continue_fn=lambda prompt: _openai_chat_completion(
                api_key=openai_api_key,
                model=openai_model,
                system_instruction=system,
                user_contents=prompt,
                temperature=0.0,
                max_tokens=1024,
            ),
        )
        suffix = _continuation_suffix(answer, completed)
        if suffix:
            _emit(suffix)
            answer = completed

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

    system = CHAT_SYSTEM_PROMPT + chat_style_addendum(chat_style) + _language_addendum(query)
    def _call_chat(user_message: str, *, temperature: float = 0.4, max_tokens: int = 4096) -> str:
        return ollama_chat(
            cfg=ollama_cfg,
            system=system,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    result = _call_chat(f"SORU: {query}") or "Anlayamadım, tekrar eder misin?"
    result = _complete_if_incomplete(
        result,
        query=query,
        continue_fn=lambda prompt: _call_chat(prompt, temperature=0.2, max_tokens=4096),
    )
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
    system = SYSTEM_PROMPT_BASE + _language_addendum(query)
    coverage_expected: Optional[int] = None
    if retrieval.intent == "section_list" and retrieval.coverage:
        coverage_expected = retrieval.coverage.expected_items
        system += SECTION_LIST_ADDENDUM.format(expected=coverage_expected)
    elif retrieval.intent == "multi_section":
        system += MULTI_SECTION_ADDENDUM

    user_message = (
        f"BAĞLAM:\n{context}\n\n"
        f"---\n\n"
        f"SORU: {query}"
    )

    def _call_local(sys: str, msg: str, temp: float, max_tok: int = 4096) -> str:
        return ollama_chat(cfg=ollama_cfg, system=sys, user_message=msg, temperature=temp, max_tokens=max_tok)

    answer = _call_local(system, user_message, 0.1) or "Belgede bu bilgi bulunamadı."
    if answer.strip() != "Belgede bu bilgi bulunamadı.":
        answer = _complete_if_incomplete(
            answer,
            query=query,
            continue_fn=lambda prompt: _call_local(system, prompt, 0.0, 1024),
        )

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
    from .local_llm import OllamaConfig, ollama_chat, ollama_chat_stream  # noqa: F811

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
    system = SYSTEM_PROMPT_BASE + _language_addendum(query)
    coverage_expected: Optional[int] = None
    if retrieval.intent == "section_list" and retrieval.coverage:
        coverage_expected = retrieval.coverage.expected_items
        system += SECTION_LIST_ADDENDUM.format(expected=coverage_expected)
    elif retrieval.intent == "multi_section":
        system += MULTI_SECTION_ADDENDUM

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
    if answer.strip() != "Belgede bu bilgi bulunamadı.":
        completed = _complete_if_incomplete(
            answer,
            query=query,
            continue_fn=lambda prompt: ollama_chat(
                cfg=ollama_cfg,
                system=system,
                user_message=prompt,
                temperature=0.0,
                max_tokens=1024,
            ),
        )
        suffix = _continuation_suffix(answer, completed)
        if suffix:
            _emit(suffix)
            answer = completed

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
        cite = _format_citation(file_name, ev)
        # Deduplicate same page content
        key = f"{file_name}:{ev.page_start}:{ev.page_end}:{ev.region_label}:{ev.text[:80]}"
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
