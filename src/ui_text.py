from __future__ import annotations

import re
from typing import Any, Sequence


_SMALLTALK_PATTERNS = [
    r"^(merhaba|selam|slm|selamlar)\b",
    r"\bnasılsın\b|\bnaber\b|\bnasılsınız\b",
    r"^(teşekkür(ler)?|tesekkur(ler)?|sağ ol|sagol|eyvallah|rica ederim)\b",
    r"^(günaydın|iyi akşamlar|iyi geceler|iyi günler)\b",
    r"\bkimsin\b|\bsen kimsin\b|\bne yapıyorsun\b",
    r"\bben\s+nas\w*ls\w*m\b",
    r"\bsorm\w*\s+m\w*s\w*n\b",
    r"\bemin\s+m\w*s\w*n\b|\bgercekten\s+mi\b|\bciddi\s+misin\b",
    r"^(hi|hello|hey)\b",
    r"\bhow are you\b|\bhow's it going\b",
    r"^(thanks|thank you)\b",
    r"\bwho are you\b",
    r"\bare you sure\b|\breally\??\b",
]

_PRAISE_PATTERNS = [
    r"^(aferin|bravo|helal|tebrik(ler)?|güzel|guzel|iyi\s*i[şs])\b",
    r"\b(harikasın|harikasin|mükemmel|mukemmel|süpersin|supersin|kralsın|kralsin)\b",
    r"\b(great job|well done|nice work|awesome|you are awesome|you're awesome|congrats)\b",
]

_NEGATIVE_FEELING_PATTERNS = [
    r"\b(üzgünüm|uzgunum|moralim bozuk|canım sıkkın|canim sikkin)\b",
    r"\b(kötüyüm|kotuyum|berbatım|berbatim|cok kotuyum|çok kötüyüm)\b",
    r"\b(stresliyim|kaygılıyım|kaygiliyim|endişeliyim|endiseliyim|yoruldum|bıktım|biktim)\b",
    r"\b(i am sad|i'm sad|i feel bad|i am upset|i'm upset|bad day|feeling down)\b",
]

_CHAT_MODE_REQUEST_PATTERNS = [
    r"\bsohbet\s+modu\b",
    r"\bsohbet\s+moduna\b.*\b(geç|gec|aç|ac)\w*\b",
    r"\bchat\s+modu\b",
    r"\bchat\s+moduna\b.*\b(geç|gec)\w*\b",
    r"\bsohbete\b.*\b(geç|gec)\w*\b",
    r"\bchat\s+mode\b",
    r"\bswitch\s+to\s+chat\b",
]

_DOC_CUE_PATTERNS = [
    r"\bbelge\b|\bdoküman\b|\bdokuman\b|\bpdf\b|\bdosya\b",
    r"\bsayfa\b|\bbaşlık\b|\bbölüm\b|\bmadde\b|\biçerik\b|\bicerik\b",
    r"\bnelerdir\b|\blistele\b|\bsırala\b|\bsirala\b|\bhepsi\b|\btümü\b|\btumu\b",
]

_DOC_MODE_REQUEST_PATTERNS = [
    r"\bbelge\s+modu\b|\bdoküman\s+modu\b|\bdokuman\s+modu\b",
    r"\bbelge\s+moduna\b.*\b(dön|don|geç|gec)\w*\b",
    r"\bbelge\s+moduna\s+nasıl\b|\bbelge\s+moduna\s+nas\w*l\b",
    r"\bbelge\s+moduna\s+nas\w*l\s+d\w*n\w*",
    r"\bdoc\s+mode\b|\bdocument\s+mode\b",
]


def shorten_for_sidebar(text: str, limit: int = 84) -> str:
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)] + "..."


def embedding_runtime_label(model_name: str | None, device: str | None) -> str:
    model = (model_name or "").strip().lower()
    dev = (device or "").strip().lower() or "-"
    if model.startswith("gemini-embedding-"):
        return "remote api"
    return dev


def _sidebar_chip(text: str | None) -> str:
    return f"`{(text or '-').strip()}`"


def render_sidebar_panel(
    *,
    mode: str,
    llm_provider: str,
    llm_model: str,
    embedding_model: str,
    embedding_runtime: str,
    ocr_backend: str,
    ocr_enabled: str,
    vlm_provider: str,
    vlm_mode: str,
    vlm_max_pages: str,
    visual_chunk_level: str,
    visual_region_source: str,
    visual_detector_backend: str,
    active_doc: str | None,
    docs: Sequence[str],
) -> str:
    loaded_docs = "\n".join(f"- {doc}" for doc in list(docs)[:6]) if docs else "- Henüz belge yüklenmedi"
    active_doc_text = active_doc or "Aktif belge yok"
    return "\n".join(
        [
            "**Oturum Durumu**",
            f"- **Mod**: {_sidebar_chip(mode)}",
            f"- **LLM**: {_sidebar_chip(llm_provider)} {_sidebar_chip(llm_model)}",
            f"- **Embedding**: {_sidebar_chip(embedding_model)} {_sidebar_chip(embedding_runtime)}",
            f"- **OCR**: {_sidebar_chip(ocr_backend)} {_sidebar_chip(ocr_enabled)}",
            f"- **VLM**: {_sidebar_chip(vlm_provider)} {_sidebar_chip(vlm_mode)} {_sidebar_chip(f'pages {vlm_max_pages}')}",
            f"- **Visual Level**: {_sidebar_chip(visual_chunk_level)}",
            f"- **Region Source**: {_sidebar_chip(visual_region_source)}",
            f"- **Detector Backend**: {_sidebar_chip(visual_detector_backend)}",
            "",
            "---",
            "",
            "**Belge Bağlamı**",
            f"- **Aktif Belge**: {active_doc_text}",
            "- **Yüklenen Belgeler**:",
            loaded_docs,
            "",
            "---",
            "",
            "**Hızlı Komutlar**",
            f"- Aktif belge seç: {_sidebar_chip('/use <dosya>')}",
            f"- Belge modu: {_sidebar_chip('/doc')}",
            f"- Sohbet modu: {_sidebar_chip('/chat')}",
        ]
    )


def looks_like_doc_mode_request(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    return any(re.search(pat, q, flags=re.IGNORECASE) for pat in _DOC_MODE_REQUEST_PATTERNS)


def looks_like_chat_mode_request(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    return any(re.search(pat, q, flags=re.IGNORECASE) for pat in _CHAT_MODE_REQUEST_PATTERNS)


def looks_like_doc_switch(query: str, *, has_documents: bool, document_names: Sequence[str]) -> bool:
    if not has_documents:
        return False
    q = (query or "").strip().lower()
    if not q:
        return False
    for name in document_names:
        if name and name.lower() in q:
            return True
    return any(re.search(pat, q, flags=re.IGNORECASE) for pat in _DOC_CUE_PATTERNS[:2])


def looks_like_smalltalk(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    if any(re.search(pat, q, flags=re.IGNORECASE) for pat in _DOC_CUE_PATTERNS):
        return False
    if len(q) > 80:
        return False
    return any(re.search(pat, q, flags=re.IGNORECASE) for pat in (_SMALLTALK_PATTERNS + _PRAISE_PATTERNS + _NEGATIVE_FEELING_PATTERNS))


def smalltalk_style(query: str) -> str:
    q = (query or "").strip().lower()
    if not q:
        return "neutral"
    if any(re.search(pat, q, flags=re.IGNORECASE) for pat in _DOC_CUE_PATTERNS):
        return "neutral"
    if any(re.search(pat, q, flags=re.IGNORECASE) for pat in _NEGATIVE_FEELING_PATTERNS):
        return "empathetic"
    if any(re.search(pat, q, flags=re.IGNORECASE) for pat in _PRAISE_PATTERNS):
        return "congratulatory"
    return "neutral"


def build_qa_debug_suffix(result: Any, mode: str) -> str:
    debug_lines = [
        f"- **Mod**: {mode}",
        f"- **Intent**: {result.intent}",
        f"- **Citation sayisi**: {result.citations_found}",
    ]
    if result.coverage_expected is not None:
        status_emoji = "OK" if result.coverage_ok else "EKSIK"
        debug_lines.append(
            f"- **Kapsam**: beklenen={result.coverage_expected}, "
            f"bulunan={result.coverage_actual}, durum={status_emoji}"
        )
    debug_text = "\n".join(debug_lines)
    return (
        f"\n\n"
        f"---\n"
        f"<details><summary>Debug Bilgisi</summary>\n\n"
        f"{debug_text}\n\n"
        f"</details>"
    )


def build_evidence_panel(result: Any) -> str:
    evidence_summary = getattr(result, "evidence_summary", None) or []
    if not evidence_summary:
        return ""
    return "\n".join(["**Kullanilan Kanitlar**", "", *evidence_summary])


def format_standard_error(title: str, err: Exception | str) -> str:
    detail = re.sub(r"\s+", " ", str(err or "")).strip() or "Bilinmeyen hata"
    if len(detail) > 320:
        detail = detail[:320] + "..."
    return (
        f"**{title}**\n"
        f"- Islem tamamlanamadi.\n"
        f"- Detay: `{detail}`\n"
        f"- `Tekrar dene` ile ayni istegi yeniden calistirabilirsin."
    )
