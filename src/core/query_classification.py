"""
Query intent classification for the RAG retrieval pipeline.

Extracted from retrieval.py for modularity.
Contains: QueryIntent type, classify_query(), visual/section patterns,
heading relevance checking, and morphological query expansion.
"""
from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional, Set, Tuple


# ── Query Intent Type ─────────────────────────────────────────────────────────
QueryIntent = Literal["section_list", "multi_section", "normal_qa"]

# ── Patterns that signal "give me everything under a heading / list all X" ────
_SECTION_LIST_PATTERNS: list[re.Pattern[str]] = [
    # ── Turkish — direct question forms ──────────────────────────────────
    re.compile(r"nelerdir", re.IGNORECASE),
    re.compile(r"neler\b", re.IGNORECASE),
    re.compile(r"nedir.*(maddeleri|listesi|gereksinimleri|başlıkları)", re.IGNORECASE),
    re.compile(r"(listele|sırala|say\b|maddeleri)", re.IGNORECASE),

    # ── Turkish — "all items under X" style ──────────────────────────────
    re.compile(r"(altında(ki)?|içindeki)\s+(tüm|her|bütün)", re.IGNORECASE),
    re.compile(r"(tüm|bütün|hepsi|eksiksiz)\s+.*(madde|gereksinim|başlık|teslimat|adım)", re.IGNORECASE),
    re.compile(r"kaç\s+(madde|gereksinim|başlık|teslimat|adım|adet)", re.IGNORECASE),

    # ── Turkish — "X başlığı/bölümü altındakiler" ────────────────────────
    re.compile(
        r"(başlığ\w*|bölüm\w*|kısm\w*)\s+.*(altında\w*|içinde\w*|altındak\w*|içindek\w*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(bilgi(leri|sini|lerinin)|madde(leri|sini|lerinin)|satır(ları|larını))\s+.*(hepsini|tamamını|tümünü)\s+(ver|yaz|göster)",
        re.IGNORECASE,
    ),

    # ── Turkish — passive / formal / indirect question forms ─────────────
    re.compile(
        r"(hangi|ne\s+tür|ne\s+gibi)\s+\w+\s*(beklenmekte|bekleniyor|istenmekte|isteniyor|"
        r"belirtilmi[şs]|tanımlanmı[şs]|öngörülm[üu][şs]|planlanmı[şs]|yer\s+al[ıi]yor)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\w+(lar|ler)\s+(mevcut\s+mu|yer\s+al[ıi]yor\s+mu)",
        re.IGNORECASE,
    ),
    re.compile(r"(içeriyor|kapsıyor|kapsamında|kapsamındaki)", re.IGNORECASE),

    # ── English ──────────────────────────────────────────────────────────
    re.compile(r"what\s+are\s+(the|all)", re.IGNORECASE),
    re.compile(r"list\s+(all|the|every)", re.IGNORECASE),
    re.compile(r"(enumerate|summarize\s+all)", re.IGNORECASE),
    re.compile(r"how\s+many\s+(items?|requirements?|sections?|deliverables?)", re.IGNORECASE),
]

_MULTI_SECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(karşılaştır|kıyasla|mukayese)", re.IGNORECASE),
    re.compile(r"(arasındaki|arasında)\s+(fark|ilişki|bağlantı|benzerlik)", re.IGNORECASE),
    re.compile(r"(hem\s+.+\s+hem\s+)", re.IGNORECASE),
    re.compile(r"(ile|ve)\s+.*(arasında|karşılaştır|kıyasla)", re.IGNORECASE),
    re.compile(r"bölüm\s*\d+.*(ve|ile)\s*bölüm\s*\d+", re.IGNORECASE),
    re.compile(r"\d+\.\s*(ve|ile)\s*\d+\.\s*(bölüm|kısım|madde)", re.IGNORECASE),
    re.compile(r"\b(compare|comparison|versus|vs\.?)\b", re.IGNORECASE),
    re.compile(r"\b(difference|similarities?)\s+between\b", re.IGNORECASE),
    re.compile(r"\bboth\s+.+\s+and\s+", re.IGNORECASE),
    re.compile(r"section\s*\d+.*(and|vs)\s*section\s*\d+", re.IGNORECASE),
]

_VISUAL_REGION_QUERY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(tablo|table|satır|satir|row|sütun|sutun|column|hücre|hucre|cell)\b", re.IGNORECASE),
    re.compile(r"\b(form|alan|field|checkbox|label|değer|deger|value|imza|signature)\b", re.IGNORECASE),
    re.compile(r"\b(görsel|gorsel|resim|image|layout|yerleşim|yerlesim|sayfadaki|page|region|bölge|bolge)\b", re.IGNORECASE),
]


def classify_query(query: str) -> QueryIntent:
    """Rule-based intent classifier (document-agnostic)."""
    for pat in _MULTI_SECTION_PATTERNS:
        if pat.search(query):
            return "multi_section"
    for pat in _SECTION_LIST_PATTERNS:
        if pat.search(query):
            return "section_list"
    return "normal_qa"


def query_prefers_visual_region(query: str) -> bool:
    text = (query or "").strip()
    if not text:
        return False
    return any(pat.search(text) for pat in _VISUAL_REGION_QUERY_PATTERNS)


# ── Heading–query matching ────────────────────────────────────────────────────

def tokenize_simple(text: str) -> Set[str]:
    """Simple word tokenizer for TR/EN overlap matching."""
    text = text.lower()
    text = re.sub(r"[^0-9a-zçğıöşüâîû\-]+", " ", text)
    return {t for t in text.split() if len(t) > 1}


def heading_query_overlap(heading_path: str, query: str) -> float:
    """Compute how many query tokens appear in the heading path (0..1)."""
    q_tokens = tokenize_simple(query)
    h_tokens = tokenize_simple(heading_path)
    if not q_tokens:
        return 0.0
    overlap = q_tokens & h_tokens
    return len(overlap) / len(q_tokens)


# Common structural/question words to exclude from topic matching
_QUESTION_WORDS: Set[str] = {
    "nelerdir", "nedir", "nelerden", "nelerini", "nelerle",
    "listele", "listesi", "sirala",
    "kaç", "kac", "kadar",
    "nasıl", "nasil", "neden", "niçin", "nicin",
    "hepsini", "tamamını", "tamamini", "tümünü", "tumunu",
    "neler", "hangi", "hangisi", "hangileri",
    "mi", "mu", "mı",
    "beklenmekte", "bekleniyor", "istenmekte", "isteniyor",
    "belirtilmiş", "belirtilmis", "tanımlanmış", "tanimlanmis",
    "öngörülmüş", "ongorulmus", "planlanmış", "planlanmis",
    "içeriyor", "kapsamında", "kapsamindaki", "kapsıyor",
    "mevcut", "var",
    "what", "list", "all", "every", "how", "enumerate", "are", "the",
    "is", "which", "many", "much",
}


def topic_heading_relevant(query: str, heading_path: str, min_prefix: int = 5) -> bool:
    """
    Check that the query's topic words have meaningful lexical overlap
    with the section heading. Uses prefix-based matching for Turkish morphology.
    """
    heading = heading_path
    if " / " in heading:
        heading = heading.split(" / ", 1)[1]
    heading = (heading or "").strip()
    if len(heading) < 3:
        return True

    q_tokens = tokenize_simple(query)
    h_tokens = tokenize_simple(heading)
    if not q_tokens or not h_tokens:
        return True

    topic = {t for t in q_tokens if t not in _QUESTION_WORDS and len(t) > 2}
    if not topic:
        return True

    for qt in topic:
        found = False
        for ht in h_tokens:
            if qt == ht:
                found = True
                break
            pfx = min(min_prefix, min(len(qt), len(ht)))
            if pfx >= 4:
                common = 0
                for i in range(min(len(qt), len(ht))):
                    if qt[i] == ht[i]:
                        common += 1
                    else:
                        break
                if common >= pfx:
                    found = True
                    break
        if not found:
            return False
    return True


# ── Morphological query expansion ─────────────────────────────────────────────

_TR_SUFFIXES = [
    "lerinin", "larının", "lerinde", "larında",
    "lerini", "larını", "lerine", "larına",
    "lerin", "ların", "lerde", "larda",
    "leri", "ları", "ler", "lar",
    "nden", "ndan", "inin", "ının",
    "inde", "ında", "sine", "sına",
    "ini", "ını", "ine", "ına",
    "den", "dan", "nin", "nın",
    "de", "da", "ne", "na",
    "in", "ın", "si", "sı",
    "le", "la", "yi", "yı",
]


def expand_query_morphological(query: str) -> str:
    """Add morphologically stripped forms of Turkish words to the query."""
    words = re.findall(r"[a-zçğıöşüâîû]{4,}", query.lower())
    stems: Set[str] = set()
    for w in words:
        for sfx in _TR_SUFFIXES:
            if w.endswith(sfx) and len(w) - len(sfx) >= 3:
                stem = w[:-len(sfx)]
                if stem not in {w}:
                    stems.add(stem)
                break
    if not stems:
        return query
    return query + " " + " ".join(sorted(stems))
