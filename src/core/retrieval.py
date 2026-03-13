from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

from .indexing import LocalIndex
from .models import Chunk


# ── 1) Query Classification ──────────────────────────────────────────────────
QueryIntent = Literal["section_list", "normal_qa"]

# Patterns that signal "give me everything under a heading / list all X"
_SECTION_LIST_PATTERNS: list[re.Pattern[str]] = [
    # Turkish
    re.compile(r"nelerdir", re.IGNORECASE),
    re.compile(r"nedir.*(maddeleri|listesi|gereksinimleri|başlıkları)", re.IGNORECASE),
    re.compile(r"(listele|sırala|say\b|maddeleri)", re.IGNORECASE),
    re.compile(r"(altında(ki)?|içindeki)\s+(tüm|her|bütün)", re.IGNORECASE),
    re.compile(r"(tüm|bütün|hepsi|eksiksiz)\s+.*(madde|gereksinim|başlık|teslimat|adım)", re.IGNORECASE),
    re.compile(r"kaç\s+(madde|gereksinim|başlık|teslimat|adım)", re.IGNORECASE),
    # "X başlığı/bölümü altındakiler" style requests
    re.compile(
        r"(başlığ\w*|bölüm\w*|kısm\w*)\s+.*(altında\w*|içinde\w*|altındak\w*|içindek\w*)",
        re.IGNORECASE,
    ),
    # "hepsini/tamamını ver" when it's clearly asking for all details/items
    re.compile(
        r"(bilgi(leri|sini|lerinin)|madde(leri|sini|lerinin)|satır(ları|larını))\s+.*(hepsini|tamamını|tümünü)\s+(ver|yaz|göster)",
        re.IGNORECASE,
    ),
    # English
    re.compile(r"what\s+are\s+(the|all)", re.IGNORECASE),
    re.compile(r"list\s+(all|the|every)", re.IGNORECASE),
    re.compile(r"(enumerate|summarize\s+all)", re.IGNORECASE),
    re.compile(r"how\s+many\s+(items?|requirements?|sections?)", re.IGNORECASE),
]


def classify_query(query: str) -> QueryIntent:
    """
    Rule-based intent classifier (document-agnostic).

    Returns "section_list" if the query looks like it wants a full list /
    section extraction; "normal_qa" otherwise.
    """
    for pat in _SECTION_LIST_PATTERNS:
        if pat.search(query):
            return "section_list"
    return "normal_qa"


def _section_fetch_max_depth() -> int:
    """
    Depth limit for section subtree fetch (document-agnostic).
    Default keeps current behavior (≈ children + grandchildren).
    """
    raw = (os.getenv("SECTION_FETCH_MAX_DEPTH", "2") or "").strip()
    try:
        n = int(raw)
    except Exception:
        n = 2
    # Clamp to avoid accidental explosions.
    return max(0, min(10, n))


# ── 2) Evidence gathering ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    text: str
    section_id: str
    heading_path: str
    page_start: int
    page_end: int
    kind: str  # parent / child
    score: float
    modality: str = "text"
    image_path: str = ""


@dataclass
class RetrievalResult:
    intent: QueryIntent
    evidences: List[Evidence]
    section_complete: bool  # True if we did complete-section fetch
    coverage: Optional[CoverageInfo] = None


@dataclass(frozen=True)
class CoverageInfo:
    expected_items: int
    heading_path: str
    section_id: str


def _meta_to_evidence(meta: dict, doc: str, chunk_id: str, score: float) -> Evidence:
    return Evidence(
        chunk_id=chunk_id,
        text=doc,
        section_id=meta.get("section_id", ""),
        heading_path=meta.get("heading_path", ""),
        page_start=meta.get("page_start", 0),
        page_end=meta.get("page_end", 0),
        kind=meta.get("kind", ""),
        score=score,
        modality=meta.get("modality", "text"),
        image_path=meta.get("image_path", ""),
    )


# ── Heading-aware section matching ───────────────────────────────────────────

def _tokenize_simple(text: str) -> Set[str]:
    """Simple word tokenizer for TR/EN overlap matching."""
    text = text.lower()
    text = re.sub(r"[^0-9a-zçğıöşüâîû\-]+", " ", text)
    return {t for t in text.split() if len(t) > 1}


def _heading_query_overlap(heading_path: str, query: str) -> float:
    """
    Compute how many query tokens appear in the heading path.
    Returns a score 0..1  (fraction of query tokens found in heading).
    """
    q_tokens = _tokenize_simple(query)
    h_tokens = _tokenize_simple(heading_path)
    if not q_tokens:
        return 0.0
    overlap = q_tokens & h_tokens
    return len(overlap) / len(q_tokens)


# ── Heading–query topic relevance (confidence guard) ─────────────────────────

# Common Turkish/English question and structural words that should NOT
# count as "topic" words when checking heading relevance.
_QUESTION_WORDS: Set[str] = {
    # Turkish
    "nelerdir", "nedir", "nelerden", "nelerini", "nelerle",
    "listele", "listesi", "sirala",
    "kaç", "kac", "kadar",
    "nasıl", "nasil", "neden", "niçin", "nicin",
    "hepsini", "tamamını", "tamamini", "tümünü", "tumunu",
    "neler", "hangi", "hangisi", "hangileri",
    "mi", "mu",
    # English
    "what", "list", "all", "every", "how", "enumerate", "are", "the",
    "is", "which", "many", "much",
}


def _topic_heading_relevant(query: str, heading_path: str, min_prefix: int = 5) -> bool:
    """
    Check that the query's **topic words** have meaningful lexical overlap
    with the section heading.

    Uses prefix-based matching (minimum *min_prefix* shared leading chars)
    to tolerate Turkish morphological suffixes
    (e.g. "gereksinimleri" ≈ "gereksinimler").

    Returns ``True`` (= pass) when the heading is missing/short, when the
    query has no extractable topic words, or when **all** topic words match
    a heading token.  Returns ``False`` only when at least one topic word
    has **no** counterpart in the heading, indicating a likely keyword
    false-positive.
    """
    # Strip the file-name prefix:
    #   "Case_Study.pdf / 2. Heading / 2.1 Sub"  →  "2. Heading / 2.1 Sub"
    heading = heading_path
    if " / " in heading:
        heading = heading.split(" / ", 1)[1]
    heading = (heading or "").strip()
    if len(heading) < 3:
        return True  # too short to compare meaningfully

    q_tokens = _tokenize_simple(query)
    h_tokens = _tokenize_simple(heading)
    if not q_tokens or not h_tokens:
        return True

    # Keep only topic words (remove question / structural words + very
    # short tokens that are likely suffixes / noise).
    topic = {t for t in q_tokens if t not in _QUESTION_WORDS and len(t) > 2}
    if not topic:
        return True  # query is all question words → don't filter

    # Check every topic word for a match in the heading.
    for qt in topic:
        found = False
        for ht in h_tokens:
            # 1) Exact match
            if qt == ht:
                found = True
                break
            # 2) Prefix match (handles Turkish suffixes)
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
            return False  # at least one topic word has no heading counterpart

    return True


def _pick_best_section(
    got_ids: List[str],
    got_metas: List[dict],
    hybrid_scores: Dict[str, float],
    query: str,
) -> Optional[tuple[str, str]]:
    """
    From the hybrid search results, pick the best section_id.
    Strategy: combine hybrid score with heading-overlap bonus.
    This prevents "teslimatlar nelerdir" from matching "Teknik Yaklaşım" just
    because it ranks slightly higher in the embedding space.
    """
    # Collect unique section candidates
    # (doc_id, section_id) -> (best_hybrid_score, heading_overlap)
    candidates: Dict[tuple[str, str], Tuple[float, float]] = {}
    for cid, meta in zip(got_ids, got_metas):
        if (meta or {}).get("modality") == "visual" or (meta or {}).get("kind") == "visual":
            continue
        did = meta.get("doc_id", "")
        sid = meta.get("section_id", "")
        if not did or not sid or sid == "root":
            continue
        hp = meta.get("heading_path", "")
        hs = hybrid_scores.get(cid, 0.0)
        ho = _heading_query_overlap(hp, query)
        key = (did, sid)
        if key not in candidates or hs > candidates[key][0]:
            candidates[key] = (hs, ho)

    if not candidates:
        # Fallback: some documents have no detected headings; allow selecting root.
        for cid, meta in zip(got_ids, got_metas):
            if (meta or {}).get("modality") == "visual" or (meta or {}).get("kind") == "visual":
                continue
            did = meta.get("doc_id", "")
            sid = meta.get("section_id", "")
            if not did or sid != "root":
                continue
            hp = meta.get("heading_path", "")
            hs = hybrid_scores.get(cid, 0.0)
            ho = _heading_query_overlap(hp, query)
            candidates[(did, sid)] = (hs, ho)

        if not candidates:
            return None

    # Score = hybrid_score + 0.5 * heading_overlap
    # The 0.5 bonus is enough to nudge a heading-matching section to the top
    # when hybrid scores are close, but won't override a vastly better match.
    best_key = max(candidates, key=lambda k: candidates[k][0] + 0.5 * candidates[k][1])
    return best_key


# ── Section + subtree fetch ──────────────────────────────────────────────────

def _fetch_section_and_subtree(
    index: LocalIndex,
    doc_id: str,
    section_id: str,
    *,
    max_depth: int = 2,
) -> List[Evidence]:
    """
    Fetch the section's own chunks (parent + children) AND all chunks whose
    parent_id equals this section_id (the subtree).  This ensures that asking
    for "4. Teslimatlar" also brings in 4.1, 4.2, etc.
    """
    col = index.store._get_collection()

    # 1) Chunks belonging directly to this section
    own = col.get(
        where={"$and": [{"doc_id": doc_id}, {"section_id": section_id}]},
        include=["documents", "metadatas"],
    )

    # 2) Subtree fetch (depth-limited BFS on section hierarchy)
    # We walk "section_id -> child sections" via:
    #   parent chunk meta: parent_id == <parent_section_id>
    # and we pull all chunks where parent_id == <current_section_id>, which includes:
    #   - child chunks of the current section
    #   - parent chunks of immediate sub-sections
    max_depth = max(0, int(max_depth))
    subtree_results = []
    frontier: Set[str] = {section_id}
    seen_sections: Set[str] = set(frontier)

    for _depth in range(max_depth):
        if not frontier:
            break

        next_frontier: Set[str] = set()
        for cur_sid in sorted(frontier):
            res = col.get(
                where={"$and": [{"doc_id": doc_id}, {"parent_id": cur_sid}]},
                include=["documents", "metadatas"],
            )
            subtree_results.append(res)

            # Discover immediate sub-sections: only parent chunks qualify as section nodes.
            for meta in res.get("metadatas", []) or []:
                try:
                    if (meta or {}).get("kind") != "parent":
                        continue
                    sid = (meta or {}).get("section_id", "")
                    pid = (meta or {}).get("parent_id", "")
                    if not sid or sid == cur_sid:
                        continue
                    if pid != cur_sid:
                        continue
                    if sid in seen_sections:
                        continue
                    seen_sections.add(sid)
                    next_frontier.add(sid)
                except Exception:
                    continue

        frontier = next_frontier

    # Merge all into a single evidence list (deduplicated)
    seen: Set[str] = set()
    evidences: List[Evidence] = []

    for result_set in [own] + subtree_results:
        ids = result_set.get("ids", [])
        docs = result_set.get("documents", [])
        metas = result_set.get("metadatas", [])
        for cid, doc, meta in zip(ids, docs, metas):
            if cid not in seen:
                seen.add(cid)
                evidences.append(_meta_to_evidence(meta, doc or "", cid, score=1.0))

    return evidences


# ── Coverage counting ────────────────────────────────────────────────────────

def _count_list_items(text: str) -> int:
    """
    Heuristic: count bullet/numbered items, table rows, or repeated
    structural patterns in a section text.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0

    # Ignore the first line if it looks like a section heading/title.
    # Parent chunks usually start with the section title (e.g., "2. Foo").
    first = lines[0]
    if re.match(r"^(\d+(?:\.\d+)*[.\)]\s|[A-Z]\.\d+(?:\.\d+)*\s)", first) or len(lines) > 1:
        lines = lines[1:]
    if not lines:
        return 0

    count = 0

    # 1) Numbered items: "1.", "1)", "- ", "• ", "* ", "a)", "a."
    for ln in lines:
        if re.match(r"^(\d+[\.\)]\s|[a-zA-Z][\.\)]\s|[-•*]\s|[#]\s+\d)", ln):
            count += 1

    if count >= 2:
        return count

    # 1.5) Indexed-table heuristic (purely structural):
    # Many PDF tables extract into patterns like:
    #   1
    #   DEVLOG.md
    #   Description...
    #   2
    #   TESTING.md
    #   Description...
    #
    # We count the number of numeric row indices that are followed by a non-numeric label.
    idx_rows = 0
    i = 0
    while i < len(lines) - 1:
        a = lines[i].strip()
        b = lines[i + 1].strip()
        if re.match(r"^\d{1,3}$", a) and not re.match(r"^\d{1,3}$", b):
            # Basic sanity: label shouldn't be extremely long
            if 2 <= len(b) <= 120:
                idx_rows += 1
                i += 2
                continue
        i += 1

    if idx_rows >= 3:
        return idx_rows

    # 2) Table-like row heuristic (purely structural, no vocabulary lists):
    #    Count patterns like:
    #      <short label line>
    #      <longer description line>
    #
    # High-confidence only: if we detect fewer than 3 rows, return 0 to avoid false alarms.
    def _token_count(s: str) -> int:
        return len([t for t in re.split(r"\s+", s.strip()) if t])

    def _looks_like_label(s: str) -> bool:
        if not (3 <= len(s) <= 60):
            return False
        if s.endswith((".", "!", "?", ":", ";")):
            return False
        if re.match(r"^\d+$", s):  # pure number lines are often table row indices
            return False
        # Labels tend to be short phrases (few tokens)
        return _token_count(s) <= 7

    def _looks_like_description(s: str) -> bool:
        # Descriptions tend to be longer and/or sentence-like.
        if len(s) >= 80:
            return True
        if _token_count(s) >= 10:
            return True
        if any(p in s for p in (".", ";", ":")) and _token_count(s) >= 6:
            return True
        return False

    rows = 0
    i = 0
    while i < len(lines) - 1:
        a = lines[i]
        b = lines[i + 1]

        # Structural header guard: if we see multiple short lines in a row, it's likely a header block.
        if _looks_like_label(a) and _looks_like_label(b) and not _looks_like_description(b):
            i += 1
            continue

        if _looks_like_label(a) and _looks_like_description(b):
            rows += 1
            i += 2
        else:
            i += 1

    if rows >= 3:
        return rows

    # 3) Count sub-section headings within the text
    heading_count = 0
    for ln in lines:
        if re.match(r"^[A-Z0-9]+\.\d+", ln) or re.match(r"^\d+\.\d+", ln):
            heading_count += 1
    if heading_count >= 3:
        return heading_count

    return count


# ── 3) Main retrieval pipeline ───────────────────────────────────────────────

def retrieve(
    index: LocalIndex,
    query: str,
    dense_k: int = 10,
    sparse_k: int = 10,
    final_k: int = 8,
    doc_id: Optional[str] = None,
) -> RetrievalResult:
    """
    Full retrieval pipeline with query routing.

    1. Classify intent
    2. Hybrid search to find best-matching section
    3. If section_list → heading-aware section selection → complete subtree
       fetch + coverage info
    4. If normal_qa → return top-k evidence
    """
    intent = classify_query(query)

    # Always start with hybrid search to locate the best section
    doc_ids = {doc_id} if doc_id else None
    hybrid = index.hybrid_search(
        query,
        dense_k=dense_k,
        sparse_k=sparse_k,
        final_k=final_k,
        doc_ids=doc_ids,
    )

    if not hybrid.ids:
        return RetrievalResult(
            intent=intent,
            evidences=[],
            section_complete=False,
        )

    # Get metadata for top results
    got = index.store.get(hybrid.ids)
    got_ids = got.get("ids", [])
    got_docs = got.get("documents", [])
    got_metas = got.get("metadatas", [])

    if intent == "section_list":
        best = _pick_best_section(
            got_ids, got_metas, hybrid.scores, query,
        )

        if best:
            best_doc_id, best_section_id = best

            # ── Confidence guard ─────────────────────────────────────
            # Verify that the query topic semantically matches the
            # selected section heading.  Low similarity indicates a
            # keyword false-positive (e.g. "sunucu gereksinimleri"
            # matched "Fonksiyonel Gereksinimler" via the shared
            # stem "gereksinim").
            #
            # When confidence is low we skip the deterministic section
            # fetch and fall through to the normal-QA / LLM path,
            # which can decide "Belgede bu bilgi bulunamadı."
            heading_path_str = ""
            for meta in got_metas:
                if meta.get("section_id") == best_section_id and meta.get("heading_path"):
                    heading_path_str = meta["heading_path"]
                    break

            if _topic_heading_relevant(query, heading_path_str):
                # High confidence → complete section + subtree fetch
                section_evidences = _fetch_section_and_subtree(
                    index,
                    best_doc_id,
                    best_section_id,
                    max_depth=_section_fetch_max_depth(),
                )

                # Coverage info from the parent chunk text
                coverage = None
                for ev in section_evidences:
                    if ev.kind == "parent" and ev.section_id == best_section_id:
                        n = _count_list_items(ev.text)
                        if n > 0:
                            coverage = CoverageInfo(
                                expected_items=n,
                                heading_path=ev.heading_path,
                                section_id=ev.section_id,
                            )
                        break

                return RetrievalResult(
                    intent=intent,
                    evidences=section_evidences,
                    section_complete=True,
                    coverage=coverage,
                )
            # else: low confidence → fall through to normal-QA evidence

    # Normal QA: return hybrid top-k evidence
    evidences: List[Evidence] = []
    for cid, doc, meta in zip(got_ids, got_docs, got_metas):
        score = hybrid.scores.get(cid, 0.0)
        evidences.append(_meta_to_evidence(meta, doc or "", cid, score))

    return RetrievalResult(
        intent=intent,
        evidences=evidences,
        section_complete=False,
    )
