"""
Ranking and scoring utilities for the RAG retrieval pipeline.

Extracted from retrieval.py for modularity.
Contains: region-aware scoring, embedding re-ranking, evidence filtering,
section selection, and coverage counting.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from .indexing import LocalIndex
from .query_classification import (
    heading_query_overlap,
    query_prefers_visual_region,
    tokenize_simple,
)

logger = logging.getLogger(__name__)


# ── Region-aware scoring ──────────────────────────────────────────────────────

def _page_number_from_meta(meta: dict) -> int:
    try:
        return int(meta.get("page_start", 0) or 0)
    except Exception:
        return 0


def _meta_is_region(meta: dict) -> bool:
    region_label = (meta.get("region_label", "") or "").strip()
    region_id = (meta.get("region_id", "") or "").strip()
    crop_type = (meta.get("crop_type", "page") or "page").strip().lower()
    return bool(region_label or region_id or crop_type != "page")


def _region_overlap_bonus(query: str, meta: dict) -> float:
    summary = (meta.get("region_summary", "") or "").strip()
    if not summary:
        return 0.0
    q_tokens = tokenize_simple(query)
    s_tokens = tokenize_simple(summary)
    if not q_tokens or not s_tokens:
        return 0.0
    overlap = len(q_tokens & s_tokens)
    if overlap <= 0:
        return 0.0
    return min(0.004, 0.002 * overlap)


def region_aware_scores(
    *,
    query: str,
    got_ids: List[str],
    got_metas: List[dict],
    hybrid_scores: Dict[str, float],
) -> Dict[str, float]:
    adjusted = dict(hybrid_scores)
    if not got_ids:
        return adjusted
    if not any(_meta_is_region(meta or {}) for meta in got_metas):
        return adjusted

    visual_query = query_prefers_visual_region(query)

    page_modalities: Dict[int, Set[str]] = {}
    for meta in got_metas:
        page_no = _page_number_from_meta(meta)
        if page_no <= 0:
            continue
        if not _meta_is_region(meta or {}):
            continue
        page_modalities.setdefault(page_no, set()).add(
            (meta.get("modality", "text") or "text").strip().lower()
        )

    cross_modal_pages = {
        page_no
        for page_no, mods in page_modalities.items()
        if "text" in mods and "visual" in mods
    }

    for cid, meta in zip(got_ids, got_metas):
        base = adjusted.get(cid, 0.0)
        modality = (meta.get("modality", "text") or "text").strip().lower()
        kind = (meta.get("kind", "") or "").strip().lower()
        crop_type = (meta.get("crop_type", "page") or "page").strip().lower()
        page_no = _page_number_from_meta(meta)
        bonus = 0.0

        if not _meta_is_region(meta or {}):
            adjusted[cid] = base
            continue

        if modality == "visual":
            if meta.get("region_label"):
                bonus += 0.002
            if crop_type != "page":
                bonus += 0.002
            if visual_query:
                bonus += 0.004
            if page_no in cross_modal_pages:
                bonus += 0.003
            bonus += _region_overlap_bonus(query, meta)
        elif kind == "table":
            bonus += 0.003
            if visual_query:
                bonus += 0.006
            bonus += _region_overlap_bonus(query, meta)
        elif visual_query and page_no in cross_modal_pages:
            bonus += 0.002

        adjusted[cid] = base + bonus

    return adjusted


# ── Embedding-based re-ranking ────────────────────────────────────────────────

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rerank_by_embedding(
    index: LocalIndex,
    query: str,
    chunk_ids: List[str],
    hybrid_scores: Dict[str, float],
    *,
    blend_weight: float = 0.6,
) -> Dict[str, float]:
    """
    Re-score hybrid results using embedding cosine similarity.
    Final score = blend_weight * cosine_sim + (1 - blend_weight) * normalized_rrf
    """
    if not chunk_ids:
        return hybrid_scores

    try:
        q_emb = index.embedder.embed_query(query)
    except Exception:
        logger.warning("Embedding re-rank failed: query embedding error", exc_info=True)
        return hybrid_scores

    try:
        col = index.store._get_collection()
        result = col.get(ids=chunk_ids, include=["embeddings"])
        chunk_embeddings = result.get("embeddings", [])
        chunk_ids_got = result.get("ids", [])
    except Exception:
        logger.warning("Embedding re-rank failed: chunk embedding fetch error", exc_info=True)
        return hybrid_scores

    if not chunk_embeddings or len(chunk_embeddings) != len(chunk_ids_got):
        return hybrid_scores

    cosine_scores: Dict[str, float] = {}
    for cid, emb in zip(chunk_ids_got, chunk_embeddings):
        if emb is not None and len(emb) > 0:
            cosine_scores[cid] = _cosine_similarity(q_emb, emb)
        else:
            cosine_scores[cid] = 0.0

    rrf_vals = [hybrid_scores.get(cid, 0.0) for cid in chunk_ids]
    rrf_max = max(rrf_vals) if rrf_vals else 1.0
    rrf_min = min(rrf_vals) if rrf_vals else 0.0
    rrf_range = rrf_max - rrf_min if rrf_max > rrf_min else 1.0

    blended: Dict[str, float] = {}
    for cid in chunk_ids:
        cos = cosine_scores.get(cid, 0.0)
        rrf_norm = (hybrid_scores.get(cid, 0.0) - rrf_min) / rrf_range
        blended[cid] = blend_weight * cos + (1 - blend_weight) * rrf_norm

    return blended


# ── Evidence filtering ────────────────────────────────────────────────────────

def filter_low_relevance(
    evidences: list,
    *,
    min_score_ratio: float = 0.25,
    min_keep: int = 3,
) -> list:
    """
    Remove evidence chunks whose score is far below the best score.
    Parent enrichment chunks (score < 0) are always kept if their section survives.
    """
    if len(evidences) <= min_keep:
        return evidences

    scored = [ev for ev in evidences if ev.score > 0.0]
    if not scored:
        return evidences

    max_score = max(ev.score for ev in scored)
    if max_score <= 0:
        return evidences

    threshold = max_score * min_score_ratio

    kept: list = []
    for ev in evidences:
        if ev.kind == "parent" and ev.score < 0:
            kept.append(ev)
            continue
        if ev.score >= threshold:
            kept.append(ev)

    if len(kept) < min_keep:
        kept = list(evidences[:min_keep])

    surviving_sections = {ev.section_id for ev in kept if ev.kind != "parent"}
    final: list = []
    for ev in kept:
        if ev.kind == "parent" and ev.score < 0 and ev.section_id not in surviving_sections:
            continue
        final.append(ev)

    return final if final else evidences[:min_keep]


# ── Section selection ─────────────────────────────────────────────────────────

def pick_best_section(
    got_ids: List[str],
    got_metas: List[dict],
    hybrid_scores: Dict[str, float],
    query: str,
) -> Optional[tuple[str, str]]:
    """
    From hybrid search results, pick the best section for section_list.
    Strategy: heading-first scoring + ancestor promotion.
    """
    candidates: Dict[tuple[str, str], Tuple[float, float, Optional[str], str]] = {}

    for cid, meta in zip(got_ids, got_metas):
        modality = (meta or {}).get("modality", "text") or "text"
        kind = (meta or {}).get("kind", "") or ""
        if modality == "visual" or kind in ("visual", "toc"):
            continue

        did = meta.get("doc_id", "")
        sid = meta.get("section_id", "")
        if not did or not sid or sid in ("root", "toc"):
            continue

        hp = meta.get("heading_path", "")
        hs = hybrid_scores.get(cid, 0.0)
        ho = heading_query_overlap(hp, query)
        pid = meta.get("parent_id") or None
        key = (did, sid)

        if key not in candidates or hs > candidates[key][0]:
            candidates[key] = (hs, ho, pid, hp)

    # Fallback: allow root
    if not candidates:
        for cid, meta in zip(got_ids, got_metas):
            modality = (meta or {}).get("modality", "text") or "text"
            kind = (meta or {}).get("kind", "") or ""
            if modality == "visual" or kind in ("visual", "toc"):
                continue
            did = meta.get("doc_id", "")
            sid = meta.get("section_id", "")
            if not did or sid != "root":
                continue
            hp = meta.get("heading_path", "")
            hs = hybrid_scores.get(cid, 0.0)
            ho = heading_query_overlap(hp, query)
            candidates[(did, sid)] = (hs, ho, None, hp)

        if not candidates:
            return None

    def _score(info: Tuple[float, float, Optional[str], str]) -> float:
        hs, ho, _, _ = info
        return ho + 0.3 * hs

    best_key = max(candidates, key=lambda k: _score(candidates[k]))

    # Ancestor promotion
    cur_key = best_key
    best_ho = candidates[best_key][1]
    for _ in range(3):
        _, _, parent_id_cur, _ = candidates[cur_key]
        if not parent_id_cur or parent_id_cur in ("root", "toc"):
            break
        parent_key = (cur_key[0], parent_id_cur)
        if parent_key not in candidates:
            break
        parent_ho = candidates[parent_key][1]
        if parent_ho >= best_ho:
            cur_key = parent_key
            best_ho = parent_ho
        else:
            break

    return cur_key


# ── Coverage counting ─────────────────────────────────────────────────────────

def count_list_items(text: str) -> int:
    """Heuristic: count bullet/numbered items, table rows, or structural patterns."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0

    first = lines[0]
    if re.match(r"^(\d+(?:\.\d+)*[.\)]\s|[A-Z]\.\d+(?:\.\d+)*\s)", first) or len(lines) > 1:
        lines = lines[1:]
    if not lines:
        return 0

    count = 0
    for ln in lines:
        if re.match(r"^(\d+[\.\)]\s|[a-zA-Z][\.\)]\s|[-•*]\s|[#]\s+\d)", ln):
            count += 1
    if count >= 2:
        return count

    # Indexed-table heuristic
    idx_rows = 0
    i = 0
    while i < len(lines) - 1:
        a = lines[i].strip()
        b = lines[i + 1].strip()
        if re.match(r"^\d{1,3}$", a) and not re.match(r"^\d{1,3}$", b):
            if 2 <= len(b) <= 120:
                idx_rows += 1
                i += 2
                continue
        i += 1
    if idx_rows >= 3:
        return idx_rows

    # Table-like row heuristic
    def _token_count(s: str) -> int:
        return len([t for t in re.split(r"\s+", s.strip()) if t])

    def _looks_like_label(s: str) -> bool:
        if not (3 <= len(s) <= 60):
            return False
        if s.endswith((".", "!", "?", ":", ";")):
            return False
        if re.match(r"^\d+$", s):
            return False
        return _token_count(s) <= 7

    def _looks_like_description(s: str) -> bool:
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

    # Sub-section headings
    heading_count = 0
    for ln in lines:
        if re.match(r"^[A-Z0-9]+\.\d+", ln) or re.match(r"^\d+\.\d+", ln):
            heading_count += 1
    if heading_count >= 3:
        return heading_count

    return count
