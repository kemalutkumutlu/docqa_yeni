from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

from .indexing import LocalIndex
from .models import Chunk

logger = logging.getLogger(__name__)

from .query_classification import (
    QueryIntent,
    classify_query,
    query_prefers_visual_region,
    topic_heading_relevant,
    expand_query_morphological,
)
from .evidence import (
    Evidence,
    RetrievalResult,
    CoverageInfo,
    meta_to_evidence,
    fetch_section_and_subtree,
    enrich_with_parent_context,
)
from .ranking import (
    region_aware_scores,
    rerank_by_embedding,
    filter_low_relevance,
    pick_best_section,
    count_list_items,
)

# ── Query expansion ─────────────────────────────────────────────────────────

def _multi_query_expand(
    query: str,
    *,
    gemini_api_key: str = "",
    gemini_model: str = "",
) -> List[str]:
    """
    Use Gemini to generate 2 alternative query formulations.
    Returns [original_query, alt1, alt2].
    Falls back to [original_query] on any error.
    """
    import os
    api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    model = gemini_model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    if not api_key:
        return [query]

    try:
        from .gemini_client import build_gemini_client
        client = build_gemini_client(api_key=api_key)
        prompt = (
            "Aşağıdaki soruyu bir belge arama sistemi için 2 farklı şekilde yeniden formüle et. "
            "Her formülasyon aynı bilgiyi bulmaya yönelik olsun ama farklı kelimeler kullansın. "
            "Sadece 2 alternatif sorgu yaz, her biri ayrı satırda. Başka bir şey yazma.\n\n"
            f"Orijinal: {query}"
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = (response.text or "").strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and len(ln.strip()) > 5]
        return [query] + lines[:2]
    except Exception:
        logger.warning("Multi-query expansion failed, using original query only", exc_info=True)
        return [query]


# ── 3) Main retrieval pipeline ───────────────────────────────────────────────

def retrieve(
    index: LocalIndex,
    query: str,
    dense_k: int = 10,
    sparse_k: int = 10,
    final_k: int = 8,
    doc_id: Optional[str] = None,
    section_fetch_max_depth: int = 2,
    *,
    rerank_blend_weight: float = 0.6,
    relevance_min_score_ratio: float = 0.25,
    relevance_min_keep: int = 3,
    grounding_min_avg_score: float = 0.15,
    multi_section_max: int = 3,
    query_expansion_enabled: bool = True,
    toc_text_min_chars: int = 120,
    empty_section_min_chars: int = 30,
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

    # Morphological expansion improves BM25 recall for Turkish
    expanded_query = expand_query_morphological(query)

    # Optional: Gemini-based multi-query expansion for broader recall
    if query_expansion_enabled:
        import os
        alt_queries = _multi_query_expand(
            query,
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        )
    else:
        alt_queries = [query]

    # Always start with hybrid search to locate the best section
    doc_ids = {doc_id} if doc_id else None

    # Run hybrid search for each query variant, merge results
    if len(alt_queries) > 1:
        from collections import defaultdict
        merged_scores: Dict[str, float] = defaultdict(float)
        merged_ids_set: set[str] = set()
        for i, aq in enumerate(alt_queries):
            aq_expanded = expand_query_morphological(aq) if i > 0 else expanded_query
            h = index.hybrid_search(
                aq_expanded,
                dense_k=dense_k,
                sparse_k=sparse_k,
                final_k=final_k,
                doc_ids=doc_ids,
            )
            weight = 1.0 if i == 0 else 0.5  # original query gets higher weight
            for cid in h.ids:
                merged_scores[cid] = max(merged_scores[cid], h.scores.get(cid, 0.0) * weight)
                merged_ids_set.add(cid)
        # Sort by score and take top final_k
        sorted_merged = sorted(merged_ids_set, key=lambda c: merged_scores[c], reverse=True)[:final_k]
        from .hybrid import HybridResult
        hybrid = HybridResult(ids=sorted_merged, scores={c: merged_scores[c] for c in sorted_merged})
    else:
        hybrid = index.hybrid_search(
            expanded_query,
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
    adjusted_scores = region_aware_scores(
        query=query,
        got_ids=got_ids,
        got_metas=got_metas,
        hybrid_scores=hybrid.scores,
    )

    # Re-rank using embedding cosine similarity (recovers magnitude lost by RRF)
    # Use ORIGINAL query (not expanded) for re-ranking — embedding captures semantics
    reranked_scores = rerank_by_embedding(
        index, query, got_ids, adjusted_scores,
        blend_weight=rerank_blend_weight,
    )

    ranked = sorted(
        zip(got_ids, got_docs, got_metas),
        key=lambda item: reranked_scores.get(item[0], 0.0),
        reverse=True,
    )
    if ranked:
        got_ids = [item[0] for item in ranked]
        got_docs = [item[1] for item in ranked]
        got_metas = [item[2] for item in ranked]

    # Determine if the query wants visual content (tables, figures, forms etc.)
    # If so, don't filter out visual chunks from section fetches.
    _wants_visual = query_prefers_visual_region(query)
    _text_only = not _wants_visual

    # ── Multi-section: fetch top-N distinct sections ───────────────────
    if intent == "multi_section":
        _depth = max(0, min(10, int(section_fetch_max_depth)))
        seen_sids: Set[str] = set()
        all_section_evidences: List[Evidence] = []
        max_sections = multi_section_max

        for _cid, _meta in zip(got_ids, got_metas):
            modality = (_meta or {}).get("modality", "text") or "text"
            kind = (_meta or {}).get("kind", "") or ""
            if modality == "visual" or kind in ("visual", "toc"):
                continue
            did = (_meta or {}).get("doc_id", "")
            sid = (_meta or {}).get("section_id", "")
            if not did or not sid or sid in ("root", "toc") or sid in seen_sids:
                continue
            seen_sids.add(sid)
            evs = fetch_section_and_subtree(
                index, did, sid, max_depth=_depth, text_only=_text_only,
            )
            all_section_evidences.extend(evs)
            if len(seen_sids) >= max_sections:
                break

        if all_section_evidences:
            return RetrievalResult(
                intent=intent,
                evidences=all_section_evidences,
                section_complete=True,
            )
        # else: fall through to normal_qa

    # ── Section list: single best section + subtree ──────────────────────
    if intent == "section_list":
        best = pick_best_section(
            got_ids, got_metas, reranked_scores, query,
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

            if topic_heading_relevant(query, heading_path_str):
                # High confidence → complete section + subtree fetch
                # text_only is dynamic: True for text queries, False when
                # the user asks about visual content (tables, figures, etc.).
                _depth = max(0, min(10, int(section_fetch_max_depth)))
                section_evidences = fetch_section_and_subtree(
                    index,
                    best_doc_id,
                    best_section_id,
                    max_depth=_depth,
                    text_only=_text_only,
                )

                # TOC-entry guard: if the fetched section has almost no
                # meaningful text (e.g. a table-of-contents row like
                # "2. Fonksiyonel Gereksinimler (s.2)"), try sibling
                # sections that share the same heading topic before
                # falling through to normal-QA.
                _total_text = sum(len(ev.text) for ev in section_evidences)
                if _total_text < toc_text_min_chars:
                    _seen_sids: Set[str] = {best_section_id}
                    _best_alt: List[Evidence] = []
                    _best_alt_len = _total_text
                    for _cid, _meta in zip(got_ids, got_metas):
                        _sid = (_meta or {}).get("section_id", "")
                        if not _sid or _sid in _seen_sids:
                            continue
                        if (_meta or {}).get("modality") == "visual":
                            continue
                        _seen_sids.add(_sid)
                        _hp = (_meta or {}).get("heading_path", "")
                        if topic_heading_relevant(query, _hp):
                            _evs = fetch_section_and_subtree(
                                index,
                                best_doc_id,
                                _sid,
                                max_depth=_depth,
                                text_only=_text_only,
                            )
                            _evs_len = sum(len(e.text) for e in _evs)
                            if _evs_len > _best_alt_len:
                                _best_alt = _evs
                                _best_alt_len = _evs_len
                    if _best_alt:
                        section_evidences = _best_alt

                # Re-check: if still empty/tiny, fall through to normal-QA
                if sum(len(ev.text) for ev in section_evidences) < empty_section_min_chars:
                    pass  # fall through
                else:
                    # Coverage info from the parent chunk text
                    coverage = None
                    for ev in section_evidences:
                        if ev.kind == "parent" and ev.section_id == best_section_id:
                            n = count_list_items(ev.text)
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
            # else: low confidence or empty section → fall through to normal-QA evidence

    # Normal QA: return hybrid top-k evidence + parent context enrichment
    # For each child chunk, also fetch its parent section text so the LLM
    # has hierarchical context (heading + surrounding content).
    evidences: List[Evidence] = []
    for cid, doc, meta in zip(got_ids, got_docs, got_metas):
        score = reranked_scores.get(cid, hybrid.scores.get(cid, 0.0))
        evidences.append(meta_to_evidence(meta, doc or "", cid, score))

    evidences = enrich_with_parent_context(index, evidences)
    evidences = filter_low_relevance(
        evidences,
        min_score_ratio=relevance_min_score_ratio,
        min_keep=relevance_min_keep,
    )

    return RetrievalResult(
        intent=intent,
        evidences=evidences,
        section_complete=False,
    )
