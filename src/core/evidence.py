"""
Evidence gathering and section filtering utilities for the RAG retrieval pipeline.

Extracted from retrieval.py for modularity.
Contains: Evidence, RetrievalResult, CoverageInfo, fetching subtree chunks,
and parent context enrichment.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Set

from .indexing import LocalIndex
from .query_classification import QueryIntent

logger = logging.getLogger(__name__)


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
    region_label: str = ""
    region_id: str = ""
    crop_type: str = "page"
    region_summary: str = ""
    proposal_source: str = "heuristic"
    proposal_confidence: float = 0.0
    bbox_left: int = 0
    bbox_top: int = 0
    bbox_right: int = 0
    bbox_bottom: int = 0
    bbox_left_norm: float = 0.0
    bbox_top_norm: float = 0.0
    bbox_right_norm: float = 1.0
    bbox_bottom_norm: float = 1.0


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


def meta_to_evidence(meta: dict, doc: str, chunk_id: str, score: float) -> Evidence:
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
        region_label=meta.get("region_label", ""),
        region_id=meta.get("region_id", ""),
        crop_type=meta.get("crop_type", "page"),
        region_summary=meta.get("region_summary", ""),
        proposal_source=meta.get("proposal_source", "heuristic"),
        proposal_confidence=float(meta.get("proposal_confidence", 0.0) or 0.0),
        bbox_left=int(meta.get("bbox_left", 0) or 0),
        bbox_top=int(meta.get("bbox_top", 0) or 0),
        bbox_right=int(meta.get("bbox_right", 0) or 0),
        bbox_bottom=int(meta.get("bbox_bottom", 0) or 0),
        bbox_left_norm=float(meta.get("bbox_left_norm", 0.0) or 0.0),
        bbox_top_norm=float(meta.get("bbox_top_norm", 0.0) or 0.0),
        bbox_right_norm=float(meta.get("bbox_right_norm", 1.0) or 1.0),
        bbox_bottom_norm=float(meta.get("bbox_bottom_norm", 1.0) or 1.0),
    )


def fetch_section_and_subtree(
    index: LocalIndex,
    doc_id: str,
    section_id: str,
    *,
    max_depth: int = 2,
    text_only: bool = False,
) -> List[Evidence]:
    """
    Fetch the section's own chunks (parent + children) AND all chunks whose
    parent_id equals this section_id (the subtree).  This ensures that asking
    for "4. Teslimatlar" also brings in 4.1, 4.2, etc.

    When *text_only* is True, visual and table-image chunks are excluded from
    the result.  This prevents visual chunks from contaminating text-based
    section retrieval (e.g. section_list intent).
    """
    col = index.store._get_collection()

    # Modalities / kinds to exclude when text_only is requested
    _VISUAL_MODALITIES = {"visual"}
    _VISUAL_KINDS = {"visual"}

    def _accept(meta: dict) -> bool:
        """Return False for chunks that should be excluded in text_only mode."""
        if not text_only:
            return True
        modality = (meta.get("modality", "text") or "text").strip().lower()
        kind = (meta.get("kind", "") or "").strip().lower()
        if modality in _VISUAL_MODALITIES or kind in _VISUAL_KINDS:
            return False
        return True

    # 1) Chunks belonging directly to this section
    own = col.get(
        where={"$and": [{"doc_id": doc_id}, {"section_id": section_id}]},
        include=["documents", "metadatas"],
    )

    # 2) Subtree fetch (depth-limited BFS on section hierarchy)
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

            # Discover immediate sub-sections: only parent chunks qualify.
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
                    logger.debug("Subtree BFS: skipping malformed metadata", exc_info=True)
                    continue

        frontier = next_frontier

    # Merge all into a single evidence list (deduplicated, filtered)
    seen: Set[str] = set()
    evidences: List[Evidence] = []

    for result_set in [own] + subtree_results:
        ids = result_set.get("ids", [])
        docs = result_set.get("documents", [])
        metas = result_set.get("metadatas", [])
        for cid, doc, meta in zip(ids, docs, metas):
            if cid in seen:
                continue
            seen.add(cid)
            if not _accept(meta or {}):
                continue
            evidences.append(meta_to_evidence(meta, doc or "", cid, score=1.0))

    return evidences


# ── Parent context enrichment for normal_qa ─────────────────────────────────

def enrich_with_parent_context(
    index: LocalIndex,
    evidences: List[Evidence],
) -> List[Evidence]:
    """
    For each child chunk in the evidence list, fetch its parent chunk from
    ChromaDB and prepend it.  This gives the LLM hierarchical context
    (section heading + overview) alongside the specific child content.

    Deduplicates: each parent is added at most once.
    """
    if not evidences:
        return evidences

    col = index.store._get_collection()
    seen_ids: Set[str] = {ev.chunk_id for ev in evidences}
    # Collect (doc_id, section_id) pairs for child chunks that need parents
    parent_needs: Set[tuple[str, str]] = set()
    parent_section_ids: Set[str] = {ev.section_id for ev in evidences if ev.kind == "parent"}

    for ev in evidences:
        if ev.kind == "child":
            sid = ev.section_id
            if sid and sid not in parent_section_ids:
                parent_needs.add((ev.chunk_id.rsplit(":", 1)[0] if ":" in ev.chunk_id else "", sid))
                # Retrieve using doc_id from evidence metadata
                parent_needs.add((ev.chunk_id, sid))

    # Fetch parent chunks for sections not already present
    enriched: List[Evidence] = list(evidences)
    fetched_sections: Set[str] = set(parent_section_ids)

    for ev in evidences:
        if ev.kind != "child":
            continue
        sid = ev.section_id
        if not sid or sid in fetched_sections:
            continue
        fetched_sections.add(sid)

        # Find the parent chunk: kind="parent" AND section_id=sid
        try:
            doc_id_val = ""
            for e2 in evidences:
                if e2.section_id == sid:
                    # Extract doc_id from chunk_id pattern: "doc_id:section:kind:idx"
                    parts = e2.chunk_id.split(":")
                    if len(parts) >= 2:
                        doc_id_val = parts[0]
                    break

            if not doc_id_val:
                continue

            res = col.get(
                where={"$and": [
                    {"doc_id": doc_id_val},
                    {"section_id": sid},
                    {"kind": "parent"},
                ]},
                include=["documents", "metadatas"],
            )
            p_ids = res.get("ids", [])
            p_docs = res.get("documents", [])
            p_metas = res.get("metadatas", [])
            for pid, pdoc, pmeta in zip(p_ids, p_docs, p_metas):
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    # Insert parent before its children with a slightly boosted score
                    # Sentinel score -1.0 marks parent enrichment chunks
                    # (never conflicts with real hybrid/reranked scores ≥ 0)
                    enriched.insert(0, meta_to_evidence(pmeta, pdoc or "", pid, score=-1.0))
        except Exception:
            logger.debug("Parent enrichment failed for section %s", sid, exc_info=True)
            continue

    return enriched


# ── Evidence relevance filtering ─────────────────────────────────────────────

