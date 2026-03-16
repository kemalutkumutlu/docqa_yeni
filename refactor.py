import re
import os

src_dir = "src/core"
retrieval_path = os.path.join(src_dir, "retrieval.py")
evidence_path = os.path.join(src_dir, "evidence.py")

with open(retrieval_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

def get_block(start_marker, end_marker=None, include_end=False):
    start_idx = -1
    for i, line in enumerate(lines):
        if start_marker in line:
            start_idx = i
            break
    if start_idx == -1:
        return []
    
    if end_marker is None:
        return lines[start_idx:]
        
    end_idx = -1
    for i in range(start_idx + 1, len(lines)):
        if end_marker in lines[i]:
            end_idx = i
            break
            
    if end_idx == -1:
        return lines[start_idx:]
        
    if include_end:
        return lines[start_idx:end_idx+1]
    else:
        return lines[start_idx:end_idx]

# Extract evidence.py content
evidence_imports = """\"\"\"
Evidence gathering and section filtering utilities for the RAG retrieval pipeline.

Extracted from retrieval.py for modularity.
Contains: Evidence, RetrievalResult, CoverageInfo, fetching subtree chunks,
and parent context enrichment.
\"\"\"
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Set

from .indexing import LocalIndex
from .query_classification import QueryIntent

logger = logging.getLogger(__name__)


"""

evidence_classes = get_block("@dataclass(frozen=True)", "def _meta_to_evidence")
meta_to_evidence = get_block("def _meta_to_evidence", "def _page_number_from_meta")
fetch_subtree = get_block("def _fetch_section_and_subtree(", "def _enrich_with_parent_context(")
enrich_context = get_block("def _enrich_with_parent_context(", "def _filter_low_relevance(")

with open(evidence_path, "w", encoding="utf-8") as f:
    f.write(evidence_imports)
    f.writelines(evidence_classes)
    f.writelines(meta_to_evidence)
    f.writelines(fetch_subtree)
    f.writelines(enrich_context)

print(f"Created {evidence_path}")

# Now rewrite retrieval.py
retrieval_new = []

# Keep imports up to logger
import_end = -1
for i, line in enumerate(lines):
    retrieval_new.append(line)
    if "logger = logging.getLogger(__name__)" in line:
        import_end = i
        break

retrieval_new.append("\nfrom .query_classification import (\n")
retrieval_new.append("    QueryIntent,\n")
retrieval_new.append("    classify_query,\n")
retrieval_new.append("    _query_prefers_visual_region,\n")
retrieval_new.append("    _topic_heading_relevant,\n")
retrieval_new.append("    _expand_query_morphological,\n")
retrieval_new.append(")\n")
retrieval_new.append("from .evidence import (\n")
retrieval_new.append("    Evidence,\n")
retrieval_new.append("    RetrievalResult,\n")
retrieval_new.append("    CoverageInfo,\n")
retrieval_new.append("    _meta_to_evidence,\n")
retrieval_new.append("    _fetch_section_and_subtree,\n")
retrieval_new.append("    _enrich_with_parent_context,\n")
retrieval_new.append(")\n")
retrieval_new.append("from .ranking import (\n")
retrieval_new.append("    _region_aware_scores,\n")
retrieval_new.append("    _rerank_by_embedding,\n")
retrieval_new.append("    _filter_low_relevance,\n")
retrieval_new.append("    _pick_best_section,\n")
retrieval_new.append("    _count_list_items,\n")
retrieval_new.append(")\n\n")

# Find where `def _multi_query_expand` starts
mqe_start = -1
for i, line in enumerate(lines):
    if "def _multi_query_expand(" in line:
        mqe_start = i
        break

# Find where `def _multi_query_expand` ends, which is just before `def retrieve(`
retrieve_start = -1
for i in range(mqe_start, len(lines)):
    if "def retrieve(" in line:
        retrieve_start = i
        break

# Find where `# ── Query expansion ─────────────────────────────────────────────────────────` starts
qe_comment_idx = -1
for i in range(import_end + 1, len(lines)):
    if "# ── Query expansion ─────────────────────────────────────────────────────────" in lines[i]:
        qe_comment_idx = i
        break

# Append from `def _multi_query_expand` to end
retrieval_new.append(lines[qe_comment_idx])
retrieval_new.append("\n")
retrieval_new.extend(lines[mqe_start:])

with open(retrieval_path, "w", encoding="utf-8") as f:
    f.writelines(retrieval_new)

print(f"Rewrote {retrieval_path}")
