import re

def fix(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    replacements = [
        ("_query_prefers_visual_region", "query_prefers_visual_region"),
        ("_topic_heading_relevant", "topic_heading_relevant"),
        ("_expand_query_morphological", "expand_query_morphological"),
        ("_meta_to_evidence", "meta_to_evidence"),
        ("_fetch_section_and_subtree", "fetch_section_and_subtree"),
        ("_enrich_with_parent_context", "enrich_with_parent_context"),
        ("_region_aware_scores", "region_aware_scores"),
        ("_rerank_by_embedding", "rerank_by_embedding"),
        ("_filter_low_relevance", "filter_low_relevance"),
        ("_pick_best_section", "pick_best_section"),
        ("_count_list_items", "count_list_items"),
        ("_tokenize_simple", "tokenize_simple"),
        ("_heading_query_overlap", "heading_query_overlap"),
    ]
    for old, new in replacements:
        # Avoid double replacing `def query_prefers` to `def query_prefers` etc.
        # It's simple string replace, so if I replace `_foo` with `foo`, we might get `__foo`? No.
        text = text.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

fix("src/core/evidence.py")
fix("src/core/ranking.py")
