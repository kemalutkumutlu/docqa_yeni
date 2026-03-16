import re

with open("src/core/structure.py", "r", encoding="utf-8") as f:
    text = f.read()

# Add tiktoken import
import_str = """
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENC = None

def _count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text, disallowed_special=()))
    return max(1, len(text) // 4)
"""
text = text.replace("from .models import Chunk, IngestResult, PageText\n", "from .models import Chunk, IngestResult, PageText\n" + import_str)

# Replace _split_text_semantically
old_split = """def _split_text_semantically(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    \"\"\"
    Deterministic paragraph-aware splitter:
    - Split by blank lines into paragraphs
    - Accumulate paragraphs into ~max_chars windows
    - Paragraph-aligned overlap: carry the last N chars worth of whole
      paragraphs from the previous chunk (never cuts mid-paragraph)
    \"\"\"
    paras = [p.strip() for p in re.split(r"\\n\\s*\\n", text) if p.strip()]
    if not paras:
        return []

    # Build paragraph groups that fit within max_chars
    groups: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    for p in paras:
        add_len = len(p) + (2 if cur else 0)
        if cur and cur_len + add_len > max_chars:
            groups.append(cur)
            cur = []
            cur_len = 0
        cur.append(p)
        cur_len += add_len
    if cur:
        groups.append(cur)

    if overlap_chars <= 0 or len(groups) <= 1:
        return ["\\n\\n".join(g).strip() for g in groups]

    # Paragraph-aligned overlap: for each group after the first, prepend
    # whole paragraphs from the previous group that fit within overlap_chars.
    chunks: list[str] = ["\\n\\n".join(groups[0]).strip()]
    for i in range(1, len(groups)):
        prev = groups[i - 1]
        # Walk backwards through previous group's paragraphs
        overlap_paras: list[str] = []
        overlap_len = 0
        for p in reversed(prev):
            if overlap_len + len(p) + 2 > overlap_chars:
                break
            overlap_paras.insert(0, p)
            overlap_len += len(p) + 2
        merged_parts = overlap_paras + groups[i]
        chunks.append("\\n\\n".join(merged_parts).strip())

    return chunks"""

new_split = """def _split_text_semantically(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    \"\"\"
    Deterministic paragraph-aware token splitter:
    - Split by blank lines into paragraphs
    - Accumulate paragraphs into ~max_tokens windows
    - Paragraph-aligned overlap: carry the last N tokens worth of whole
      paragraphs from the previous chunk (never cuts mid-paragraph)
    \"\"\"
    paras = [p.strip() for p in re.split(r"\\n\\s*\\n", text) if p.strip()]
    if not paras:
        return []

    # Build paragraph groups that fit within max_tokens
    groups: list[list[str]] = []
    cur: list[str] = []
    cur_toks = 0
    for p in paras:
        p_toks = _count_tokens(p)
        add_toks = p_toks + (1 if cur else 0)  # rough token cost for \\n\\n
        # If a single paragraph is larger than max_tokens, it gets its own chunk anyway
        if cur and cur_toks + add_toks > max_tokens:
            groups.append(cur)
            cur = []
            cur_toks = 0
        cur.append(p)
        cur_toks += add_toks
    if cur:
        groups.append(cur)

    if overlap_tokens <= 0 or len(groups) <= 1:
        return ["\\n\\n".join(g).strip() for g in groups]

    # Paragraph-aligned overlap: for each group after the first, prepend
    # whole paragraphs from the previous group that fit within overlap_tokens.
    chunks: list[str] = ["\\n\\n".join(groups[0]).strip()]
    for i in range(1, len(groups)):
        prev = groups[i - 1]
        overlap_paras: list[str] = []
        overlap_cur_toks = 0
        for p in reversed(prev):
            p_toks = _count_tokens(p) + 1
            if overlap_cur_toks + p_toks > overlap_tokens:
                break
            overlap_paras.insert(0, p)
            overlap_cur_toks += p_toks
        merged_parts = overlap_paras + groups[i]
        chunks.append("\\n\\n".join(merged_parts).strip())

    return chunks"""

text = text.replace(old_split, new_split)

# Replace references in section_tree_to_chunks
text = text.replace("child_max_chars: int = 1000", "child_max_tokens: int = 250")
text = text.replace("child_overlap_chars: int = 150", "child_overlap_tokens: int = 40")
text = text.replace("max_chars=child_max_chars", "max_tokens=child_max_tokens")
text = text.replace("overlap_chars=child_overlap_chars", "overlap_tokens=child_overlap_tokens")

# Better heading detection in `_is_allcaps_heading`
old_heading = """def _is_allcaps_heading(s: str) -> bool:
        \"\"\"
        Conservative, document-agnostic fallback for heading detection.

        Only used when numbered heading detection yields too few headings.
        Intuition: many documents use ALLCAPS section labels (e.g., "INTRODUCTION"),
        which deterministic numbered regexes won't catch.
        \"\"\"
        t = s.strip()
        if not t:
            return False
        if any(ch.isdigit() for ch in t):
            return False
        if len(t) < 3 or len(t) > 40:
            return False
        if t.endswith((".", "!", "?", ":", ";")):
            return False
        # Token count: avoid treating long sentences as headings
        toks = [x for x in re.split(r"\\s+", t) if x]
        if len(toks) > 6:
            return False
        letters = [ch for ch in t if ch.isalpha()]
        if len(letters) < 3:
            return False
        upper = sum(1 for ch in letters if ch.isupper())
        # Require strong uppercase ratio, but not necessarily perfect (Turkish casing etc.)
        return (upper / max(1, len(letters))) >= 0.85"""

new_heading = """def _is_unkeyed_heading_candidate(s: str) -> bool:
        \"\"\"
        Broader, document-agnostic fallback for explicit non-numbered headings.
        
        Catches:
        - ALLCAPS headings ("INTRODUCTION")
        - Title Case short lines without punctuation
        - Very short, bold label-like lines
        \"\"\"
        t = s.strip()
        if not t:
            return False
        # Reject long lines
        if len(t) < 3 or len(t) > 60:
            return False
        # Reject lines ending in sentence-ending punctuation (colon is ok for "Summary:")
        if t.endswith((".", "!", "?", ";")):
            return False
            
        toks = [x for x in re.split(r"\\s+", t) if x]
        if len(toks) > 8:
            return False
            
        letters = [ch for ch in t if ch.isalpha()]
        if len(letters) < 3:
            return False
            
        upper = sum(1 for ch in letters if ch.isupper())
        upper_ratio = upper / max(1, len(letters))
        
        # Condition 1: Strong ALLCAPS
        if upper_ratio >= 0.85 and sum(1 for ch in t if ch.isdigit()) <= 3:
            return True
            
        # Condition 2: Title Case (First letter of most words is capitalized)
        if len(toks) <= 5:
            title_case_words = sum(1 for w in toks if w and w[0].isupper())
            if title_case_words / len(toks) >= 0.75:
                return True
                
        return False"""

text = text.replace(old_heading, new_heading)
text = text.replace("_is_allcaps_heading", "_is_unkeyed_heading_candidate")
text = text.replace("enable_allcaps", "enable_unkeyed_fallback")

with open("src/core/structure.py", "w", encoding="utf-8") as f:
    f.write(text)

print("done")
