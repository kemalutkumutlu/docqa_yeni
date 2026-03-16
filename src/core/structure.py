from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .models import Chunk, IngestResult, PageText


@dataclass(frozen=True)
class Line:
    page: int  # 1-based
    text: str


@dataclass(frozen=True)
class Heading:
    key: Optional[str]  # e.g. "2", "4.1", "A.4.1" (None if unkeyed)
    title: str
    level: int  # 1..N


@dataclass
class SectionNode:
    section_id: str
    title: str
    level: int
    heading_key: Optional[str]
    heading_path: str
    page_start: int
    page_end: int
    parent_id: Optional[str]
    body_lines: list[Line] = field(default_factory=list)
    children: list["SectionNode"] = field(default_factory=list)

    def full_text(self) -> str:
        body = "\n".join([ln.text for ln in self.body_lines]).strip()
        if body:
            return f"{self.title}\n{body}".strip()
        return self.title.strip()


_RE_NUM_DOT = re.compile(r"^(?P<num>\d+(?:\.\d+)*)\.\s+(?P<title>.+?)\s*$")
_RE_NUM_DASH = re.compile(r"^(?P<num>\d+(?:\.\d+)*)\s*[-–—]\s*(?P<title>.+?)\s*$")
_RE_ALPHA_NUM = re.compile(r"^(?P<alpha>[A-Z])\.(?P<num>\d+(?:\.\d+)*)\s+(?P<title>.+?)\s*$")


def _heading_level_from_key(key: str) -> int:
    # "2" -> 1, "4.1" -> 2, "A.4.1" -> 3
    if re.match(r"^[A-Z]\.", key):
        rest = key.split(".", 1)[1]
        segs = [s for s in rest.split(".") if s]
        return 1 + max(1, len(segs))
    segs = [s for s in key.split(".") if s]
    return max(1, len(segs))


def detect_heading(line: str) -> Optional[Heading]:
    """
    Best-effort heading detection (document-agnostic).

    We primarily rely on numbered headings:
      - "2. Title"
      - "4.1. Title"
      - "4.1 - Title"
      - "A.4.1 Title"

    Non-numbered headings are intentionally NOT detected here yet, to avoid
    false positives (e.g., repeating headers/footers).
    """
    s = line.strip()
    if not s:
        return None

    m = _RE_ALPHA_NUM.match(s)
    if m:
        key = f"{m.group('alpha')}.{m.group('num')}"
        level = _heading_level_from_key(key)
        title = f"{key} {m.group('title').strip()}"
        return Heading(key=key, title=title, level=level)

    m = _RE_NUM_DOT.match(s)
    if m:
        key = m.group("num")
        level = _heading_level_from_key(key)
        title = f"{key}. {m.group('title').strip()}"
        return Heading(key=key, title=title, level=level)

    m = _RE_NUM_DASH.match(s)
    if m:
        key = m.group("num")
        title_raw = m.group("title").strip()
        # Doc-agnostic guardrails against date/range false-positives.
        # - Reject headings where the "title" starts with a digit (often date ranges like "03-04-1987")
        if title_raw[:1].isdigit():
            return None
        level = _heading_level_from_key(key)
        title = f"{key} - {title_raw}"
        return Heading(key=key, title=title, level=level)

    return None


def _iter_page_lines(pages: Iterable[PageText]) -> list[Line]:
    lines: list[Line] = []
    for p in pages:
        for raw in p.text.splitlines():
            t = raw.strip()
            # keep blanks as separators
            lines.append(Line(page=p.page_number, text=t))
    return lines


# Patterns that are always boilerplate regardless of frequency
_BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    # Standalone page numbers: "1", "- 2 -", "Page 3", "Sayfa 5"
    re.compile(r"^[-–—\s]*\d{1,4}[-–—\s]*$"),
    re.compile(r"^(page|sayfa|s\.)\s*\d{1,4}\s*$", re.IGNORECASE),
    # "X / Y" page indicators: "3 / 12", "3/12"
    re.compile(r"^\d{1,4}\s*/\s*\d{1,4}$"),
    # Common footer patterns
    re.compile(r"^(confidential|gizli|taslak|draft)\s*$", re.IGNORECASE),
]


def _detect_boilerplate(lines: list[Line], pages_count: int) -> set[str]:
    """
    Remove repeating headers/footers (document-agnostic heuristic).

    Strategy:
    1. Pattern-based: standalone page numbers, "Page X", "Sayfa X" etc.
    2. Frequency-based: lines that repeat on >50% of pages in header/footer
       positions are treated as boilerplate.
    """
    boilerplate: set[str] = set()

    if pages_count <= 1:
        return boilerplate

    # 1) Pattern-based detection (always boilerplate)
    for ln in lines:
        if ln.text and any(pat.match(ln.text.strip()) for pat in _BOILERPLATE_PATTERNS):
            boilerplate.add(ln.text)

    # 2) Frequency-based detection (repeating headers/footers)
    per_page: dict[int, list[str]] = {}
    for ln in lines:
        if ln.text:
            per_page.setdefault(ln.page, []).append(ln.text)

    candidates: list[str] = []
    for page, ls in per_page.items():
        head = ls[:3]
        tail = ls[-3:] if len(ls) > 3 else []
        candidates.extend(head + tail)

    freq: dict[str, int] = {}
    for c in candidates:
        if len(c) <= 90:
            freq[c] = freq.get(c, 0) + 1

    # "many pages": >50% of pages (rounded down) or at least 2
    threshold = max(2, pages_count // 2 + 1)
    boilerplate.update(s for s, n in freq.items() if n >= threshold)

    return boilerplate


def build_section_tree(ingest: IngestResult) -> SectionNode:
    """
    Build a section tree from extracted page text using a deterministic stack.

    - Root node ("root") spans the whole document.
    - When a numbered heading is detected, create a new node at its level.
    - Body lines are appended to the current node on stack top.
    """
    pages = ingest.pages
    lines = _iter_page_lines(pages)
    boilerplate = _detect_boilerplate(lines, pages_count=len(pages))

    def _is_allcaps_heading(s: str) -> bool:
        """
        Conservative, document-agnostic fallback for heading detection.

        Only used when numbered heading detection yields too few headings.
        Intuition: many documents use ALLCAPS section labels (e.g., "INTRODUCTION"),
        which deterministic numbered regexes won't catch.
        """
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
        toks = [x for x in re.split(r"\s+", t) if x]
        if len(toks) > 6:
            return False
        letters = [ch for ch in t if ch.isalpha()]
        if len(letters) < 3:
            return False
        upper = sum(1 for ch in letters if ch.isupper())
        # Require strong uppercase ratio, but not necessarily perfect (Turkish casing etc.)
        return (upper / max(1, len(letters))) >= 0.85

    # Decide whether to enable ALLCAPS heading fallback.
    numbered_headings = 0
    allcaps_candidates = 0
    for ln in lines:
        if ln.text in boilerplate:
            continue
        if not ln.text:
            continue
        if detect_heading(ln.text) is not None:
            numbered_headings += 1
        if _is_allcaps_heading(ln.text):
            allcaps_candidates += 1
    enable_allcaps = numbered_headings < 2 and allcaps_candidates >= 2

    root = SectionNode(
        section_id="root",
        title=f"{ingest.file_name}",
        level=0,
        heading_key=None,
        heading_path=f"{ingest.file_name}",
        page_start=1 if pages else 1,
        page_end=max([p.page_number for p in pages], default=1),
        parent_id=None,
    )

    stack: list[SectionNode] = [root]
    auto_id = 0
    used_section_ids: set[str] = {"root"}
    dup_counts: dict[str, int] = {}

    def new_unkeyed_id() -> str:
        nonlocal auto_id
        auto_id += 1
        return f"h{auto_id:04d}"

    def unique_section_id(base: str) -> str:
        """
        Ensure section_id uniqueness within a document.

        Some PDFs contain repeated numbered headings (or false-positive headings like "2020. ...")
        on multiple pages. Using the raw heading key as section_id would collide and produce
        duplicate chunk_ids (Chroma upsert requires unique IDs per batch).
        """
        if base not in used_section_ids:
            used_section_ids.add(base)
            return base
        n = dup_counts.get(base, 2)
        while True:
            cand = f"{base}__{n:02d}"
            if cand not in used_section_ids:
                dup_counts[base] = n + 1
                used_section_ids.add(cand)
                return cand
            n += 1

    for ln in lines:
        if ln.text in boilerplate:
            continue

        h = detect_heading(ln.text)
        if h is None and enable_allcaps and _is_allcaps_heading(ln.text):
            h = Heading(key=None, title=ln.text.strip(), level=1)
        if h is not None:
            # Adjust stack for heading level
            while len(stack) > 1 and stack[-1].level >= h.level:
                stack.pop()

            parent = stack[-1]
            base_id = h.key or new_unkeyed_id()
            section_id = unique_section_id(base_id)
            heading_path = f"{parent.heading_path} / {h.title}".strip()
            node = SectionNode(
                section_id=section_id,
                title=h.title,
                level=h.level,
                heading_key=h.key,
                heading_path=heading_path,
                page_start=ln.page,
                page_end=ln.page,
                parent_id=parent.section_id,
            )
            parent.children.append(node)
            stack.append(node)
            continue

        # Body line
        cur = stack[-1]
        cur.body_lines.append(ln)
        if ln.page < cur.page_start:
            cur.page_start = ln.page
        if ln.page > cur.page_end:
            cur.page_end = ln.page

    # Expand page_end upwards for parents
    def fix_ranges(node: SectionNode) -> tuple[int, int]:
        start, end = node.page_start, node.page_end
        for ch in node.children:
            cstart, cend = fix_ranges(ch)
            start = min(start, cstart)
            end = max(end, cend)
        node.page_start, node.page_end = start, end
        return start, end

    fix_ranges(root)
    return root


def flatten_sections(root: SectionNode) -> list[SectionNode]:
    out: list[SectionNode] = []

    def walk(n: SectionNode) -> None:
        out.append(n)
        for c in n.children:
            walk(c)

    walk(root)
    return out


def _split_text_semantically(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """
    Deterministic paragraph-aware splitter:
    - Split by blank lines into paragraphs
    - Accumulate paragraphs into ~max_chars windows
    - Paragraph-aligned overlap: carry the last N chars worth of whole
      paragraphs from the previous chunk (never cuts mid-paragraph)
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
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
        return ["\n\n".join(g).strip() for g in groups]

    # Paragraph-aligned overlap: for each group after the first, prepend
    # whole paragraphs from the previous group that fit within overlap_chars.
    chunks: list[str] = ["\n\n".join(groups[0]).strip()]
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
        chunks.append("\n\n".join(merged_parts).strip())

    return chunks


def _heading_prefix(heading_path: str, file_name: str) -> str:
    """
    Extract a short heading prefix from the heading_path to prepend to
    child chunk text.  This gives the embedding model section context.

    "Case_Study.pdf / 2. Fonksiyonel Gereksinimler / 2.1 Login"
    → "[2. Fonksiyonel Gereksinimler / 2.1 Login]\n"
    """
    path = heading_path
    # Strip the file name prefix
    if " / " in path:
        path = path.split(" / ", 1)[1]
    path = path.strip()
    if not path or path == file_name.strip():
        return ""
    return f"[{path}]\n"


def section_tree_to_chunks(
    ingest: IngestResult,
    root: SectionNode,
    child_max_chars: int = 1000,
    child_overlap_chars: int = 150,
) -> list[Chunk]:
    """
    Create parent/child chunks per section node.

    - Parent chunk: full section text (title + full body)
    - Child chunks: split of full section text for retrieval granularity

    Default child_max_chars=1000 (~250 tokens) is optimized for embedding
    models that perform best at 128-256 token inputs.
    """
    chunks: list[Chunk] = []

    def add_section(node: SectionNode) -> None:
        # Include root preface content if present.
        # This matters for documents that contain important metadata BEFORE the first numbered heading
        # (e.g., delivery time, estimated work hours on page 1).
        if node.section_id == "root":
            root_text = node.full_text().strip()
            # If root has no body, full_text() is just the filename; skip indexing it.
            if root_text != ingest.file_name.strip():
                parent_chunk_id = f"{ingest.doc_id}:{node.section_id}:parent"
                chunks.append(
                    Chunk(
                        chunk_id=parent_chunk_id,
                        doc_id=ingest.doc_id,
                        file_name=ingest.file_name,
                        section_id=node.section_id,
                        parent_id=node.parent_id,
                        heading_path=node.heading_path,
                        page_start=node.page_start,
                        page_end=node.page_end,
                        text=root_text,
                        kind="parent",
                    )
                )

                child_texts = _split_text_semantically(
                    root_text,
                    max_chars=child_max_chars,
                    overlap_chars=child_overlap_chars,
                )
                if not child_texts:
                    child_texts = [root_text]

                _hpfx = _heading_prefix(node.heading_path, ingest.file_name)
                for idx, ct in enumerate(child_texts):
                    chunks.append(
                        Chunk(
                            chunk_id=f"{ingest.doc_id}:{node.section_id}:child:{idx:04d}",
                            doc_id=ingest.doc_id,
                            file_name=ingest.file_name,
                            section_id=node.section_id,
                            parent_id=node.section_id,
                            heading_path=node.heading_path,
                            page_start=node.page_start,
                            page_end=node.page_end,
                            text=_hpfx + ct,
                            kind="child",
                        )
                    )

            for ch in node.children:
                add_section(ch)
            return

        parent_text = node.full_text()
        parent_chunk_id = f"{ingest.doc_id}:{node.section_id}:parent"
        chunks.append(
            Chunk(
                chunk_id=parent_chunk_id,
                doc_id=ingest.doc_id,
                file_name=ingest.file_name,
                section_id=node.section_id,
                parent_id=node.parent_id,
                heading_path=node.heading_path,
                page_start=node.page_start,
                page_end=node.page_end,
                text=parent_text,
                kind="parent",
            )
        )

        child_texts = _split_text_semantically(
            parent_text,
            max_chars=child_max_chars,
            overlap_chars=child_overlap_chars,
        )
        if not child_texts:
            child_texts = [parent_text]

        _hpfx = _heading_prefix(node.heading_path, ingest.file_name)
        for idx, ct in enumerate(child_texts):
            chunks.append(
                Chunk(
                    chunk_id=f"{ingest.doc_id}:{node.section_id}:child:{idx:04d}",
                    doc_id=ingest.doc_id,
                    file_name=ingest.file_name,
                    section_id=node.section_id,
                    parent_id=node.section_id,  # children point to their parent section
                    heading_path=node.heading_path,
                    page_start=node.page_start,
                    page_end=node.page_end,
                    text=_hpfx + ct,
                    kind="child",
                )
            )

        for ch in node.children:
            add_section(ch)

    add_section(root)
    return chunks

