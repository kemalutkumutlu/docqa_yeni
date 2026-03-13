from __future__ import annotations

import re


CONTENT_NORMALIZER_VERSION = "canon-v1"

_RE_LIST_START = re.compile(r"^(\d+[\.\)]|[a-zA-Z][\.\)]|[-\u2022*])\s+")
_RE_NUMBERED_HEADING = re.compile(r"^([A-Z]\.)?\d+(?:\.\d+)*([.\)]|\s+-)\s+")
_RE_EXPLICIT_PAIR = re.compile(r"^(?P<label>[^:\n]{2,80}):\s+(?P<value>.+)$")


def normalize_extracted_text(text: str, *, source: str = "") -> str:
    del source  # Reserved for future source-specific heuristics.

    raw = (text or "").strip()
    if not raw:
        return ""

    raw = _merge_hyphenated_line_breaks(raw)
    blocks = [blk for blk in re.split(r"\n\s*\n", raw) if blk.strip()]
    rendered: list[str] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        rendered_block = _normalize_block(lines)
        if rendered_block:
            rendered.append(rendered_block)
    return "\n\n".join(rendered).strip()


def _normalize_block(lines: list[str]) -> str:
    if _looks_like_markdown_table(lines):
        return "\n".join(lines)

    heading, body = _split_leading_heading(lines)
    normalized_body = _normalize_body_block(body)
    out: list[str] = []
    if heading:
        out.append(heading)
    if normalized_body:
        out.append(normalized_body)
    return "\n".join(out).strip()


def _normalize_body_block(lines: list[str]) -> str:
    if not lines:
        return ""

    list_items = _canonicalize_wrapped_list(lines)
    if len(list_items) >= 2:
        return "\n".join(list_items)

    indexed_rows = _canonicalize_indexed_rows(lines)
    if len(indexed_rows) >= 2:
        return "\n".join(indexed_rows)

    label_rows = _canonicalize_label_value_rows(lines)
    if len(label_rows) >= 2:
        return "\n".join(label_rows)

    explicit_rows = _canonicalize_existing_pairs(lines)
    if len(explicit_rows) >= 2:
        return "\n".join(explicit_rows)

    if _should_preserve_dense_short_lines(lines):
        return "\n".join(lines)

    return "\n".join(_render_generic_lines(lines))


def _merge_hyphenated_line_breaks(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return ""

    out: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i].rstrip()
        if not cur:
            out.append("")
            i += 1
            continue
        if i + 1 >= len(lines):
            out.append(cur)
            i += 1
            continue

        nxt = lines[i + 1].lstrip()
        if (
            cur.endswith("-")
            and nxt
            and not _is_structural_line(cur[:-1].strip())
            and not _is_structural_line(nxt)
            and nxt[:1].isalnum()
        ):
            out.append(cur[:-1] + nxt)
            i += 2
            continue

        out.append(cur)
        i += 1
    return "\n".join(out)


def _split_leading_heading(lines: list[str]) -> tuple[str, list[str]]:
    if len(lines) >= 2 and _is_heading_like(lines[0]):
        if _is_list_start(lines[0]) and not _is_structural_line(lines[1]):
            return "", lines
        return lines[0], lines[1:]
    return "", lines


def _canonicalize_wrapped_list(lines: list[str]) -> list[str]:
    items: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not _is_list_start(line):
            return []

        parts = [line]
        j = i + 1
        while j < len(lines) and not _is_structural_line(lines[j]):
            parts.append(lines[j])
            j += 1
        items.append(_join_fragments(parts))
        i = j
    return items if len(items) >= 2 else []


def _canonicalize_indexed_rows(lines: list[str]) -> list[str]:
    items: list[str] = []
    i = 0
    while i < len(lines):
        idx = lines[i]
        if not re.match(r"^\d{1,3}$", idx):
            return []

        if i + 1 >= len(lines):
            return []
        label = lines[i + 1]
        if not _looks_like_short_label(label):
            return []

        j = i + 2
        desc_parts: list[str] = []
        while j < len(lines) and not re.match(r"^\d{1,3}$", lines[j]):
            desc_parts.append(lines[j])
            j += 1

        desc = _join_fragments(desc_parts)
        if desc:
            items.append(f"{idx}. {label}: {desc}")
        else:
            items.append(f"{idx}. {label}")
        i = j
    return items if len(items) >= 2 else []


def _canonicalize_label_value_rows(lines: list[str]) -> list[str]:
    work = list(lines)
    if (
        len(work) >= 4
        and _looks_like_short_label(work[0])
        and _looks_like_short_label(work[1])
        and not _looks_like_value_line(work[1])
    ):
        if _looks_like_short_label(work[2]) and _looks_like_value_line(work[3]):
            work = work[2:]

    items: list[str] = []
    i = 0
    while i < len(work):
        label = work[i]
        if not _looks_like_short_label(label):
            return []

        j = i + 1
        desc_parts: list[str] = []
        while j < len(work):
            cur = work[j]
            if _is_structural_line(cur):
                return []
            if _looks_like_short_label(cur) and desc_parts:
                break
            desc_parts.append(cur)
            j += 1

        desc = _join_fragments(desc_parts)
        if not desc or (len(desc_parts) == 1 and _looks_like_short_label(desc_parts[0]) and not _looks_like_value_line(desc_parts[0])):
            return []
        items.append(f"{label}: {desc}")
        i = j
    return items if len(items) >= 2 else []


def _canonicalize_existing_pairs(lines: list[str]) -> list[str]:
    items: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        match = _RE_EXPLICIT_PAIR.match(line)
        if not match:
            return []

        parts = [match.group("value").strip()]
        j = i + 1
        while j < len(lines):
            cur = lines[j]
            if _is_structural_line(cur) or _RE_EXPLICIT_PAIR.match(cur):
                break
            parts.append(cur)
            j += 1

        value = _join_fragments(parts)
        items.append(f"{match.group('label').strip()}: {value}")
        i = j
    return items if len(items) >= 2 else []


def _render_generic_lines(lines: list[str]) -> list[str]:
    rendered: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _is_heading_like(line):
            rendered.append(line)
            i += 1
            continue

        pair_match = _RE_EXPLICIT_PAIR.match(line)
        if pair_match:
            parts = [pair_match.group("value").strip()]
            j = i + 1
            while j < len(lines) and not _is_structural_line(lines[j]) and not _RE_EXPLICIT_PAIR.match(lines[j]):
                parts.append(lines[j])
                j += 1
            rendered.append(f"{pair_match.group('label').strip()}: {_join_fragments(parts)}")
            i = j
            continue

        if _is_list_start(line):
            parts = [line]
            j = i + 1
            while j < len(lines) and not _is_structural_line(lines[j]):
                parts.append(lines[j])
                j += 1
            rendered.append(_join_fragments(parts))
            i = j
            continue

        parts = [line]
        j = i + 1
        while j < len(lines) and not _is_structural_line(lines[j]) and not _RE_EXPLICIT_PAIR.match(lines[j]):
            parts.append(lines[j])
            j += 1
        rendered.append(_join_fragments(parts))
        i = j
    return rendered


def _join_fragments(lines: list[str]) -> str:
    parts: list[str] = []
    for line in lines:
        cur = line.strip()
        if not cur:
            continue
        if not parts:
            parts.append(cur)
            continue
        prev = parts[-1]
        if _needs_space(prev, cur):
            parts[-1] = f"{prev} {cur}"
        else:
            parts[-1] = f"{prev}{cur}"
    return "".join(parts).strip()


def _needs_space(prev: str, cur: str) -> bool:
    if not prev or not cur:
        return False
    if prev.endswith(("/", "(")):
        return False
    if cur[:1] in ",.;:)]}%":
        return False
    return True


def _looks_like_markdown_table(lines: list[str]) -> bool:
    if len(lines) < 2:
        return False
    pipe_lines = sum(1 for line in lines if line.count("|") >= 2)
    return pipe_lines >= 2


def _should_preserve_dense_short_lines(lines: list[str]) -> bool:
    if len(lines) < 4:
        return False
    shortish = sum(1 for line in lines if _token_count(line) <= 4 and len(line) <= 32)
    punct = sum(1 for line in lines if any(ch in line for ch in ".;:"))
    return shortish >= max(3, int(len(lines) * 0.7)) and punct <= 1


def _is_structural_line(line: str) -> bool:
    return _is_heading_like(line) or _is_list_start(line)


def _is_list_start(line: str) -> bool:
    return bool(_RE_LIST_START.match(line.strip()))


def _is_heading_like(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    if _RE_NUMBERED_HEADING.match(text):
        return True
    if any(ch.isdigit() for ch in text):
        return False
    if len(text) > 60 or text.endswith((".", "!", "?", ":", ";")):
        return False
    if _token_count(text) > 8:
        return False
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 3:
        return False
    upper = sum(1 for ch in letters if ch.isupper())
    return (upper / len(letters)) >= 0.85


def _looks_like_short_label(line: str) -> bool:
    text = line.strip()
    if not (2 <= len(text) <= 60):
        return False
    if text.endswith((".", "!", "?", ":", ";")):
        return False
    if re.match(r"^\d+$", text):
        return False
    return _token_count(text) <= 7


def _looks_like_value_line(line: str) -> bool:
    text = line.strip()
    if len(text) >= 80:
        return True
    if _token_count(text) >= 8:
        return True
    if any(ch.isdigit() for ch in text) and _token_count(text) >= 2:
        return True
    if any(ch in text for ch in ".;:") and _token_count(text) >= 5:
        return True
    return False


def _token_count(text: str) -> int:
    return len([tok for tok in re.split(r"\s+", text.strip()) if tok])
