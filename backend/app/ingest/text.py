"""Pure text processing: Gutenberg boilerplate removal, chapter detection, passage chunking."""

import re
from dataclasses import dataclass

_START_RE = re.compile(r"^\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*$", re.I | re.M)
_END_RE = re.compile(r"^\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*$", re.I | re.M)

_ROMAN = r"[IVXLC]+"
_GERMAN_ORDINAL = (
    r"(Erste|Zweite|Dritte|Vierte|Fünfte|Sechste|Siebente|Siebte|Achte|Neunte|Zehnte|Elfte|Zwölfte"
    r"|Dreizehnte|Vierzehnte|Fünfzehnte|Sechzehnte|Siebzehnte|Achtzehnte|Neunzehnte|Zwanzigste)[rsn]?"
)
_UNIT_EN = r"(CHAPTER|Chapter|BOOK|Book|PART|Part|ACT|Act|STAVE|Stave|LETTER|Letter|SCENE|Scene)"
_UNIT_DE = r"(KAPITEL|Kapitel|TEIL|Teil|BUCH|Buch|AKT|Akt|AUFZUG|Aufzug|AUFTRITT|Auftritt|BRIEF|Brief|ABSCHNITT|Abschnitt)"
_NUM = rf"({_ROMAN}|\d+|[A-Z][a-z]+|[A-Z]+)"
_ORD_EN = r"(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH)"

_HEADING_PATTERNS = [
    re.compile(rf"^{_UNIT_EN}\s+{_NUM}\b.{{0,80}}$"),  # CHAPTER XII. / Chapter 3 / BOOK ONE
    re.compile(rf"^{_UNIT_DE}\s+{_NUM}\b.{{0,80}}$"),  # Kapitel 3
    re.compile(rf"^({_ROMAN}|\d+)\.?\s+{_UNIT_DE}\b.{{0,80}}$"),  # 3. Kapitel / III. Teil
    re.compile(rf"^{_GERMAN_ORDINAL}\s+{_UNIT_DE}\b.{{0,80}}$", re.I),  # Erstes Kapitel
    re.compile(rf"^(THE\s+)?{_ORD_EN}\s+(BOOK|PART|CHAPTER)\b.{{0,60}}$", re.I),  # THE FIRST BOOK
    re.compile(rf"^(BOOK|PART|CHAPTER)\s+THE\s+{_ORD_EN}\b.{{0,60}}$", re.I),  # Book the First
    re.compile(rf"^{_ROMAN}\.?$"),  # XII
    re.compile(rf"^{_ROMAN}\.\s+[A-ZÄÖÜ][A-ZÄÖÜ ,.'’\-]{{2,70}}$"),  # I. A SCANDAL IN BOHEMIA
    re.compile(r"^\d{1,3}\.?$"),  # 12
]

MIN_CHAPTER_CHARS = 1500
FALLBACK_SECTION_CHARS = 15000


@dataclass
class Chapter:
    idx: int
    title: str
    text: str


def strip_gutenberg(raw: str) -> str:
    """Remove the Project Gutenberg header and license footer."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    start = _START_RE.search(text)
    if start:
        text = text[start.end():]
    end = _END_RE.search(text)
    if end:
        text = text[: end.start()]
    return text.strip()


def is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 90:
        return False
    return any(p.match(line) for p in _HEADING_PATTERNS)


def _paragraph_blocks(text: str) -> list[str]:
    return [b for b in re.split(r"\n\s*\n", text) if b.strip()]


_TOC_MARKERS = {"contents", "table of contents", "inhalt", "inhaltsverzeichnis", "inhaltsangabe"}


def _norm_title(line: str) -> str:
    return re.sub(r"[^\w ]", "", re.sub(r"\s+", " ", line)).strip().lower()


def _single_line(block: str) -> str | None:
    text = block.strip()
    return text if text and "\n" not in text else None


def _pattern_detector(blocks: list[str]):
    def is_head(block: str) -> str | None:
        lines = [ln.strip() for ln in block.strip().split("\n")]
        return " ".join(lines) if is_heading(lines[0]) and len(lines) <= 3 else None
    return is_head


def _toc_detector(blocks: list[str]):
    """Headings are the entries of the table of contents, found again as standalone lines after it."""
    start = next((i for i, b in enumerate(blocks) if _norm_title(b) in _TOC_MARKERS), None)
    if start is None:
        return None, 0
    entries: set[str] = set()
    end = start + 1
    while end < len(blocks):
        lines = [ln.strip() for ln in blocks[end].strip().split("\n") if ln.strip()]
        if any(len(ln) > 80 for ln in lines) or len(entries) > 300:
            break
        entries.update(_norm_title(ln) for ln in lines)
        end += 1
    entries.discard("")
    if len(entries) < 3:
        return None, 0

    def is_head(block: str) -> str | None:
        line = _single_line(block)
        return line if line and _norm_title(line) in entries else None
    return is_head, end


def _caps_detector(blocks: list[str]):
    def is_head(block: str) -> str | None:
        line = _single_line(block)
        if not line or len(line) > 60 or len(line) < 4:
            return None
        letters = [c for c in line if c.isalpha()]
        return line if len(letters) >= 4 and all(c.isupper() for c in letters) else None
    return is_head


TOC_GAP_CHARS = 200


def _drop_toc_entries(blocks: list[str], heads: dict[int, str]) -> dict[int, str]:
    """Remove headings that belong to a table of contents.

    A table of contents is a run of at least three headings with (almost) no text between them. The heading
    right after such a run is also a TOC entry when the same title appears again later in the book.
    """
    order = sorted(heads)
    gaps = []
    for k, i in enumerate(order):
        nxt = order[k + 1] if k + 1 < len(order) else len(blocks)
        gaps.append(sum(len(b) for b in blocks[i + 1:nxt]))
    toc: set[int] = set()
    run: list[int] = []
    for k, i in enumerate(order):
        if gaps[k] < TOC_GAP_CHARS:
            run.append(i)
            continue
        if len(run) >= 3:
            toc.update(run)
            title = _norm_title(heads[i])
            if any(_norm_title(heads[j]) == title for j in order[k + 1:]):
                toc.add(i)
        run = []
    return {i: t for i, t in heads.items() if i not in toc}


def _sections(blocks: list[str], is_head, start: int = 0, dedupe: bool = False) -> list[tuple[str, list[str]]]:
    heads = {i: is_head(b) for i, b in enumerate(blocks) if i >= start}
    heads = {i: t for i, t in heads.items() if t}
    if dedupe:
        heads = _drop_toc_entries(blocks, heads)
    sections: list[tuple[str, list[str]]] = [("Front matter", [])]
    for i, block in enumerate(blocks):
        if i in heads:
            sections.append((heads[i], []))
        else:
            sections[-1][1].append(block)
    return sections


def _merge(sections: list[tuple[str, list[str]]]) -> list[tuple[str, str]]:
    merged: list[tuple[str, list[str]]] = []
    for i, (title, body) in enumerate(sections):
        # A heading followed by a short subtitle line keeps the subtitle in its title.
        first = body[0].strip() if body else ""
        if (i > 0 and len(title) <= 25 and first and len(first) <= 60 and "\n" not in first
                and not first.endswith((".", "!", "?", ",", ";", ":", '"', "”", "’", "'"))):
            title = f"{title} {body[0].strip()}"
            body = body[1:]
        size = sum(len(b) for b in body)
        if merged and size < MIN_CHAPTER_CHARS and i > 0:
            merged[-1][1].extend([title, *body])
        else:
            merged.append((title, list(body)))
    return [(t, "\n\n".join(b)) for t, b in merged if any(x.strip() for x in b)]


def split_chapters(text: str) -> list[Chapter]:
    """Split a cleaned book text into chapters.

    Three heading detectors are tried in order: numbered heading patterns (CHAPTER XII, Erstes Kapitel),
    entries of the table of contents found again as standalone lines, and standalone all-caps lines.
    Sections shorter than MIN_CHAPTER_CHARS are merged into the preceding one. Books without detectable
    structure fall back to fixed-size sections so every passage still gets a stable location.
    """
    blocks = _paragraph_blocks(text)
    candidates = [_sections(blocks, _pattern_detector(blocks), dedupe=True)]
    toc_head, toc_end = _toc_detector(blocks)
    if toc_head:
        candidates.append(_sections(blocks, toc_head, start=toc_end))
    candidates.append(_sections(blocks, _caps_detector(blocks), dedupe=True))

    for sections in candidates:
        chapters = _merge(sections)
        if sum(1 for t, _ in chapters if t != "Front matter") >= 3:
            return [Chapter(idx=i, title=_clean_title(t), text=body) for i, (t, body) in enumerate(chapters)]
    return _fixed_sections(text)


def _fixed_sections(text: str) -> list[Chapter]:
    out: list[Chapter] = []
    current: list[str] = []
    size = 0
    for block in _paragraph_blocks(text):
        current.append(block)
        size += len(block)
        if size >= FALLBACK_SECTION_CHARS:
            out.append(Chapter(idx=len(out), title=f"Section {len(out) + 1}", text="\n\n".join(current)))
            current, size = [], 0
    if current:
        out.append(Chapter(idx=len(out), title=f"Section {len(out) + 1}", text="\n\n".join(current)))
    return out


def _clean_title(title: str) -> str:
    title = re.sub(r"\[Illustration[^\]]*\]?", "", title)
    title = re.sub(r"[_\[\]=+~]", "", title)
    title = re.sub(r"\s+", " ", title).strip(" .")
    return title[:120]


def normalize_paragraph(block: str) -> str:
    """Join hard-wrapped lines and collapse whitespace; drop Gutenberg markup like _italics_."""
    text = re.sub(r"\s+", " ", block).strip()
    return re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)


_SENTENCE_RE = re.compile(r"(?<=[.!?…»«\"”])\s+(?=[A-ZÄÖÜ\"“„»«'‘(])")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_RE.split(text) if s]


def chunk_text(text: str, target_chars: int = 700, max_chars: int = 1000) -> list[str]:
    """Pack paragraphs into passages of roughly target_chars.

    Paragraph boundaries are preferred; paragraphs longer than max_chars are split at sentence
    boundaries. The last sentence of each passage is repeated at the start of the next one as
    overlap, so an answer spanning a boundary is still retrievable.
    """
    units: list[str] = []
    for block in _paragraph_blocks(text):
        para = normalize_paragraph(block)
        if len(para) <= max_chars:
            units.append(para)
            continue
        buf = ""
        for sent in split_sentences(para):
            if buf and len(buf) + len(sent) + 1 > target_chars:
                units.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            units.append(buf)

    chunks: list[str] = []
    buf = ""
    for unit in units:
        if buf and len(buf) + len(unit) + 1 > max_chars:
            chunks.append(buf)
            tail = split_sentences(buf)[-1] if len(buf) > target_chars // 2 else ""
            buf = tail if len(tail) < target_chars // 3 else ""
        buf = f"{buf}\n{unit}".strip() if buf else unit
        if len(buf) >= target_chars:
            chunks.append(buf)
            tail = split_sentences(buf)[-1]
            buf = tail if len(tail) < target_chars // 3 else ""
    if buf and (not chunks or not chunks[-1].endswith(buf)):
        chunks.append(buf)
    return [c[:max_chars * 2] for c in chunks if len(c) > 40]
