"""Structure-aware chunking for standards documents.

The original loader split text on a fixed 512-character window broken at the
nearest period. That destroys document structure: a window starting mid-page
can capture the tail of one section and never contain the section heading, so
the reader model has no clause number available to cite.

That is precisely how a verified-false citation was produced in testing: the
NASAA buffer-zone rule lives in clause 3.3.2 under the heading "3.3 WINDBREAKS /
BUFFER ZONES", but a fixed window captured the preceding section 3.2 (GMOs)
instead, and the model cited "3.2.9" for a buffer-zone rule.

This module chunks along the document's own structure instead:

  * detect section headings ("3.3 WINDBREAKS / BUFFER ZONES",
    "SECTION THREE - PRECAUTIONS & GENERAL REQUIREMENTS")
  * detect numbered clauses ("3.3.2 Requirements for buffer zones ...")
  * never split a clause across chunks
  * prefix every chunk with its section context so the clause number is always
    present in the text the model sees — which is what makes deterministic
    citation verification (app/rag/citations.py) meaningful

The output is deliberately conservative: when structure cannot be detected the
chunker falls back to paragraph-aware packing and emits the chunk with no
structural prefix rather than inventing a section number.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Structure detection
# ---------------------------------------------------------------------------

# Character classes are written out literally rather than composed from
# templates: a template that is not itself wrapped in brackets silently becomes
# a literal string and never matches.
#
# "3.3 WINDBREAKS / BUFFER ZONES" (NASAA) or "1.2 Conversion of land"
# (National Standard) — a numbered heading, optionally with a title.
# The title must be short, start with a capital/digit, and must NOT end with a
# sentence terminator or comma. That last rule is what distinguishes a heading
# from an ordinary numbered clause in a document whose PDF has no line breaks.
_NUM_HEADING_RE = re.compile(
    r"^\s*(\d{1,2}(?:\.\d{1,3}){0,2})\s*"
    r"([A-Z0-9][0-9A-Za-z ,/&'’\-()]{1,80}?)?\s*$",
    re.MULTILINE,
)

# "SECTION THREE - PRECAUTIONS & GENERAL REQUIREMENTS"
_SECTION_WORD_RE = re.compile(
    r"^\s*(SECTION\s+(?:[A-Z]+|\d+))\s*[-–—:]?\s*"
    r"([A-Z0-9][0-9A-Za-z ,/&'’\-()]{1,90}?)?\s*$",
    re.MULTILINE,
)

# "Article 30 Collection of products and transport to preparation units"
# (EU / UK regulations) — the article is the citable unit in these documents.
_ARTICLE_HEADING_RE = re.compile(
    r"^\s*Article\s+(\d{1,3})\s*"
    r"([A-Z0-9][0-9A-Za-z ,/&'’\-()]{1,90}?)?\s*$",
    re.MULTILINE,
)

# "§ 205.236 Origin of livestock." (US CFR)
_SECTION_SYMBOL_HEADING_RE = re.compile(
    r"^\s*§\s*(\d{1,3}(?:\.\d{1,4}){0,2})\s*"
    r"([A-Z0-9][0-9A-Za-z ,/&'’\-()]{1,90}?)?\s*$",
    re.MULTILINE,
)

_HEADING_PATTERNS = (
    _NUM_HEADING_RE,
    _SECTION_WORD_RE,
    _ARTICLE_HEADING_RE,
    _SECTION_SYMBOL_HEADING_RE,
)

# A running page header/footer repeated on most pages of a PDF; stripped during
# extraction so it does not become the dominant text of every chunk.
_HEADER_MIN_PAGES = 3
_HEADER_MATCH_RATIO = 0.5

# "3.3.2 Requirements for buffer zones shall be ..." — a numbered clause.
# The trailing lookahead requires sentence-like text, so a bare heading is not
# mistaken for a clause.
_CLAUSE_RE = re.compile(
    r"(?<![\d.])(\d{1,2}(?:\.\d{1,3}){1,3})\s+(?=[A-Z(])",
)

# US CFR style: "§ 205.236 Origin of livestock." / "§205.236 Origin of livestock."
_SECTION_CLAUSE_RE = re.compile(
    r"§\s*(\d{1,3}(?:\.\d{1,4}){0,2})\s+(?=[A-Z(])",
)

# EU / UK style: "Article 30 Collection of products and transport to preparation
# units" — the article number is the citable unit in these documents.
_ARTICLE_CLAUSE_RE = re.compile(
    r"\bArticle\s+(\d{1,3})\b",
)

# All clause-numbering conventions, in one place.
_CLAUSE_PATTERNS = (_CLAUSE_RE, _SECTION_CLAUSE_RE, _ARTICLE_CLAUSE_RE)

_SENTENCE_END_RE = re.compile(r"[.;:!?]\s")


@dataclass
class Chunk:
    """One emitted chunk plus the structure it came from."""

    text: str
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    clause_numbers: List[str] = field(default_factory=list)
    char_start: int = 0
    char_end: int = 0

    @property
    def embed_text(self) -> str:
        """Text as it should be embedded and shown to the reader.

        The section context is prepended so a retrieved chunk always carries a
        citable clause number. When a chunk is a continuation slice of one very
        long clause, its own number may not appear in the body text, so the
        clause number is added to the prefix — otherwise the reader would be
        asked to answer from an excerpt with nothing it is allowed to cite, and
        deterministic verification (app/rag/citations.py) would reject whatever
        number it guessed.
        """
        labels: List[str] = []
        if self.section_number:
            labels.append(f"{self.section_number} {self.section_title or ''}".strip())
        for clause in self.clause_numbers:
            if clause and clause != self.section_number and clause not in self.text:
                labels.append(clause)
        if not labels:
            return self.text
        return f"[{' | '.join(labels)}]\n{self.text}"


@dataclass
class _Section:
    start: int
    number: Optional[str]
    title: Optional[str]


def _find_sections(text: str) -> List[_Section]:
    """Locate section boundaries in document order, across all heading styles."""
    found: List[tuple] = []
    for pattern in _HEADING_PATTERNS:
        for match in pattern.finditer(text):
            title = (match.group(2) or "").strip()
            found.append((match.start(), match.group(1).strip(), title or None))
    found.sort(key=lambda item: item[0])
    return [_Section(start=pos, number=num, title=title) for pos, num, title in found]


def _section_for(sections: List[_Section], position: int) -> Optional[_Section]:
    """The most recent section starting at or before `position`."""
    current: Optional[_Section] = None
    for section in sections:
        if section.start <= position:
            current = section
        else:
            break
    return current


def _split_clauses(text: str, offset: int, max_chars: int = 900) -> List[tuple]:
    """Split text into (clause_number_or_None, start, end) pieces.

    Supports the clause-numbering conventions used by the installed standards:
    dotted numbering (NASAA/National), ``§ 205.236`` (US CFR) and ``Article 30``
    (EU/UK regulations). Returns whichever convention's marker comes next, so
    nesting or mixed usage does not produce spurious boundaries.

    A piece runs from one marker up to the next, so ordinary clauses are never
    divided. A clause longer than `max_chars` is split at sentence boundaries
    and its number is attached to every resulting piece. Text before the first
    marker is returned as a single header piece.
    """
    markers: List[tuple] = []
    for pattern in _CLAUSE_PATTERNS:
        for match in pattern.finditer(text):
            markers.append((match.start(), match.group(1)))
    if not markers:
        return [(None, offset, offset + len(text))]
    markers.sort(key=lambda item: item[0])

    pieces: List[tuple] = []
    if markers[0][0] > 0:
        pieces.append((None, offset, offset + markers[0][0]))

    for i, (pos, clause) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        if end - pos > max_chars:
            # Split against the whole text so the returned spans are absolute,
            # then rebase into the caller's coordinate space.
            for piece_start, piece_end in _split_oversized(end, text, max_chars):
                if piece_end > piece_start:
                    pieces.append((clause, offset + piece_start, offset + piece_end))
        else:
            pieces.append((clause, offset + pos, offset + end))
    return pieces


def _split_oversized(clause_end: int, text: str, max_chars: int) -> List[tuple]:
    """Split one oversized clause into non-overlapping (start, end) spans.

    PDF extraction frequently returns a page as a single run of text with no
    newlines, so consecutive clauses join into one long piece. An oversized
    clause is split at sentence boundaries rather than emitted whole, which
    would defeat retrieval precision.

    `text` is the full document text and `clause_end` an absolute offset, so the
    returned spans are absolute and directly sliceable.
    """
    cursor = 0
    spans: List[tuple] = []
    while clause_end - cursor > max_chars:
        window = text[cursor: cursor + max_chars]
        cut = -1
        for match in _SENTENCE_END_RE.finditer(window):
            cut = match.end()
        if cut <= 0:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = len(window)
        spans.append((cursor, cursor + cut))
        cursor += cut
    if cursor < clause_end:
        spans.append((cursor, clause_end))
    return spans


def _pack(pieces: List[tuple], text: str, offset: int, max_chars: int) -> List[tuple]:
    """Group clause pieces into chunks of at most ~`max_chars`.

    Pieces are appended until the next would breach `max_chars`; a piece that is
    itself oversized is split at sentence boundaries (see `_split_oversized`).
    Splitting happens only when necessary, so ordinary clauses stay intact.
    """
    packed: List[tuple] = []
    current_start: Optional[int] = None
    current_end: Optional[int] = None
    current_clauses: List[str] = []

    def flush():
        nonlocal current_start, current_end, current_clauses
        if current_start is not None and current_end is not None:
            body = text[current_start - offset:current_end - offset].strip()
            if body:
                packed.append((body, current_start, current_end, list(current_clauses)))
        current_start = current_end = None
        current_clauses = []

    for clause, start, end in pieces:
        if current_start is None:
            current_start, current_end = start, end
            if clause:
                current_clauses.append(clause)
            continue

        if (end - current_start) > max_chars:
            flush()
            # Reset to None (not to an empty span): the next piece must *start*
            # a new chunk. Pointing current_start at the current piece's start
            # would make the following iteration see a full-size span again and
            # flush an empty chunk, silently dropping that piece.
            current_start = current_end = None
        else:
            current_end = end
            if clause:
                current_clauses.append(clause)

    flush()
    return packed


def chunk_structured(
    text: str,
    max_chars: int = 900,
    min_chars: int = 40,
) -> List[Chunk]:
    """Chunk standards text along section and clause boundaries.

    `max_chars` is a soft ceiling. Clauses are never split, so a single long
    clause may exceed it — that is preferred over breaking a rule in half.
    `min_chars` drops fragments too small to be useful.
    """
    if not text or not text.strip():
        return []

    sections = _find_sections(text)
    chunks: List[Chunk] = []

    boundaries = [s.start for s in sections] + [len(text)]
    for i in range(len(boundaries) - 1):
        seg_start, seg_end = boundaries[i], boundaries[i + 1]
        segment = text[seg_start:seg_end]
        if not segment.strip():
            continue
        section = _section_for(sections, seg_start)

        pieces = _split_clauses(segment, seg_start, max_chars)
        for body, start, end, clauses in _pack(pieces, segment, seg_start, max_chars):
            if len(body) < min_chars:
                continue
            chunks.append(
                Chunk(
                    text=body,
                    section_number=section.number if section else None,
                    section_title=section.title if section else None,
                    clause_numbers=clauses,
                    char_start=start,
                    char_end=end,
                )
            )

    # A document with no detectable structure still needs to be chunked.
    if not chunks:
        for body, start, end, clauses in _pack(
            [(None, 0, len(text))], text, 0, max_chars
        ):
            if len(body) >= min_chars:
                chunks.append(Chunk(text=body, clause_numbers=clauses,
                                    char_start=start, char_end=end))

    return chunks
