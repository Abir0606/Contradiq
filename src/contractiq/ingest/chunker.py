import re
from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

_HEADER_PATTERNS = [
    re.compile(
        r"^\s*(?:ARTICLE|SECTION)\s+([0-9]+|[IVXLCDM]+)\b[\.\:\-\s]*(.*)$", re.IGNORECASE
    ),
    re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2})*)[\.\)]?\s+(\S.*)$"),
]
_MAX_HEADER_LEN = 120
_MAX_TITLE_LEN = 60
_MERGE_THRESHOLD = 250
_SENTENCE_LIKE = re.compile(r"[a-z]\.\s+[A-Z]")
_CLAUSE_START = re.compile(
    r"(?:(?:ARTICLE|SECTION)\s+\d{1,3}\b)"
    r"|\d{1,2}\.\d{1,2}(?:\.\d{1,2})*\.?\s+[\"(]?[A-Z]"
    r"|\d{1,2}\.\s+[\"(]?[A-Z]"
)


def _normalize(text: str) -> str:
    if text.count("\n") >= max(10, len(text) / 1500):
        return text
    text = re.sub(r"[ \t]{2,}(?=" + _CLAUSE_START.pattern + r")", "\n", text)
    text = re.sub(r"[ \t]{3,}", "\n\n", text)
    return text


@dataclass
class Chunk:
    text: str
    section_breadcrumb: str
    char_start: int


def _match_header(line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADER_LEN:
        return None
    for pattern in _HEADER_PATTERNS:
        m = pattern.match(line)
        if m:
            number, title = m.group(1), (m.group(2) or "").strip()
            title = title.strip(".:;- \t").rstrip(".")
            if (
                len(title) > _MAX_TITLE_LEN
                or len(title.split()) > 8
                or any(ch in title for ch in "\"'();")
                or _SENTENCE_LIKE.search(title)
            ):
                return None
            depth = number.count(".") + 1
            label = f"{number}. {title}" if title else number
            return depth, label
    return None


def _split_sections(text: str) -> list[tuple[tuple[tuple[int, str], ...], str]]:
    text = _normalize(text)
    sections: list[tuple[tuple[tuple[int, str], ...], str]] = []
    stack: list[tuple[int, str]] = []
    lines = text.split("\n")
    current: list[str] = []
    current_path: tuple[tuple[int, str], ...] = ()
    for line in lines:
        match = _match_header(line)
        if match is not None:
            if current:
                body = "\n".join(current)
                if body.strip():
                    sections.append((current_path, body))
                current = []
            depth, label = match
            while stack and stack[-1][0] >= depth:
                stack.pop()
            stack.append((depth, label))
            current_path = tuple(stack)
        else:
            current.append(line)
    if current:
        body = "\n".join(current)
        if body.strip():
            sections.append((current_path, body))
    return sections


def _breadcrumb(path: tuple[tuple[int, str], ...]) -> str:
    labels = [label for _, label in path][-3:]
    return " > ".join(labels) if labels else "Preamble"


def chunk_contract(
    text: str,
    chunk_size_tokens: int = 1200,
    chunk_overlap_tokens: int = 150,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o-mini",
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
    )
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    raw: list[Chunk] = []
    for path, body in _split_sections(text):
        pieces = splitter.split_text(body)
        for piece in pieces:
            raw.append(Chunk(text=piece, section_breadcrumb=_breadcrumb(path), char_start=0))

    merged: list[Chunk] = []
    buffer: list[str] | None = None
    buffer_bc: str | None = None

    def flush() -> None:
        nonlocal buffer, buffer_bc
        if buffer is not None:
            merged.append(
                Chunk(text="\n\n".join(buffer), section_breadcrumb=buffer_bc, char_start=0)
            )
            buffer = None
            buffer_bc = None

    for chunk in raw:
        tokens = len(encoding.encode(chunk.text))
        small = tokens < _MERGE_THRESHOLD
        if small and buffer_bc == chunk.section_breadcrumb and buffer is not None:
            candidate = "\n\n".join([*buffer, chunk.text])
            if len(encoding.encode(candidate)) <= chunk_size_tokens:
                buffer.append(chunk.text)
                continue
        flush()
        if small:
            buffer = [chunk.text]
            buffer_bc = chunk.section_breadcrumb
        else:
            merged.append(chunk)
    flush()

    return merged
