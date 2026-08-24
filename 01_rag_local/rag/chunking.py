"""
Markdown-aware chunking.

Why not just `text.split()` every N characters? Because a naive split cuts
tables in half and orphans a fact from the heading that gives it meaning. A
chunk that reads "| `TLM-330` | Hypertable chunk write failed |" with no
surrounding context embeds poorly and retrieves badly.

Strategy here:
  1. Split the document on markdown headings, so each section stays whole.
  2. If a section is longer than the token budget, split it further on blank
     lines with an overlap, never mid-line.
  3. Prefix every chunk with its heading breadcrumb ("Doc > H1 > H2"). This is
     cheap "contextual retrieval" -- the chunk carries its own context into the
     embedding, which measurably lifts recall on short factual chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    id: str
    text: str          # what gets embedded (includes the breadcrumb prefix)
    body: str          # the raw section text, for display
    source: str        # file name
    breadcrumb: str    # "05-oncall-runbook.md > Atlas on-call runbook > Severity ladder"
    n_tokens: int
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def approx_tokens(text: str) -> int:
    """Cheap token estimate: ~4 chars per token for English prose.

    Deliberately not calling a real tokenizer here -- chunking runs over the
    whole corpus and an approximation within 10% is enough to pick boundaries.
    The embedding model truncates precisely anyway.
    """
    return max(1, len(text) // 4)


def _split_long(section: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split an over-long section on paragraph boundaries, with overlap."""
    paras = [p for p in section.split("\n\n") if p.strip()]
    out: list[str] = []
    cur: list[str] = []
    cur_tokens = 0

    for para in paras:
        pt = approx_tokens(para)
        if cur and cur_tokens + pt > max_tokens:
            out.append("\n\n".join(cur))
            # carry the tail of the previous chunk forward so a fact split
            # across a boundary is still fully present in one of the chunks
            carry: list[str] = []
            carried = 0
            for p in reversed(cur):
                if carried >= overlap_tokens:
                    break
                carry.insert(0, p)
                carried += approx_tokens(p)
            cur, cur_tokens = carry, carried
        cur.append(para)
        cur_tokens += pt

    if cur:
        out.append("\n\n".join(cur))
    return out


def chunk_markdown(
    path: Path,
    max_tokens: int = 320,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    """Turn one markdown file into a list of retrievable chunks."""
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    # walk the file, accumulating (heading-stack, body-lines) sections
    sections: list[tuple[list[str], list[str]]] = []
    stack: list[str] = []
    buf: list[str] = []

    for line in lines:
        m = HEADING.match(line)
        if m:
            if buf and any(s.strip() for s in buf):
                sections.append((list(stack), buf))
            buf = []
            level, title = len(m.group(1)), m.group(2).strip()
            stack = stack[: level - 1]
            while len(stack) < level - 1:
                stack.append("")
            stack.append(title)
        else:
            buf.append(line)

    if buf and any(s.strip() for s in buf):
        sections.append((list(stack), buf))

    chunks: list[Chunk] = []
    for headings, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        breadcrumb = " > ".join([path.name] + [h for h in headings if h])

        pieces = (
            [body]
            if approx_tokens(body) <= max_tokens
            else _split_long(body, max_tokens, overlap_tokens)
        )
        for i, piece in enumerate(pieces):
            text = f"{breadcrumb}\n\n{piece}"
            chunks.append(
                Chunk(
                    id=f"{path.stem}::{len(chunks):03d}",
                    text=text,
                    body=piece,
                    source=path.name,
                    breadcrumb=breadcrumb,
                    n_tokens=approx_tokens(text),
                    meta={"part": i, "of": len(pieces)},
                )
            )
    return chunks


def chunk_corpus(corpus_dir: Path, **kw) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path, **kw))
    return chunks
