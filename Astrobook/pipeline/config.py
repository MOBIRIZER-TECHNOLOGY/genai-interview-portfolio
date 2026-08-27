"""Shared paths and constants. Repo-relative so this runs on any machine."""
import os
import re

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC    = os.path.join(ROOT, "Astro_book")
BUILD  = os.path.join(ROOT, "build")

CHUNKS = os.path.join(BUILD, "chunks.jsonl")
# The same units re-packed small for the embedder. See RETRIEVAL_TARGET in
# 01_extract.py for why the retrieval index cannot share the training chunks.
RETRIEVAL_CHUNKS = os.path.join(BUILD, "retrieval_chunks.jsonl")
PAIRS  = os.path.join(BUILD, "pairs.jsonl")
STATE  = os.path.join(BUILD, "batch_state.json")

def split_path(name):
    return os.path.join(BUILD, f"{name}.jsonl")

# No usable text layer -- these need OCR, out of scope for v1.
# (4 pure scans + the 3 Lal Kitab 1952 vols, whose only extractable text
#  is a repeated distributor watermark.)
SKIP = {
    "ashtaka-varga.pdf",
    "greatness-gayatri-jyotish.pdf",
    "astrology-and-stock-market-forecasting.pdf",
    "jyotisha-siddhanta-sara.pdf",
    "lal-kitab-vol-1-1952.pdf",
    "lal-kitab-vol-2-1952.pdf",
    "lal-kitab-vol-3-1952.pdf",
    # Has a real text layer, but the Hindi is set in a legacy non-Unicode font
    # (Krutidev-family), so it extracts as mojibake: garbled Devanagari transliteration bytes.
    # 346K chars of garbage would poison the training set. Needs font remapping
    # or OCR, not a decoder.
    "lal-kitab-1941.pdf",
}

SYSTEM_PROMPT = (
    "You are a study assistant for Vedic astrology (Jyotisha). You explain what "
    "the classical texts assert, cite the text you are drawing on, and distinguish "
    "between differing traditions. You describe doctrine; you do not make "
    "predictions about individuals."
)

os.makedirs(BUILD, exist_ok=True)


# ---------------------------------------------------------------------------
# Pair-generation prompt + schema. Shared by BOTH generators (Claude Batch and
# local Ollama) so the two datasets stay directly comparable. Do not fork these.
# ---------------------------------------------------------------------------
PAIRS_PER_CHUNK = 8

GEN_SYSTEM = """You write training data for a Vedic astrology (Jyotisha) study assistant.

You are given a passage from a classical or scholarly Jyotisha text. Write \
question/answer pairs answerable ENTIRELY from that passage. Never use outside \
knowledge; never invent a rule the passage does not state.

Spread the pairs across these types:
  - definitional: what a term, yoga, or technical concept means
  - rule-application: what the text prescribes for a specific placement
  - comparative: how elements relate, are classified, ranked, or ordered
  - explanatory: why the text reasons as it does, in its own terms

Rules for the answers:
  - Attribute to the source by name: "Saravali holds that...", "Per BPHS,...".
  - Describe what the TEXT ASSERTS. Write "the text holds that Mars in the 4th \
indicates X", never "Mars in your 4th house will cause X". You are explaining a \
tradition, not making predictions about a person.
  - Keep the text's Sanskrit terminology, glossing it briefly on first use.
  - 60-200 words. Complete prose, not bullet fragments.
  - EVERY QUESTION MUST STAND ALONE, as if asked by a student who has never seen \
the passage. At training time the passage will not be there. Never use the words \
"passage", "excerpt", "the text above", or "the provided text" inside a question.
      BAD:  "What is Arudha Pada according to the passage?"
      BAD:  "What does the passage say about the 3rd house?"
      GOOD: "How is Arudha Pada calculated in Jyotisha?"
      GOOD: "What does Brihat Parashara Hora Sastra associate with the 3rd house?"

Set `grounded` false and skip the pair if the passage is too fragmentary, too \
tabular, or too garbled by OCR to support a real question. A short honest batch \
beats a padded one."""

GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["definitional",
                             "rule-application", "comparative", "explanatory"]},
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "grounded": {"type": "boolean"},
                },
                "required": ["type", "question", "answer", "grounded"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["pairs"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Retrieved context handed to the model at inference.
#
# This lived as four different hard-coded slices -- 900 chars in chat.py, 800 in
# 12_grounding.py, a 2,600 total budget in 10_rag.py and serve.py -- all sized
# for the old 1,800-token chunks, where a slice was the only way to keep the
# prompt sane. Against those chunks the 900-char cut kept 22.1% of the answer's
# supporting words; the retriever found the right passage and the model was then
# shown the first eighth of it.
#
# Retrieval chunks are ~350 tokens (~1,400 chars), so they now fit WHOLE. One
# shared budget, used by every caller, so the next change cannot apply to three
# of the four.
CONTEXT_BUDGET = 6000          # characters, across all retrieved passages
CONTEXT_PER_PASSAGE = 2200     # safety cap; a normal retrieval chunk is under it


def build_context(hits, budget=CONTEXT_BUDGET, per_passage=CONTEXT_PER_PASSAGE):
    """Concatenate retrieved passages under a character budget.

    Whitespace is collapsed (PDF text arrives full of soft wraps), each passage
    is labelled with its source title so the model can cite it, and passages are
    dropped whole rather than half-included -- a truncated final passage is how
    you get a confident answer citing a rule whose exception was cut off.
    """
    parts, used = [], 0
    for h in hits:
        body = " ".join(h["text"].split())[:per_passage]
        if used + len(body) > budget:
            break
        parts.append(f"[{h['title']}]\n{body}")
        used += len(body)
    return "\n\n".join(parts)


def user_prompt(chunk, n=PAIRS_PER_CHUNK):
    return (f"Source: {chunk['title']}\n\n<passage>\n{chunk['text']}\n</passage>"
            f"\n\nWrite {n} pairs.")


def chunk_cid(chunk_id):
    """chunk id -> batch custom_id (must be [a-zA-Z0-9_-], <=64 chars)."""
    return chunk_id.replace("::", "__").replace(".pdf", "").replace(".", "_")
