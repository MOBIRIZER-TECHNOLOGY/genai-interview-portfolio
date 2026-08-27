"""Quality control for generated instruction pairs.

Applied by both generators (Ollama and Claude Batch) before anything is written,
so the training set never contains the failure modes below.
"""
import re

# "... according to the passage", "..., as described in the excerpt"
_PASSAGE_PHRASE = re.compile(
    r"[,\s]*(?:\bas\s+)?"
    r"\b(?:according\s+to|per|described\s+in|mentioned\s+in|stated\s+in|"
    r"given\s+in|found\s+in|outlined\s+in|referenced\s+in|in|from)\s+"
    r"(?:the\s+|this\s+)?"
    r"(?:passage|excerpt|text\s+above|above\s+text|provided\s+text|given\s+text)"
    r"\b[,\s]*",
    re.I,
)

# Any surviving self-reference means the question is BUILT on it.
_PASSAGE_WORD = re.compile(
    r"\b(?:passage|excerpt|text\s+above|provided\s+text|given\s+text)\b", re.I
)


def sanitize_question(q):
    """Strip passage self-references. Return None if unsalvageable.

    Models violate the stand-alone rule ~27% of the time even when explicitly
    told not to. A question referencing a passage the student model never sees
    teaches it to hallucinate a context it was never given, so these must not
    reach the training set.

    Trailing phrases are stripped ("What is X according to the passage?" ->
    "What is X?"); questions whose subject IS the passage ("What does the
    passage say about X?") cannot be repaired and are dropped.
    """
    if not q:
        return None
    q = _PASSAGE_PHRASE.sub(" ", q)
    q = re.sub(r"\s{2,}", " ", q)
    q = re.sub(r"\s+([?.!,;:])", r"\1", q)      # "Arudha Pada ?" -> "Arudha Pada?"
    q = q.strip(" ,;:-")
    if not q or _PASSAGE_WORD.search(q):
        return None
    if not q.endswith("?"):
        q += "?"
    return q[0].upper() + q[1:]


def acceptable(question, answer, min_q=15, min_a=55):
    """Cheap structural gate. Returns (ok, reason)."""
    if not question or len(question) < min_q:
        return False, "question too short"
    if not answer or len(answer) < min_a:
        return False, "answer too short"
    if _PASSAGE_WORD.search(answer):
        return False, "answer references the passage"
    # A model that loses the plot repeats one phrase to fill the token budget.
    words = answer.lower().split()
    if len(words) > 25 and len(set(words)) / len(words) < 0.35:
        return False, "degenerate repetition"
    return True, ""
