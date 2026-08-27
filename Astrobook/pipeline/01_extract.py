"""
Stage 1: PDF -> cleaned, verse-aware chunks.jsonl.   [local, CPU, free]

Feeds both the LoRA data-generation stage and any RAG index you build later.
Skips the 7 image-only PDFs (no text layer to extract).

    python pipeline/01_extract.py [--report]
"""
import argparse, glob, json, os, re, sys
from collections import Counter

import fitz  # PyMuPDF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SRC, BUILD, CHUNKS, RETRIEVAL_CHUNKS, SKIP

TARGET, HARD_MAX = 1800, 2600          # chunk size, in approx tokens (chars/4)
MIN_CHUNK = 120                        # discard slivers below this

# ---------------------------------------------------------------- retrieval
# A SECOND chunking of the same units, for the retrieval index only.
#
# 1,800 tokens is right for generating training pairs -- a teacher model needs
# enough context to write eight non-trivial questions. It is wrong for
# retrieval: bge-small-en-v1.5 truncates at 512 tokens, so indexing the
# training chunks meant 98.6% of them were cut and only 31.7% of the corpus was
# ever embedded. The dense half of the hybrid index could not see two thirds of
# the books, and BM25 -- which reads the whole chunk -- hid it by returning the
# right source anyway.
#
# 350 leaves room for the title prefix and the [CLS]/[SEP] tokens inside the
# 512 budget with margin to spare. The overlap keeps a rule that straddles a
# boundary retrievable from either side.
RETRIEVAL_TARGET, RETRIEVAL_OVERLAP = 350, 60
RETRIEVAL_MIN = 40

# A verse / sutra / chapter marker. Matched at LINE start (MULTILINE), not just
# paragraph start -- the cleaner collapses soft wraps, so many books arrive as
# one long run of lines with no blank-line paragraph breaks left to split on.
VERSE_LINE = re.compile(
    r"^(?=\s*(?:"
    r"(?:Sloka|Slokas|Stanza|Verse|Chapter|Adhyaya|Sutra|Ch\.)\s*\.?\s*[\dIVXLC]+"
    r"|[\dIVXLC]{1,4}(?:\s*[-\u2013]\s*[\dIVXLC]{1,4}[\u00bd\u00bc\u00be]?)?\s*[.)]\s+[A-Z\u015a\u1e62]"
    r"))",
    re.M,
)

# Sentence boundary, last-resort splitter. Avoids the common abbreviations that
# appear in this corpus so we do not cut "Ch. 4" or "viz. the" in half.
ABBR = r"(?<!\bviz)(?<!\bCh)(?<!\bNo)(?<!\bvol)(?<!\bVol)(?<!\bp)(?<!\bpp)(?<!\bFig)(?<!\bi\.e)(?<!\be\.g)"
SENT = re.compile(ABBR + r"(?<=[.!?\u2019\"])\s+(?=[A-Z\u015a\u1e62\u1e5a\d])")

LEADER = re.compile(r"\.{4,}\s*\d+\s*$", re.M)


def approx_tokens(s):
    return len(s) // 4


def find_furniture(pages, min_frac=0.25):
    """Lines repeating across >=25% of pages are running heads/feet."""
    seen = Counter()
    for t in pages:
        lines = [l.strip() for l in t.splitlines() if l.strip()]
        for l in set(lines[:2] + lines[-2:]):
            seen[re.sub(r"\d+", "#", l)] += 1          # collapse page numbers
    cut = max(3, int(len(pages) * min_frac))
    return {k for k, v in seen.items() if v >= cut and len(k) < 90}


def is_front_matter(text):
    """TOC / index page: dominated by dot-leader lines."""
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 5:
        return False
    return len(LEADER.findall(text)) >= max(4, len(lines) * 0.35)


def clean(text, furniture):
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            out.append("")
            continue
        if re.sub(r"\d+", "#", s) in furniture:
            continue
        if re.fullmatch(r"[\dIVXLCivxlc\s.\-\u2013\u2014|]+", s):   # bare page no.
            continue
        out.append(s)
    t = "\n".join(out)
    # Private-use-area codepoints: broken/subsetted symbol fonts render as
    # garbage here (crux-of-astrology alone emits ~2.1k of them).
    t = re.sub(r"[-]", "", t)
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)                 # rejoin hyphenation
    t = re.sub(r"(?<![.!?:;\"'\)])\n(?=[a-z])", " ", t)    # unwrap soft breaks
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _split_keep(text, rx):
    """Split at every match position, keeping the marker with the text after it."""
    idx = [m.start() for m in rx.finditer(text)]
    if not idx:
        return [text]
    if idx[0] != 0:
        idx.insert(0, 0)
    idx.append(len(text))
    return [text[idx[i]:idx[i + 1]].strip()
            for i in range(len(idx) - 1) if text[idx[i]:idx[i + 1]].strip()]


def _sentence_pack(text, cap):
    """Break an oversized run on sentence boundaries -- never mid-sentence."""
    sents = SENT.split(text)
    out, buf, n = [], [], 0
    for s in sents:
        st = approx_tokens(s)
        if st > cap:                    # a single monstrous "sentence" (a table)
            if buf:
                out.append(" ".join(buf)); buf, n = [], 0
            step = cap * 4
            out += [s[i:i + step] for i in range(0, len(s), step)]
            continue
        if n + st > cap and buf:
            out.append(" ".join(buf)); buf, n = [], 0
        buf.append(s); n += st
    if buf:
        out.append(" ".join(buf))
    return out


def _merge_orphans(units):
    """A real verse/paragraph/sentence unit never begins with a lowercase word.
    One that does is the tail of the unit before it, severed by a bogus boundary
    -- usually a verse marker that matched inside running prose. Glue it back."""
    out = []
    for u in units:
        if (out and u[:1].islower()
                and approx_tokens(out[-1]) + approx_tokens(u) <= HARD_MAX):
            out[-1] = out[-1].rstrip() + " " + u.lstrip()
        else:
            out.append(u)
    return out


def split_units(text):
    """Cascade: verse markers -> blank-line paragraphs -> sentences."""
    units = []
    for part in _split_keep(text, VERSE_LINE):
        if approx_tokens(part) <= HARD_MAX:
            units.append(part)
            continue
        for para in re.split(r"\n\s*\n", part):
            para = para.strip()
            if not para:
                continue
            if approx_tokens(para) <= HARD_MAX:
                units.append(para)
            else:
                units += _sentence_pack(para, HARD_MAX)
    return _merge_orphans(units)


def pack(units, src, title, target=TARGET, min_chunk=MIN_CHUNK, overlap=0,
         id_prefix="", fine_split=False):
    """Greedily fill chunks toward `target` without ever splitting a unit.

    Defaults reproduce the original training chunks exactly -- chunks.jsonl is
    the input the shipped adapter's dataset was generated from, so it must not
    move. The parameters exist for the retrieval pass, which wants a very
    different size (see RETRIEVAL_TARGET).

    fine_split  a unit larger than `target` is broken on sentence boundaries
                first. Off for training, where a unit up to HARD_MAX is allowed
                to be a chunk by itself; on for retrieval, where a 2,600-token
                unit would defeat the entire point of a small chunk.
    overlap     tokens of the previous chunk to repeat at the start of the next,
                so a rule split across a boundary is still findable from either
                side. Zero for training: repeating text there would just
                generate duplicate Q&A pairs.
    """
    # An overlap approaching the target would leave no room for new material and
    # stall forward progress. Half the target is already far more than useful.
    overlap = min(overlap, target // 2)

    if fine_split:
        fine = []
        for u in units:
            fine += _sentence_pack(u, target) if approx_tokens(u) > target else [u]
        units = fine

    chunks, buf, n = [], [], 0

    def flush():
        nonlocal buf, n
        if buf:
            body = "\n\n".join(buf).strip()
            if approx_tokens(body) >= min_chunk:
                chunks.append({
                    "id": f"{src}::{id_prefix}{len(chunks):04d}",
                    "source": src,
                    "title": title,
                    "text": body,
                    "approx_tokens": approx_tokens(body),
                })
            # Seed the next chunk with the tail of this one. Walk backwards so
            # the carried text is the units nearest the boundary.
            carry, carried = [], 0
            if overlap:
                for u in reversed(buf):
                    ut = approx_tokens(u)
                    if carried + ut > overlap:
                        break
                    carry.insert(0, u)
                    carried += ut
            buf, n = carry, carried
        else:
            buf, n = [], 0

    for u in units:
        ut = approx_tokens(u)
        if n + ut > target and buf:
            flush()
        buf.append(u)
        n += ut
    # final flush must not re-carry, or the last chunk duplicates its own tail
    overlap = 0
    flush()
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="print size histogram")
    args = ap.parse_args()

    all_chunks, retrieval_chunks, report = [], [], []
    for path in sorted(glob.glob(os.path.join(SRC, "*.pdf"))):
        name = os.path.basename(path)
        if name in SKIP:
            continue
        title = name[:-4].replace("-", " ").title()

        doc = fitz.open(path)
        pages = [doc[i].get_text("text") for i in range(len(doc))]
        doc.close()

        furniture = find_furniture(pages)
        body = [p for p in pages if not is_front_matter(p)]
        text = "\n\n".join(clean(p, furniture) for p in body)
        # ONE extraction, TWO chunkings. split_units() is the expensive, fragile
        # part -- the verse/paragraph/sentence cascade and the orphan merge --
        # and both consumers want its output at different granularities. Running
        # it once guarantees the retrieval index and the training data are built
        # from identical text.
        units = split_units(text)
        chunks = pack(units, name, title)
        all_chunks += chunks
        retrieval_chunks += pack(
            units, name, title,
            target=RETRIEVAL_TARGET, min_chunk=RETRIEVAL_MIN,
            overlap=RETRIEVAL_OVERLAP, id_prefix="r", fine_split=True)

        report.append({"file": name, "pages": len(pages),
                       "front_matter_dropped": len(pages) - len(body),
                       "furniture_lines": len(furniture),
                       "chars": len(text), "chunks": len(chunks)})
        print(f"{name:<44}{len(pages):>5}p  -{len(pages)-len(body):<3}toc  "
              f"{len(furniture):>2}hdr  {len(text):>9,}ch  {len(chunks):>5} chunks")

    with open(CHUNKS, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(RETRIEVAL_CHUNKS, "w", encoding="utf-8") as f:
        for c in retrieval_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(os.path.join(BUILD, "extract_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    tok = sum(c["approx_tokens"] for c in all_chunks)
    print(f"\n{len(report)} files -> {len(all_chunks):,} chunks, ~{tok:,} tokens")
    print(f"mean {tok // max(1, len(all_chunks))} tok/chunk")

    if args.report:
        h = Counter(c["approx_tokens"] // 400 * 400 for c in all_chunks)
        pinned = sum(1 for c in all_chunks if c["approx_tokens"] >= HARD_MAX - 40)
        print("\nsize histogram:")
        for k in sorted(h):
            print(f"  {k:>5}-{k+399:<5} {'#' * max(1, h[k] // 4)} {h[k]}")
        print(f"\npinned at hard cap: {pinned} ({100*pinned/max(1,len(all_chunks)):.1f}%)"
              "   <- want this near zero")
    rt = sum(c["approx_tokens"] for c in retrieval_chunks)
    sizes = sorted(c["approx_tokens"] for c in retrieval_chunks)
    m = sizes[len(sizes) // 2] if sizes else 0
    print(f"retrieval  -> {len(retrieval_chunks):,} chunks, ~{rt:,} tokens "
          f"(median {m}, max {sizes[-1] if sizes else 0}, cap {RETRIEVAL_TARGET})")
    over = sum(1 for x in sizes if x > 480)
    print(f"           {over} chunks over 480 approx-tokens "
          f"({100*over/max(1,len(sizes)):.1f}%)  <- must stay near zero, the "
          "embedder truncates at 512")
    print(f"wrote {CHUNKS}")
    print(f"wrote {RETRIEVAL_CHUNKS}")


if __name__ == "__main__":
    main()
