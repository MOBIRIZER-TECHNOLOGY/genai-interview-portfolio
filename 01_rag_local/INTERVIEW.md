# 🎤 Interview notes — RAG

Questions a Sr GenAI interview actually asks about retrieval, with answers you
can defend, plus the trade-offs behind them. Numbers cited here come from
`eval/results.json` on this machine — re-run it before an interview so you're
quoting live results.

---

## The 60-second project pitch

> "Local RAG over a private corpus. Hybrid retrieval — dense BGE embeddings plus
> BM25, fused with reciprocal rank fusion — then a cross-encoder reranker cuts 20
> candidates down to 4, and a local 7B model answers with citations that I verify
> mechanically against the blocks I actually sent it. The part I care about most
> is the eval harness: 20 labelled questions, three of them deliberately
> unanswerable, measuring retrieval and generation separately. Reranking took
> recall@1 from 0.88 to 0.94. Abstention is 3 for 3. Warm p50 is 368 ms, all on
> one consumer GPU with nothing leaving the machine."

Two things that pitch does deliberately: it leads with **measurement**, and it
names a **limitation** you're ready to discuss. Both signal seniority.

---

## Core questions

### "Walk me through your chunking strategy."

Heading-aware, not fixed-size. I split on markdown headings so a table or a
section stays intact, and only sub-split a section if it exceeds ~320 tokens —
then on paragraph boundaries with ~60 tokens of overlap.

The detail worth mentioning: **every chunk is prefixed with its heading
breadcrumb** (`05-oncall-runbook.md > Severity ladder`). A chunk that reads
`| SEV2 | Throughput down 30% | 15 min |` is nearly meaningless in isolation —
it embeds poorly and retrieves badly. The breadcrumb carries its own context into
the embedding. It's a cheap version of Anthropic's "contextual retrieval" idea:
you can go further and have an LLM write a one-line context header per chunk at
ingest time, which costs more but helps more.

**Follow-up they'll ask: "how did you pick 320 tokens?"**
Honest answer: it's a starting point, and the eval harness is how you'd tune it.
Too small and a fact gets separated from the qualifier that makes it correct. Too
large and you dilute the embedding — a 2000-token chunk averages so many topics
that its vector points nowhere in particular, and you burn context window on
irrelevant text.

**The part I'd actually lead with, because it's the part that bit me:** the
budget is not the invariant that matters — *the embedder's window* is. My
chunker had two separate bugs that both ended in the same place, content past
512 tokens silently truncated and unretrievable:

1. A paragraph with no blank lines had no boundary to split on, so it passed
   through whole — 1,128 tokens against a 320 budget. Fixed with a hard split
   on sentence/space boundaries.
2. The fix was **incomplete**, and this is the more interesting one. The overlap
   carry copies the tail of the previous chunk forward, and it carried *whole
   paragraphs*. After the hard split, a paragraph **is** the whole budget — so
   every chunk after the first came out at 320 + 320 = **640 tokens**. The
   truncation bug, reintroduced one loop below its own fix.

Neither raised an error. Retrieval kept working, slightly worse, forever. The
lesson I'd offer: **assert the property, not the mechanism.** A test that says
"the section got split" passes on both broken versions. A test that says "no
chunk exceeds the embedder's window" fails on both — and that is the one I
now have, parametrised over the pathological inputs (one paragraph, no
sentences, no spaces at all).

### "Why hybrid retrieval? Aren't embeddings strictly better?"

They fail in opposite directions.

- Dense handles **paraphrase**. "How long do we keep camera footage" retrieves the
  retention table despite sharing almost no words with it.
- Dense is weak on **rare literal tokens**. A 384-dim embedding may have no
  distinct direction for `TLM-330` or `ATLAS_STARVATION_ROUNDS`.
- BM25 is the reverse: IDF gives rare tokens the *highest* weight, so exact
  identifiers are its strength — and a pure paraphrase scores zero.

Production corpora are full of error codes, config keys, SKUs, ticket IDs.
That's why hybrid is the default in real systems.

**Be ready for the honest caveat.** On my 30-chunk corpus hybrid tied with dense.
The corpus is too small for the failure mode to appear. I'd rather say that than
oversell a number.

### "Why RRF instead of a weighted score blend?"

Scale mismatch. Cosine similarity is bounded in ~[0.5, 0.9] for a normalised
encoder; BM25 is unbounded and depends on corpus statistics. Any `α·dense +
(1-α)·bm25` requires a normalisation that you must recalibrate whenever the
corpus changes — and it will rot silently.

RRF uses only rank: `Σ 1/(k + rank)`, k=60. Nothing to calibrate, robust to one
arm returning garbage scores. The cost is that you throw away score *magnitude* —
you lose the information that hit #1 was far better than hit #2. If you need
that, learn a fusion model, but then you need training data.

### "Explain bi-encoder vs cross-encoder, and why you use both."

A **bi-encoder** embeds the query and the passage independently and compares with
a dot product. Passages get embedded once, offline, so search is a single matrix
op — that's what makes it fast. But the model never sees query and passage
together, so it can't reason about whether *this* question is answered by *this*
text.

A **cross-encoder** concatenates `[query, passage]` into one forward pass and
outputs a relevance score. Much more accurate, and O(N) forward passes per query
— completely infeasible over a whole corpus.

So: retrieve 20 cheaply, rerank those 20 precisely, keep 4. Recall comes from
stage one, precision from stage two. That two-stage shape is the single
highest-leverage upgrade to naive RAG, and it's why my recall@1 went 0.88 → 0.94
for 40 ms.

### "How do you know your RAG system is any good?"

By measuring retrieval and generation **separately**, because they fail
separately and the fixes are completely different.

*Retrieval:* recall@k and MRR against hand-labelled gold documents. Recall@k
answers "did the answer reach the context window at all" — if that's low, no
prompt engineering will save you. MRR answers "is it ranked first", which is what
reranking moves.

*Generation:* given that the gold document *was* in context, did the answer state
the right facts, and did every citation resolve to a real block?

The split is diagnostic. Wrong answer + gold not retrieved → fix chunking,
embeddings, or the reranker. Wrong answer + gold *was* retrieved → fix the prompt
or the model. Without the split you're guessing.

**The metric most people skip: abstention.** Three of my 20 questions have no
answer in the corpus. A system that scores 95% on answerable questions and 0% on
abstention is *worse in production* than one scoring 85% and 100%, because the
first one lies fluently and gives you no signal that it did.

### "How do you stop hallucination?"

You reduce it and you detect it; you don't eliminate it. Four layers:

1. **Retrieval quality.** Most "hallucinations" are the model doing its best with
   the wrong paragraph. Fix retrieval first.
2. **An explicit abstain path.** `NOT_FOUND: <what's missing>` is a named,
   first-class output in the system prompt. Without one, the model's least-bad
   option is to guess.
3. **Numbered blocks + `[n]` citations.** Asking for citations "in free text"
   yields invented file names. A block index is a small, closed vocabulary.
4. **Mechanical verification.** `verify_citations()` parses every `[n]` and
   checks it against the blocks actually sent. A citation you can't check is
   decoration. This also gives you a production metric — ungrounded-answer rate —
   without needing an LLM judge.

The layer beyond this is an LLM-as-judge faithfulness check (does each sentence
follow from its cited block), which is what RAGAS does. It's more thorough and it
costs a model call per answer.

### "Your context window is 128k. Why not skip retrieval and stuff everything in?"

Sometimes you should — if the whole corpus is 50k tokens and latency doesn't
matter, that's genuinely simpler and you should say so rather than
over-engineering.

It stops working for three reasons:
- **Cost and latency** scale with prompt length on every single query. My prompt
  is ~600 tokens; the full corpus would be ~8k. That's 13× the prefill on every
  request, forever.
- **"Lost in the middle."** Accuracy on facts buried in the middle of a long
  context is measurably worse than facts at the start or end. More context is not
  monotonically better.
- **It doesn't scale.** It works at 8k tokens and fails at 8M. Retrieval is the
  thing that keeps working.

The middle ground worth naming: **prompt caching**. If your corpus is stable and
smallish, cache the prefix and you pay for it once. That changes the cost
calculus a lot.

### "How would you scale this to 10 million documents?"

Change these things, in this order:

1. **Index:** `IndexFlatIP` is O(N) exact search. Move to HNSW or IVF-PQ.
   Recall becomes a *tunable* (`efSearch`) rather than a guarantee — you now have
   an approximation error budget to measure. Add product quantisation when the
   raw vectors no longer fit in RAM.
2. **Sharding + metadata filters.** Most queries are scoped (one tenant, one
   product, a date range). Filtering before ANN search beats searching everything.
3. **Embedding throughput.** Batch, fp16, and treat re-embedding as a migration:
   you cannot mix embedding models in one index, so a model upgrade means a full
   rebuild with a dual-read cutover.
4. **Caching.** Query embeddings, and full answers for repeated questions. Real
   query distributions are extremely skewed.
5. **Two-stage stays.** Retrieve 100 → rerank 20 → keep 5. The reranker budget is
   what you tune under load.

### "What's the hardest bug you'd expect in a RAG system?"

**Embedding model mismatch between ingest and query.** If the index was built
with `bge-small` and you query with `bge-base`, you get vectors in a different
space — search still returns results, ranked by nothing meaningful. No exception,
no error log, just quietly terrible answers. That's why the model name is
persisted in `meta.json` and `RagPipeline.load()` reads it back rather than
letting the caller pick.

Runner-up: **the asymmetric prefix.** BGE expects an instruction prefix on
queries only. Forget it and recall drops a few points with nothing in the logs
to tell you.

**And the one I actually hit, which I'd rather talk about than either:** chunks
being silently truncated by the embedder because the overlap carry pushed them
to 2× the token budget (see the chunking answer above). It shares a shape with
both bugs above and it's the shape worth naming — **the failure is invisible at
every layer that could report it.** The chunker returns chunks. The embedder
returns vectors. Search returns results. Nothing is null, nothing throws, and
the answer is merely a bit worse than it should be, which is indistinguishable
from "RAG is hard".

That class — *degradation without an error* — is why I hold that grounding and
retrieval need **mechanical** checks rather than eyeballing: verified citations,
an abstention path, and invariant tests on the ingest side. If a failure can't
raise, it has to be asserted.

### "How do you know your tests are worth anything?"

Because I break the code and check they fail. Every regression test in this repo
was mutation-checked: reintroduce the original bug, confirm the suite goes red,
and note *how many* tests fire. Reintroducing the chunker bug fires 11.

That habit caught something worth admitting to. One of my chunking tests
asserted `chunks[0].body[-40:] in chunks[0].body` — a tautology. It passes for
any input, including no input. It had been sitting there advertising that
overlap was tested while testing nothing at all.

**A suite that passes on known-broken code is worse than no suite, because it
manufactures confidence.** The full argument, plus a 15-row ledger of the real
bugs and the test that pins each one, is in
[tests/INTERVIEW.md](../tests/INTERVIEW.md).

---

## Questions to ask *them*

These land well because they're the questions someone who's operated a RAG
system asks:

- "What does your eval set look like, and who maintains the labels?"
- "How do you detect retrieval quality regressing in production, as opposed to in CI?"
- "When you re-embed the corpus with a new model, what does the cutover look like?"
- "What's your abstention rate, and do you track it?"

---

## Related projects in this repo

- **[05_mcp_server](../05_mcp_server/)** wraps this exact pipeline as MCP tools,
  so Claude Code can query the Atlas corpus directly.
- **[06_local_gpu_inference](../06_local_gpu_inference/)** benchmarks the
  generation half — quantisation, batching and VRAM.
