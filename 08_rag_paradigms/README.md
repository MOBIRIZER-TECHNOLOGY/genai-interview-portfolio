# 🧭 Project 08 — Three RAG Paradigms: vector vs graph vs agentic, measured

Implement GraphRAG and Agentic RAG next to the project-01 vector pipeline, run
all three over the **same corpus** against the **same labelled questions**, and
answer the only question that matters: *when does each win?*

> **In one sentence:** the fancy paradigms **lost** — vector RAG scored 100% on
> the multi-hop set that graph and agentic were built for, and the *why* is the
> most useful finding in this project.

---

## 🧠 The idea (for non-experts)

"RAG" is not one thing. Three paradigms differ in *who decides what gets
retrieved*:

| paradigm | retrieval is driven by | built for |
|---|---|---|
| **Vector** (project 01) | similarity between the question and chunks | lookups, paraphrase |
| **Graph** (this project) | entities in the question + edges between facts | relational joins: "the X of the Y that Z" |
| **Agentic** (this project) | an LLM that searches, reads, and searches again | compositional questions where hop 2 depends on reading hop 1 |

Standard theory: vector RAG fails on **multi-hop** questions ("what action fixes
the incident that causes the most pages?") because no chunk resembles the whole
question — the join lives across chunks. Graphs store the join as an edge;
agents discover it by iterating.

This project implements all three, holds the corpus, questions, generator and
grounding discipline constant, and measures.

---

## ✅ The result — and it contradicts the theory (here)

Two question sets: project 01's 20 **standard** questions, plus 7 hand-written
**multihop** questions whose answers require joining facts from different
chunks (each chain verified against the corpus by hand).

| paradigm | standard fact recall | multihop fact recall | p50 | LLM calls/q |
|---|---:|---:|---:|---:|
| **vector** | **100%** | **100%** | 353 ms | 1.0 |
| graph | 65% | 29% | 296 ms | 0.8 |
| agentic | 94% | 29% | 801–951 ms | 3.2–4.0 |

(All three: 100% abstention on the unanswerable questions — grounding discipline
held everywhere.)

### 🔍 Why vector won multihop — the finding that matters

**At 30 chunks, "multi-hop" is not actually hard.** Hybrid retrieval with top-4
pulls **both** hop chunks into context simultaneously — top-4 of 30 is 13% of
the corpus — and the generator does the join in-context:

```
m2: "Restart ntp-relay in the cell namespace to fix the incident type that
     causes the most pages, which is TLM-101..."     <- both hops, one shot
```

Multi-hop retrieval is only structurally hard when the hops **cannot
co-retrieve** — when top-k over the question misses the second hop's chunk.

### 🧪 That mechanism was an argument. Now it's measured — and it was partly wrong

`probe_scale_crossover.py` dilutes the Atlas corpus with real FineWeb-Edu
passages and asks the only question the explanation depends on: **are both hop
chunks still in the top-k?** No LLM, no generation — pure retrieval.

| corpus | both hops retrieved | per question |
|---:|---:|---|
| 30 (the eval corpus) | **6/7** | `BBBBB.B` |
| 10,000 | 6/7 | `BBBBBxB` |
| 100,000 | **5/7** | `BxBBBxB` |
| 300,000 | 5/7 | `BxBBBxB` |

**Co-retrieval falls from 6/7 to 5/7 across a 10,000× increase in corpus size.**
The direction the README predicted is real. The *reasoning* was not.

This project used to argue from the fraction of the corpus you retrieve — "top-4
of 30 is 13%, top-4 of 13.6 M is 0.00003%". That framing is wrong, and the
measurement shows why: **the fraction is not the variable.** Atlas chunks stay
retrievable against 300,000 web passages because their vocabulary
(`shed mode`, `TLM-101`, `atlas-vision`) does not compete with generic web text.
Adding documents that cannot win the ranking changes nothing, however many
there are.

Look at *which* two questions break:

- **m2** — bridging fact is "**41%**" (the incident causing the most pages)
- **m6** — bridging fact is "**18 ms**" (the model with that latency budget)

Both hops hinge on a **generic numeric token** that hundreds of thousands of web
pages also contain. The named-entity hops (`shed mode`, `Code-128`,
`Rotterdam`) survive every corpus size tested. And raising k to 8 does **not**
recover them — the displaced chunks are not sitting at rank 5–8, they are gone.

**Corrected claim:** the crossover is driven by **distractor competition**, not
corpus size. Size matters only because more documents mean more chances that one
of them competes. A hop that hangs on a distinctive entity survives scale; a hop
that hangs on "41%" or "18 ms" does not — and *that* is when the graph earns its
extraction cost.

**Honest limit of this probe:** FineWeb-Edu is generic web text, the friendliest
possible distractor. 300,000 pages of *other ops runbooks* would compete far
harder and degrade co-retrieval sooner. So these numbers are a **lower bound** on
degradation, not an upper one.

### Why graph scored 29% — every failure is entity linking

```
m2 "incident type that causes the most pages"   -> NOT_FOUND (no entity linked)
m3 "data class with the longest retention"      -> NOT_FOUND (no entity linked)
m4 "model that reads Code-128 barcodes"         -> NOT_FOUND (no entity linked)
```

The questions describe entities by **property or superlative** rather than by
name — and lexical entity linking has nothing to anchor on. "Shed mode" links;
"the incident type that causes the most pages" does not, because *most pages* is
a comparison, not a node. GraphRAG's precondition is that questions name
entities; these deliberately don't. (Where an entity *was* named — m1 "shed
mode", m7 "barcode retries" — the graph passed.)

### Why agentic scored 29% — the 7B is the ceiling

The loop itself worked: searches were sensible, evidence accumulated, citations
verified. The failures were *reasoning*:

- m2: picked the **22%** incident over the **41%** one — deterministic misreading
  of a comparison, reproduced across runs
- m3: answered "the operator audit log can access..." — inverted the relation
- m4–m6: burned the 4-step budget searching without committing to an answer

Agentic RAG moves the bottleneck from retrieval to **per-step reasoning
reliability**, and a 7B model pays that toll on every hop. A frontier model
would score differently; that is exactly the experiment the harness makes
one-flag cheap (`AgenticRag(model=...)`).

### The cost column

Agentic burned **3–4× the LLM calls and ~2.5× the latency** to score worse.
A comparison that omits cost always flatters the expensive paradigm; this one
doesn't.

---

## 📁 What's in this project

```
08_rag_paradigms/
├── graphrag/
│   ├── extract.py        LLM triple extraction over the corpus (cached)
│   ├── graph.py          networkx graph, deterministic entity linking, k-hop walks
│   ├── answer.py         walked facts + source chunks -> project-01 generator
│   └── graph_data.json   190 extracted triples (cache)
├── agentic/
│   └── agent.py          typed action loop: search / answer / abstain, bounded steps
├── evaluate_paradigms.py the three-way experiment
└── paradigm_results.json
```

Both paradigms **reuse project 01's generator wholesale** — numbered blocks,
mandatory `[n]` citations, mechanical verification, NOT_FOUND abstention.
Holding generation constant is what makes this a retrieval experiment rather
than a prompt-engineering one.

---

## 🚀 How to run it

```powershell
..\activate.ps1                          # and Ollama running with qwen2.5:7b

python -m graphrag.extract               # build the graph (~40 s, cached)
python -m graphrag.answer "what is the response time for the severity level that shed mode is classified as?"
python -m agentic.agent  "what action fixes the incident type that causes the most pages?"

python evaluate_paradigms.py             # the full three-way comparison
python evaluate_paradigms.py --subset multihop --verbose
```

---

## 🔬 GraphRAG: what was built, and the three bugs on the way

**Pipeline:** one LLM extraction call per chunk → `(subject, relation, object)`
triples with source chunk ids → networkx MultiDiGraph → deterministic n-gram
entity linking → k-hop neighborhood walk → facts + source chunks to the
generator.

**Bug 1 — extraction misses are permanent and silent.** The first prompt dropped
the parenthetical in "SEV3 ... (e.g. shed mode)", so the crucial
`shed mode → sev3` edge never existed and the flagship two-hop question returned
NOT_FOUND. Fixed by teaching the prompt that named concepts and parenthetical
examples are entities. *Extraction quality is system quality* — a missed
relation is unretrievable forever, with no error.

**Bug 2 — entity resolution decides connectivity.** The extractor oscillated
between `shed mode` and `shed_mode` across chunks: two disconnected nodes, the
edge present but unreachable. One normalisation line fixed it; a pinned test
(`test_underscore_and_space_variants_unify`) keeps it fixed. The same class of
bug ate possessives ("shed mode's severity" → "shed modes") — also found by
test, also pinned.

**Design choice worth defending:** entity linking is **lexical, not LLM-based**.
It is testable (an LLM linker makes every retrieval test flaky), it costs ~1 ms,
and its failure mode is honest — a miss returns nothing rather than
hallucinating a plausible entity. The 29% multihop score *is* that trade-off,
measured rather than hidden.

**The economics:** extraction cost 30 LLM calls for 30 chunks (~40 s). At
project-07 scale that is ~50 days of local inference. GraphRAG is for corpora
that are small, stable and entity-dense — an ops handbook, not a web crawl.

---

## 🤖 Agentic RAG: what was built

A **typed action loop**, not a framework: each turn the model sees the question
plus all evidence so far and emits exactly one JSON action —
`search` / `answer` / `abstain`. Constraints, each a lesson from this repo:

- **Bounded steps** (default 4): an agent that can loop can loop forever; the
  budget converts "stuck" into "abstain with a trajectory"
- **Stable evidence numbering + mechanical citation verification**: agency
  changes retrieval, never grounding discipline
- **Repeat-query detection**: the classic small-model spin, surfaced in the
  transcript instead of silently burning budget
- **The trajectory is a first-class artifact** — every eval row records what the
  agent searched and why it stopped:

```
Q: what action fixes the incident type that causes the most pages?
  1. search: incident type most pages  (+3 blocks)
  2. search: vision gantry incident type  (+1 blocks)
  3. search: incident type causes most pages  (+1 blocks)
  4. answer: ...
  -> answered
```

---

## 📚 The "three types of RAG" — interview cheat sheet

Two taxonomies get called "the three types"; know both.

**The survey taxonomy (Gao et al.):**

| type | definition | where this repo demonstrates it |
|---|---|---|
| **Naive** | chunk → embed → top-k → prompt | project 01 minus rerank/hybrid |
| **Advanced** | + pre-retrieval (rewriting, routing, HyDE) and post-retrieval (rerank) | project 01 full + project 07 `techniques/query_side.py` |
| **Modular** | swappable modules, per-query routing, iterative flows | 07's router; 05's MCP tools; this project's paradigm switch |

**The paradigm taxonomy (this project):** vector / graph / agentic — measured
head-to-head above.

The senior answer ties them together: *Naive→Advanced→Modular is about how much
machinery surrounds one retrieval; vector/graph/agentic is about who drives
retrieval. And the measured lesson here is that paradigm choice follows corpus
scale — at 30 chunks the machinery is overhead, at 13.6 M it's the difference
between answering and not.*

---

## ❓ FAQ

**Would the graph win at larger scale?**
Its *preconditions* improve: at scale, co-retrieval of both hops stops being
free, which is the failure graphs fix. But its costs scale too — extraction is
per-chunk LLM work, and entity linking still requires questions that name
entities. The honest claim: the crossover exists, and this corpus is on the
wrong side of it.

**Would a better model fix agentic?**
Most of its failures, probably — m2/m3 are reasoning errors a frontier model
rarely makes. The harness takes `model=` for exactly that experiment. The
cost column (3–4× calls) stays regardless.

**Why not LLM-based entity linking for the graph?**
It would fix the superlative-phrasing misses (m2–m4) at the price of an LLM
call per query, non-determinism in every test, and a new hallucination surface.
The right production answer is hybrid: lexical first, LLM fallback — and that
fallback's win-rate is measurable with this same harness.

**Is 7 multihop questions enough to conclude anything?**
Enough to demonstrate the *mechanisms* (co-retrieval, linking misses, reasoning
errors) — each failure was characterised individually, not averaged away. Not
enough for percentage claims to two digits. The set is small because each chain
was hand-verified against the corpus; scaling it is mechanical.

---

## Related projects

- **[01_rag_local](../01_rag_local/)** — the vector baseline all of this runs on
- **[07_rag_at_scale](../07_rag_at_scale/)** — the scale at which the paradigm
  trade-offs would actually flip
- **[05_mcp_server](../05_mcp_server/)** — the tool substrate an agentic RAG
  would use in production
