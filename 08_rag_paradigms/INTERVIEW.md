# 🎤 Interview notes — RAG paradigms (vector / graph / agentic)

---

## The 60-second pitch

> "I implemented GraphRAG and Agentic RAG next to my vector pipeline and ran all
> three over the same corpus against the same labelled questions — including
> seven hand-verified multi-hop questions, the case graphs and agents are
> supposedly built for.
>
> The fancy paradigms lost. Vector RAG scored 100% on multihop; graph and
> agentic scored 29% each. And the *why* is the real finding: at 30 chunks,
> top-4 retrieval pulls both hop chunks into context at once, so the generator
> does the join in-context — multi-hop is only structurally hard when the hops
> can't co-retrieve, which is a property of corpus **scale**, not question
> shape. Graph's failures were all entity linking on superlative phrasings;
> agentic's were per-step reasoning errors in the 7B driver plus budget
> exhaustion, at 3–4× the LLM calls.
>
> So my answer to 'which RAG should we use' is: measure where your corpus sits
> relative to the co-retrieval crossover before buying machinery."

---

## Core questions

### "What are the three types of RAG?"

Disambiguate first — two taxonomies share the name:

**Survey taxonomy** (Gao et al.): **Naive** (chunk→embed→top-k→prompt),
**Advanced** (add pre-retrieval: query rewriting, routing, HyDE; and
post-retrieval: reranking, compression), **Modular** (swappable components,
per-query strategy, iterative flows).

**Paradigm taxonomy**: **vector** (similarity-driven), **graph**
(entity/edge-driven), **agentic** (LLM-driven, iterative).

They compose — an agentic system usually calls an advanced vector retriever as
its tool. My repo demonstrates all of both: projects 01/07 are Advanced, 07's
router is Modular, and project 08 measures the three paradigms head-to-head.

### "When does GraphRAG beat vector RAG?"

Three preconditions, all measured in my experiment:

1. **The hops cannot co-retrieve.** At 30 chunks, top-4 grabs both hop chunks
   and vector wins outright (100% vs my graph's 29%). The graph's advantage only
   exists past the corpus size where similarity search over the *question*
   misses the second hop.
2. **Questions name entities.** My graph passed exactly the multihop questions
   that named an entity ("shed mode") and failed exactly the ones that described
   entities by property ("the incident type that causes the most pages") —
   there is no node called *most pages*.
3. **The corpus affords extraction.** One LLM call per chunk at index time: 40 s
   for my 30 chunks, ~50 days at project 07's 13.6 M. Graphs suit small, stable,
   entity-dense corpora — handbooks, not crawls.

And a fourth people forget: someone must own extraction quality, because a
missed relation is unretrievable forever with no error.

### "Walk me through your GraphRAG's failure modes — you measured 29%."

Gladly, because each failure has a mechanism, not a mystery:

- **Extraction miss** (fixed): the extractor dropped "(e.g. shed mode)" from the
  severity table, so the flagship two-hop chain was broken at hop 1. Fixed by
  prompting that parentheticals and named concepts are entities. The lesson:
  *extraction quality is system quality*.
- **Entity resolution** (fixed, pinned by test): `shed mode` vs `shed_mode`
  became two disconnected nodes — the edge existed and was unreachable.
  Possessives had the same class of bug: "shed mode's severity" normalised to
  "shed modes", matching no node. Both are one-line normalisations whose absence
  produces silent NOT_FOUNDs.

  There's a postscript I'd tell on myself, because it's the more useful story.
  The possessive fix carried a code comment saying "found by test" and the
  README said "also pinned" — and **no such test existed**. I found it by
  auditing whether my own documentation was true, not by reviewing code. The
  test exists now (`test_possessive_links_to_the_base_entity`), and the general
  lesson went into my suite: *"pinned by test" is itself a claim that needs
  verifying*, so a doc test now fails if any bug in my ledger cites a test that
  isn't there.
- **Linking on descriptions** (unfixed, deliberate): my linker is lexical, so
  "the data class with the longest retention" anchors nothing. An LLM linker
  would fix it at the cost of per-query latency, flaky tests and a new
  hallucination surface. Production answer: lexical first, LLM fallback,
  measure the fallback's win rate.

### "Why did your agent score 29% on the questions it was designed for?"

Because agentic RAG moves the bottleneck from retrieval to **per-step reasoning
reliability**, and my driver is a 7B.

Concretely, from the trajectories (which the harness records for every
question): it picked the 22%-of-pages incident over the 41% one —
deterministically, at temperature 0; it inverted a relation ("the audit log can
access..."); and on three questions it burned its 4-step budget searching
without committing. The loop mechanics were fine — searches sensible, citations
mechanically verified, budget converting "stuck" into a clean abstain.

Two design points I'd defend: **bounded steps** (an agent that can loop can loop
forever; the budget makes failure legible), and **the trajectory as a
first-class artifact** — an agent you cannot audit is an agent you cannot debug.

And the cost: 3–4× the LLM calls, ~2.5× the latency, for a worse score here.
Any agentic pitch that omits the cost column is marketing.

### "So is the standard multi-hop narrative wrong?"

No — it's **conditional**, and I can tell you exactly on what, because I tested
it rather than argued it.

I used to say "scale-conditional" and reason from the fraction retrieved: top-4
of 30 chunks is 13% of the corpus, top-4 of 13.6 M is 0.00003%. **That reasoning
was wrong**, and measuring it is what showed me.

I diluted the corpus with real FineWeb-Edu passages and measured co-retrieval —
are both hop chunks still in top-k? No LLM, pure retrieval:

| corpus | both hops |
|---:|---:|
| 30 | 6/7 |
| 10,000 | 6/7 |
| 100,000 | 5/7 |
| 300,000 | 5/7 |

**A 10,000× increase in corpus size cost me one question.** The fraction is not
the variable. Atlas chunks stay retrievable against 300,000 web passages because
`shed mode` and `TLM-101` don't compete with generic web text — adding documents
that can't win the ranking changes nothing, however many you add.

What actually predicts failure is *which token the hop hangs on*. The two
questions that broke bridge on "**41%**" and "**18 ms**" — generic numerics that
hundreds of thousands of pages also contain. Every named-entity hop survived
every size I tested, and raising k to 8 didn't recover the lost ones; they're
gone, not at rank 5.

So the corrected claim: **the crossover is driven by distractor competition, not
corpus size.** Size matters only because more documents mean more chances one of
them competes. That also sharpens when a graph earns its extraction cost — not
"when you get big", but "when your joins hinge on non-distinctive tokens".

And the limit I'd volunteer: FineWeb is the friendliest possible distractor.
300,000 pages of *other ops runbooks* would compete much harder, so my numbers
are a lower bound on degradation.

### "How did you keep the comparison fair?"

- Same corpus, same questions, same gold labels (deterministic `must_contain`,
  no judge)
- **Same generator and grounding discipline** — all three paradigms feed
  project 01's numbered-block prompt with mechanical citation verification, so
  it's a retrieval experiment, not a prompt-engineering one
- Cost reported alongside accuracy (LLM calls and latency per question)
- Failures characterised individually, not averaged — 7 multihop questions is
  enough to demonstrate mechanisms, not enough for two-digit percentages, and
  the README says so

---

## Questions to ask *them*

- "Roughly how many chunks is your corpus — and have you measured whether your
  hard questions' evidence actually fails to co-retrieve?"
- "If you use a graph: who owns extraction quality, and how do you detect a
  relation that was never extracted?"
- "If you use agents: what's your step budget, and what does a trajectory audit
  look like when an answer is wrong?"
- "What does an added paradigm have to beat — do you have a tuned single-shot
  baseline on the same eval set?"

That last one is this whole project in one question.

---

## Related projects

- **[01_rag_local](../01_rag_local/)** — the baseline that won
- **[tests/INTERVIEW.md](../tests/INTERVIEW.md)** — testing a non-deterministic
  system, and the bug ledger these two entity bugs sit in
- **[07_rag_at_scale](../07_rag_at_scale/)** — the scale where it would stop winning
