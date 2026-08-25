# 🎤 Interview notes — testing an LLM system

---

## The 60-second pitch

> "LLM systems have two failure modes and they need two instruments, so my suite
> has two layers. The deterministic layer — 158 tests, 99% coverage — pins
> everything that must never flake: chunk boundaries, quantisation codecs,
> threading shutdown, entity resolution, citation verification. Every one of
> those tests started life as a real bug in this repo; there's a 14-row ledger
> mapping each bug to the test that keeps it fixed.
>
> The second layer is an LLM judge, because no deterministic test can tell you
> whether an *answer* is any good. That layer found a hallucination that carried
> a **valid** citation and passed every mechanical check — which is precisely
> the class of bug the first layer cannot see.
>
> And I calibrate the judge before trusting it, because a 7B judge is an
> unreliable instrument and I can show you the four cases where it's wrong."

---

## Core questions

### "How do you test a non-deterministic system?"

You separate what must be deterministic from what genuinely isn't, and you make
that boundary explicit rather than throwing up your hands.

Almost all of a RAG system is deterministic: chunking, tokenisation, index
build, fusion arithmetic, citation verification, entity normalisation. That's
where the bugs actually live, and it's ordinary software — test it ordinarily,
with no GPU and no network. My whole deterministic layer runs in 88 seconds
under `-m "not llm and not gpu"`, so CI needs no infrastructure.

The genuinely non-deterministic part is one thing: **the text the model
generates.** For that you need a judge, and a judge is an instrument that needs
calibrating (below).

The mistake I'd call out in a code review: mocking the LLM and calling that a
test of the pipeline. It tests your mock. What earns its place is faking the LLM
*only* at the network boundary and letting everything else run for real — my
graph tests do full extraction with a scripted HTTP response, so chunking,
normalisation, graph building and the walk are all genuinely exercised.

### "How do you know your tests are worth anything?"

I break the code and check they fail. Every regression test here was
**mutation-checked**: reintroduce the original bug, confirm the suite goes red,
record how many tests fire. The chunker bug fires 11. The threading bug fires 9.

I'd volunteer the embarrassing one, because it's the best evidence the habit is
needed. One chunking test asserted:

```python
assert chunks[0].body[-40:] in chunks[0].body     # always true
```

A tautology. It passes for every input in the universe, and it had been sitting
there claiming overlap was tested. Coverage was 99% and that line was *covered* —
covered and worthless. **Coverage tells you a line ran, not that anything was
checked.**

A suite that passes on known-broken code is worse than no suite, because it
manufactures confidence.

### "Walk me through a bug your tests caught that a human wouldn't."

The one I'd pick is the threaded index builder, because the fix changed how I
test races generally.

Completion was signalled by sentinels pushed through **bounded** queues. Under
backpressure a `put` times out, the sentinel is dropped, and the pipeline hangs
forever at 0% CPU with no output. It killed a shard after 3.25 M successfully
embedded chunks.

**My first two attempts to test it both passed on the broken code.** A
merely-slow consumer never keeps the queue full long enough; a fully-gated
consumer blocks the reader so the completion state is never reached at all.
Chasing the timing was the wrong instinct.

The invariant is what matters:

> A completion signal must not travel through a channel that can drop it.

So the test spies on both queues and asserts no sentinel is ever enqueued. It's
deterministic, it cannot flake, and it fires on the old design. **When a race is
hard to reproduce, test the property that makes it impossible rather than the
timing that makes it visible.**

### "You use an LLM as a judge. Why should I trust it?"

You shouldn't, and I can show you why with numbers from my own run:

| question | faithfulness | relevancy | G-Eval citation |
|---|---:|---:|---:|
| Rotterdam rule (correct, cited `[2]`) | 1.00 | **0.00** | **0.00** |
| TLM-330 (correct, cited `[1]`) | **0.50** | 0.67 | 0.90 |
| barcode 0.92 (correct, cited `[1]`) | 1.00 | 1.00 | **0.00** |
| vision frames (**hallucinated**) | **0.33** | 0.25 | 1.00 |

Two failure modes, both verified by hand: it penalised an answer for correctly
*ignoring* retrieval noise, and it read "**not** retained cold" as "retained
cold" — negation. And the citation metric scored 0.00 on answers visibly
containing `[2]` and `[1]` while scoring 1.00 on another — self-contradictory,
so it measures nothing.

So four disciplines:

1. **A calibration test runs first.** It checks the judge separates a known-good
   from a known-bad answer by ≥ 0.5. If that fails, the gate's verdicts are
   meaningless and the *judge* is what needs fixing, not the pipeline.
2. **One metric gates**, faithfulness, at a threshold **derived from
   measurement** — known hallucination 0.33, known-good 0.50–1.00, floor at
   0.40. Not aspiration: 0.7 produced 2 false positives out of 4, and a gate
   that fails correct work gets switched off within a week.
3. **Everything else is advisory** — printed, never failing.
4. **I deleted the G-Eval citation metric.** Citation validity is already
   verified exactly and for free. *An unreliable LLM gate over a mechanically
   checkable property is strictly worse than the mechanical check alone.*

The 0.17 margin between 0.33 and 0.50 is thin, and that's a statement about the
7B judge rather than the pipeline — a bigger judge justifies a stricter gate,
and re-running calibration is how you'd earn it.

### "What did the judge catch that your other tests couldn't?"

This is the answer that justifies the whole layer:

```
answer:  "Vision frames are retained for 14 days hot and indefinitely cold [1]."
corpus:  | Vision frames | 14 days | none | 14 days |
```

Cold storage is **none**. The model invented indefinite retention — and cited
block `[1]`, which genuinely *was* sent, so mechanical citation verification
passed it.

**Mechanical verification confirms a citation points at a real block; only a
judge can tell you the sentence misrepresents that block.** Different failure
mode, different instrument.

Then the important half: once the failure was known, I pinned it
deterministically (`test_no_invented_cold_retention`) and fixed the system
prompt. **A judge is for discovering unknown failures; once a failure is known,
checking it exactly costs nothing and needs no judge.**

### "99% coverage — is that a vanity metric?"

Mostly yes, and I'd rather tell you the honest history than defend the number.

Coverage was **not** an original goal. The suite was regression-targeted — every
test pinned a real bug — and when I first measured, it was **56%**. Three
modules sat at 0% because no bug had happened there *yet*.

The coverage pass was worth doing, but not for the number. It found a real bug
(a paragraph with no blank lines silently truncated past the embedding window),
found a real regression (a rewrite had dropped stall detection from two polling
loops), and deleted 12 statements of dead code — because dead code isn't a
coverage problem, and the fix is removal, not a test.

What coverage cannot tell you is whether an assertion is meaningful. My
tautological test was covered. So I'd give you the number with the caveat
attached: **coverage finds untested code; mutation testing finds untesting
tests.** The second is the one I'd defend.

### "How do you keep documentation from lying?"

You compile it. This is the answer I'd give that most candidates won't have.

My test count went stale in three documents simultaneously — 105 in one, 141 in
another, actually 143 — each correct when written, none re-derived. So I added
`test_doc_drift.py`, which parses the claims out of the Markdown and asserts
them against live collection: counts match what pytest collects, coverage
percentages agree across documents, `N statements M missed P%` actually
satisfies the arithmetic, every test file is documented, and every test the bug
ledger cites **exists**.

That last check earned itself immediately. Two places claimed a fix was "found
by test, also pinned" when no such test existed — and chasing one of them found
the fix was *incomplete*, a live 2×-budget chunking bug in the project
everything else builds on.

**"Pinned by test" is itself a claim, and claims need verifying.** Two of the
fourteen bugs in my ledger were found by auditing documentation, not code.

### "What isn't tested, and why?"

Stated plainly, because a candidate who claims full coverage is telling you
something about their honesty rather than their suite:

- **The OAuth 2.1 server** is verified end to end by a script — PKCE rejection,
  refresh rotation, revocation — but that's a script, not pytest. It should be
  ported.
- **The training loops** (projects 02/03/04) have no tests. Their `evaluate.py`
  harnesses measure output quality but assert nothing. Fine for a portfolio,
  not fine for a system anyone depends on.
- **The graph and agent's LLM-dependent halves** — extraction quality, agent
  reasoning — are measured by an eval harness, not asserted. They're
  non-deterministic by nature; asserting on them would produce a flaky suite
  that people learn to ignore.

That third one is a real design position, not an excuse: **a flaky test is worse
than a missing one, because it trains the team to ignore red.**

---

## Questions to ask *them*

- "What's in your CI for the LLM parts — and does it gate merges, or is it
  advisory?"
- "If a judge model scores an answer badly, how do you tell 'the answer is bad'
  from 'the judge is wrong'?"
- "When you fix a prompt, what stops the next prompt change from reintroducing
  the bug?"
- "Has anyone here mutation-checked the test suite? What fired?"

That last one is a good-natured way to find out whether their tests are load
bearing or decorative.

---

## Related projects

- **[01_rag_local](../01_rag_local/)** — the pipeline most of these tests guard
- **[07_rag_at_scale](../07_rag_at_scale/)** — where the threading and
  quantisation bugs came from
- **[08_rag_paradigms](../08_rag_paradigms/)** — the entity-resolution bugs
