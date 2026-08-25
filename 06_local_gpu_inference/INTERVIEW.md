# 🎤 Interview notes — GPU inference, quantisation, serving

---

## The 60-second project pitch

> "A benchmark harness for local LLM inference: five precision variants, memory,
> time-to-first-token, decode speed, a batching sweep, and — the part most
> benchmarks skip — a perplexity column so you can see what the memory savings
> cost you.
>
> The interesting result is that it **refuted the assumption I started with**. I'd
> written the standard 'decoding is memory-bandwidth bound' claim into the
> script. Then I ran a 0.5B and a 1.5B: three times the weights, essentially the
> same decode speed. At that scale with HuggingFace `generate()` you're
> overhead-bound, not bandwidth-bound — Python loop and kernel launches, not
> bytes moved. Which is exactly why int4 bought me 2.2× less memory and zero
> speedup, and why int8 was five times *slower*. Knowing which regime you're in
> is the whole game, because it decides whether quantising or batching or
> changing runtime is the actual win."

---

## Core questions

### "Walk me through the memory a model uses at inference."

Four buckets:

1. **Weights** — `params × bytes_per_param`. 7B in fp16 ≈ 14 GB. This is the one
   quantisation attacks.
2. **KV cache** — `2 × layers × kv_heads × head_dim × seq_len × batch × bytes`.
   Linear in context length **and** batch size. At long context this becomes the
   dominant term, not the weights.
3. **Activations** — transient, scales with batch × sequence length.
4. **Fragmentation / allocator overhead** — real, usually 5–10%.

The mistake people make is quoting bucket 1 and being surprised by the OOM.
That's why my benchmark reports **weights and peak side by side** — the gap
between the bars is buckets 2–4, and quantisation doesn't shrink it.

At training you add optimizer state (AdamW is 2 extra fp32 copies per trainable
param) and stored activations for the backward pass — which is exactly why QLoRA
didn't help in project 02: at 0.5B the weights weren't the problem.

### "Explain the quantisation options."

- **fp16 / bf16** — 2 bytes. bf16 has fp32's exponent range with less mantissa;
  preferred for training because no gradient scaler is needed. Essentially free
  quality-wise.
- **int8 / LLM.int8()** — decomposes each matmul, routes outlier features through
  a separate fp16 path, recombines. Preserves quality well (+2.3% perplexity in
  my run) but the machinery is expensive: **5× slower** on both models I tested.
  It's for *fitting* large models, not for speed.
- **NF4 (QLoRA's 4-bit)** — a datatype whose levels sit at the quantiles of a
  normal distribution. NN weights are roughly normal, so NF4 loses less than
  uniform int4 at the same width. Plus double quantisation (quantise the
  quantisation constants, ~0.4 bits/param saved).
- **GPTQ / AWQ** — calibration-based post-training quantisation. Uses a small
  dataset to decide *which* weights matter, so they generally beat naive 4-bit at
  the same size. Slower to produce, better to serve.
- **GGUF (llama.cpp)** — a family of k-quants (`Q4_K_M` and friends) tuned for
  CPU and mixed CPU/GPU. What Ollama serves.

The number I'd want them to hear: **int4 cost +9.2% perplexity on the 0.5B and
+15.4% on the 1.5B.** Quantisation is a trade. Anyone who quotes only the memory
saving hasn't measured the other half.

### "Why is decoding slow, and what would you do about it?"

Prefill and decode are different problems:

- **Prefill** processes the whole prompt in parallel. Compute bound, uses the
  GPU well. This is your TTFT.
- **Decode** generates one token at a time, each depending on the last. It's
  inherently sequential.

The textbook answer for decode is *memory-bandwidth bound* — one pass over all
weights per token, ~2 FLOPs per weight. **My measurements say that's not what's
happening at 0.5–1.5B with HF `generate()`**: 3× the weights gave the same speed,
so per-token overhead (Python loop, kernel launches, small sequential matmuls) is
the binding constraint, not bytes moved.

Fixes, matched to the regime:
- **Overhead-bound** (small model, eager runtime): batching, CUDA graphs,
  `torch.compile`, or a compiled runtime like vLLM/TensorRT-LLM. Quantisation
  will *not* help.
- **Bandwidth-bound** (large model): quantisation genuinely speeds you up,
  because there are fewer bytes to read per token.
- **Either way:** speculative decoding — a small draft model proposes k tokens,
  the big model verifies them in one parallel pass. Turns sequential decode into
  batched verification.

### "Why is batching so effective?"

Because the weights are read once per forward pass **regardless of batch size**.
Batch 32 doesn't read 32× the weights, it reads them once and does 32× the work
with them.

My numbers: batch 32 gave **32.5× the total throughput** for 0.18 GB more memory,
with per-sequence speed essentially unchanged. In the overhead-bound regime the
effect is even stronger than the bandwidth argument predicts, because you're
amortising per-token overhead across 32 requests.

The limits are the KV cache (grows linearly with batch) and latency: at some
batch size individual requests start waiting. That's the fundamental
**throughput vs latency** trade, and it's why interactive serving and offline
batch jobs land on very different batch sizes.

**Continuous batching** (vLLM) is the production answer: instead of assembling a
fixed batch and waiting for the slowest to finish, sequences join and leave the
running batch every step. Keeps the GPU full without making anyone wait for a
batch to form.

### "How do you benchmark properly?"

The mistakes are more instructive than the recipe:

- **Not synchronising CUDA.** It's asynchronous. Time without
  `torch.cuda.synchronize()` and you're measuring kernel *launches* — numbers
  ~10× too good. This is the single most common error in published LLM benchmarks.
- **Blending prefill into decode.** One "tokens/sec" number hides both. Report
  TTFT and decode rate separately; they answer different user questions ("did it
  respond?" vs "is it fast?").
- **No warmup.** First call pays context setup, autotuning, allocator warmup.
- **Not resetting peak memory between variants.** Peak leaks forward and every
  result after the first is wrong.
- **Reporting memory without quality.** That's marketing.
- **Sampling on.** Makes runs incomparable. `do_sample=False`.

And the one I'd own up to, because I got caught by it: **single timed runs**.

I re-ran this benchmark months later. Memory and perplexity reproduced *to the
digit* — they're deterministic. Decode speed did not. bf16 on the 0.5B had been
44.8 tok/s; the re-run gave **61.0**, and three isolated runs gave 45.1, 40.7,
43.5. So I measured the spread properly: **36.6% across five runs in one
process.**

That is larger than every difference between fp32, fp16, bf16 and int4 put
together — and it flipped the *sign* of one of my published comparisons. I had
written "int4 buys zero speed, same decode rate as bf16"; a re-run made int4 look
19% slower, and isolated runs made it 8–15% faster. The honest statement is that
the difference is below the noise floor: median 49.1 vs 47.9, 0.98×.

So the benchmark now takes `--repeats` (default 3) and reports the median with
the spread, and the README marks each claim robust or unsupported. What survives
is what should: int8 being 4–5× slower, batching at ~30×, and the headline
overhead-bound finding — all of them order-of-magnitude effects that a 30% noise
floor cannot touch. What doesn't survive is "fp32 is the fastest variant", which
I should never have stated from one sample.

**A number without an error bar invites over-reading, and I over-read my own.**

### "How would you serve a 7B model to 100 concurrent users?"

- **Runtime:** vLLM or TensorRT-LLM, not HF transformers. Continuous batching and
  paged attention are the two features that matter, and paged attention is what
  stops KV-cache fragmentation from capping your concurrency.
- **Precision:** AWQ or GPTQ 4-bit. At 7B you *are* bandwidth-bound, so
  quantisation buys speed as well as memory — unlike what I measured at 1.5B.
- **Capacity planning:** the binding constraint is usually KV cache, not weights.
  Work out `2 × layers × kv_heads × head_dim × max_seq × max_concurrent × bytes`
  and size from there. GQA (which Qwen and Llama 3 use) shrinks this a lot by
  sharing KV heads across query heads.
- **Autoscaling:** cold start is brutal — loading 14 GB of weights is tens of
  seconds. Keep a warm pool; don't scale to zero on interactive traffic.
- **What I'd measure:** p50/p99 TTFT, p50/p99 inter-token latency, tokens/sec/GPU,
  and cost per million tokens. Then set the batch size from the p99 target rather
  than from peak throughput.

---

## Numbers worth having memorised

- **Weights:** `params × bytes`. 7B fp16 = 14 GB, int4 ≈ 3.5 GB.
- **KV cache:** `2 × layers × kv_heads × head_dim × seq × batch × bytes`.
- **Arithmetic intensity of decode:** ~2 FLOPs per weight byte read — which is
  *why* the bandwidth-bound claim is made, and why it only bites once per-token
  overhead is small relative to the read.
- **Prefill vs decode:** prefill is O(n²) attention over the prompt but fully
  parallel; decode is O(n) per token but strictly sequential.

---

## Questions to ask *them*

- "What runtime are you serving on, and did you benchmark it against the
  alternatives or inherit it?"
- "Is your bottleneck KV cache or weights? That changes the whole optimisation
  path."
- "Do you track quality after quantising, or just memory and speed?"
- "What's your p99 TTFT target, and what batch size did that force you to?"

---

## Related projects

- **[02_lora_text](../02_lora_text/)** — the training-side version of the same
  lesson: QLoRA lost at 0.5B because the memory was activations, not weights
- **[01_rag_local](../01_rag_local/)** — the generation stage measured here in
  a real pipeline
