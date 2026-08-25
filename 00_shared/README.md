# 00_shared — environment checks and small helpers

Two files, no project depends on them to run.

## `check_env.py`

Run this first, and again whenever something breaks.

```powershell
python 00_shared\check_env.py
```

It reports, in one screen:

- Python version (3.10–3.12 is the torch-supported band)
- PyTorch version and whether it's a CUDA build
- GPU name, VRAM, and **compute capability** — plus an explicit check that your
  torch CUDA build is new enough for that architecture. This is the failure that
  wastes the most time: an sm_120 (RTX 50-series) card with a pre-12.8 wheel
  installs fine and dies at the first kernel launch.
- bfloat16 support, and current free VRAM
- Which optional libraries are present, and which project each one is for
- Ollama: reachable, and which models are pulled. Only `qwen2.5:7b` is
  required — the 0.5B used by projects 02 and 06 is a HuggingFace model
  loaded through transformers, not an Ollama pull

`[FAIL]` blocks a project. `[WARN]` only matters for the project that needs it.

## `gpu.py`

**Nothing imports this, and that is worth saying plainly** rather than letting
the section imply otherwise. The projects call `torch.cuda` directly; this file
is kept as a reference implementation because the two notes below are the
substance, and both are things people get wrong in interviews. If you want it,
it is a copy-paste away — which is the honest description of its status:

```python
from gpu import pick_device, pick_dtype, vram, reset_peak, timer

with timer("generation") as t:
    model.generate(...)
print(vram())        # alloc / reserved / peak / free
```

Two things in here are worth reading rather than just using:

- **`pick_dtype`** prefers bf16 over fp16 on modern NVIDIA. Same speed on Ampere+,
  but the wider exponent range means no gradient scaler and no silent inf/NaN.
- **`timer`** calls `torch.cuda.synchronize()` on both sides. Without it you are
  timing kernel *launches*, not kernel *execution* — the classic way to publish
  benchmark numbers that are an order of magnitude too good.
