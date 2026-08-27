"""
Phase 0 gate: does the training stack actually work on THIS GPU?

Blackwell (RTX 50-series) is compute capability 12.0 / sm_120. A torch wheel
built only up to sm_90 imports fine, reports cuda.is_available() == True, and
then dies at the first real kernel with "no kernel image is available for
execution on the device". So importing is not evidence -- every check below
runs an actual kernel.

    python pipeline/check_env.py

Designed to fit in a few hundred MB of VRAM so it can run while Ollama holds
the rest of the card. Pass --skip-gpu to run only the import/version checks.
"""
import argparse, sys, traceback

PASS, FAIL, WARN = "  PASS", "  FAIL", "  WARN"
results = []


def check(name, fn, fatal=False):
    try:
        detail = fn()
        results.append((PASS, name, detail or ""))
        print(f"{PASS}  {name}" + (f"  --  {detail}" if detail else ""))
        return True
    except Exception as e:
        msg = f"{type(e).__name__}: {e}".replace("\n", " ")[:180]
        results.append((FAIL, name, msg))
        print(f"{FAIL}  {name}\n         {msg}")
        if fatal:
            print("\nFatal -- later checks depend on this. Stopping.")
            sys.exit(1)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gpu", action="store_true")
    a = ap.parse_args()

    print("=" * 70)
    print("TRAINING STACK CHECK")
    print("=" * 70)

    import torch
    print(f"\ntorch {torch.__version__} | python {sys.version.split()[0]}")

    # ---------------------------------------------------------------- versions
    def _cuda_build():
        v = torch.version.cuda
        if not v:
            raise RuntimeError("CPU-only torch build -- reinstall from the "
                               "cu128 index URL")
        return f"built against CUDA {v}"
    check("torch has a CUDA build", _cuda_build, fatal=True)

    def _avail():
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        return torch.cuda.get_device_name(0)
    check("CUDA device visible", _avail, fatal=True)

    cap = torch.cuda.get_device_capability(0)
    sm = f"sm_{cap[0]}{cap[1]}"

    def _arch():
        arches = torch.cuda.get_arch_list()
        if sm not in arches:
            raise RuntimeError(
                f"this GPU is {sm} but the wheel was built for {arches}. "
                "Kernels will fail at runtime.")
        return f"{sm} present in {len(arches)} compiled arches"
    # THE critical check for Blackwell. Everything else can pass while this
    # fails, right up until the first matmul.
    check(f"wheel contains kernels for {sm}", _arch)

    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = torch.cuda.mem_get_info()[0] / 1e9
    print(f"\n  VRAM {free:.1f} GB free of {total:.1f} GB "
          f"({'tight -- Ollama likely resident' if free < 2 else 'ok'})")

    if a.skip_gpu:
        return

    # ------------------------------------------------------------ real kernels
    def _matmul():
        x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
        y = (x @ x).sum().item()
        if y != y:                                    # NaN
            raise RuntimeError("matmul produced NaN")
        return "fp16 512x512 matmul ok"
    check("fp16 kernel executes", _matmul, fatal=True)

    def _bf16():
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 unsupported (pre-Ampere)")
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        (x @ x).sum().item()
        return "bf16 supported and executing"
    check("bf16 kernel executes", _bf16)

    def _sdpa():
        q = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float16)
        torch.nn.functional.scaled_dot_product_attention(q, q, q)
        return "fused attention path ok"
    check("scaled_dot_product_attention", _sdpa)

    # ------------------------------------------------------------ bitsandbytes
    def _bnb():
        import bitsandbytes as bnb
        from bitsandbytes.nn import Linear4bit
        layer = Linear4bit(256, 256, bias=False, compute_dtype=torch.float16,
                           quant_type="nf4").cuda()
        out = layer(torch.randn(4, 256, device="cuda", dtype=torch.float16))
        if out.isnan().any():
            raise RuntimeError("Linear4bit returned NaN")
        return f"bnb {bnb.__version__} NF4 forward ok {tuple(out.shape)}"
    # This is the one that decides whether QLoRA is possible at all.
    bnb_ok = check("bitsandbytes NF4 4-bit forward", _bnb)

    # -------------------------------------------------------------- peft/LoRA
    def _peft():
        import peft
        from peft import LoraConfig, get_peft_model
        import torch.nn as nn

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(128, 128)
                self.v_proj = nn.Linear(128, 128)

            def forward(self, x):
                return self.v_proj(self.q_proj(x))

        m = get_peft_model(Tiny(), LoraConfig(r=8, lora_alpha=16,
                                              target_modules=["q_proj", "v_proj"]))
        m = m.cuda()
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        allp = sum(p.numel() for p in m.parameters())
        loss = m(torch.randn(2, 128, device="cuda")).sum()
        loss.backward()
        grads = [n for n, p in m.named_parameters()
                 if p.requires_grad and p.grad is not None]
        if not grads:
            raise RuntimeError("no LoRA gradients after backward")
        if any("lora" not in g for g in grads):
            raise RuntimeError(f"non-LoRA params got gradients: {grads}")
        return (f"peft {peft.__version__} | {trainable:,}/{allp:,} trainable "
                f"({100*trainable/allp:.2f}%) | grads flow to LoRA only")
    check("peft LoRA fwd+bwd, base frozen", _peft)

    def _trl():
        import trl, transformers
        return f"trl {trl.__version__}, transformers {transformers.__version__}"
    check("trl / transformers import", _trl)

    # ------------------------------------------------------------------ verdict
    print("\n" + "=" * 70)
    fails = [r for r in results if r[0] == FAIL]
    if not fails:
        print("ALL CHECKS PASSED -- LoRA and QLoRA are both viable on this GPU.")
    else:
        print(f"{len(fails)} CHECK(S) FAILED:")
        for _, n, m in fails:
            print(f"  - {n}: {m}")
        if not bnb_ok:
            print("\n  bitsandbytes is the QLoRA dependency. Without it you can\n"
                  "  still run plain LoRA in bf16 -- a 4B model fits in 16 GB.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
