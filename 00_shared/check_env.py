"""
Environment sanity check for every project in this repo.

Run this FIRST. It tells you, in one screen, whether your machine is ready:
GPU visible, CUDA build correct, VRAM budget, Ollama reachable, and which
optional libraries are installed.

    python 00_shared/check_env.py
"""

import importlib
import importlib.metadata as md
import platform
import shutil
import subprocess
import sys

OK, WARN, BAD = "[ OK ]", "[WARN]", "[FAIL]"


def line(status: str, label: str, detail: str = "") -> None:
    print(f"{status} {label:<28} {detail}")


def check_python() -> None:
    v = sys.version_info
    good = (3, 10) <= (v.major, v.minor) < (3, 13)
    line(OK if good else WARN, "Python", f"{platform.python_version()}  ({sys.executable})")
    if not good:
        print("       -> torch wheels target 3.10-3.12. Use the uv venv (see SETUP.md).")


def check_torch() -> None:
    try:
        import torch
    except ImportError:
        line(BAD, "PyTorch", "not installed -> see SETUP.md")
        return

    line(OK, "PyTorch", torch.__version__)

    if not torch.cuda.is_available():
        line(BAD, "CUDA", "torch.cuda.is_available() is False (CPU-only wheel?)")
        return

    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    cap = torch.cuda.get_device_capability(0)
    line(OK, "GPU", f"{name}  |  {total:.1f} GB  |  sm_{cap[0]}{cap[1]}")
    line(OK, "CUDA (torch build)", torch.version.cuda)

    # Blackwell (sm_120) needs a cu128+ build; older wheels silently fall over.
    if cap[0] >= 12 and int(str(torch.version.cuda).split(".")[0]) < 12:
        line(BAD, "CUDA/arch match", "sm_120 GPU with a pre-12.8 torch build")

    # bf16 is the preferred training dtype on Ampere+ (no loss-scaling needed).
    line(OK if torch.cuda.is_bf16_supported() else WARN, "bfloat16", str(torch.cuda.is_bf16_supported()))

    free, _ = torch.cuda.mem_get_info()
    line(OK, "VRAM free right now", f"{free / 1024**3:.1f} GB")


def check_libs() -> None:
    wanted = [
        ("transformers", "LLM + Whisper models"),
        ("peft", "LoRA adapters"),
        ("datasets", "training data loading"),
        ("accelerate", "training loop / device placement"),
        ("diffusers", "Stable Diffusion (project 03)"),
        ("sentence_transformers", "embeddings + reranker (project 01)"),
        ("faiss", "vector index (project 01)"),
        ("bitsandbytes", "4-bit quantization (projects 02, 06)"),
        ("mcp", "Model Context Protocol SDK (project 05)"),
        ("soundfile", "audio I/O (project 04)"),
    ]
    print("\n-- optional libraries " + "-" * 44)
    for mod, why in wanted:
        try:
            importlib.import_module(mod)
            try:
                ver = md.version({"faiss": "faiss-cpu", "sentence_transformers": "sentence-transformers"}.get(mod, mod))
            except Exception:
                ver = "?"
            line(OK, mod, f"{ver:<12} {why}")
        except Exception:
            line(WARN, mod, f"{'-':<12} {why}  (install from that project's requirements.txt)")


def check_ollama() -> None:
    print("\n-- ollama " + "-" * 55)
    if shutil.which("ollama") is None:
        line(WARN, "ollama", "not on PATH -> https://ollama.com/download")
        return
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=20)
    except Exception as exc:
        line(BAD, "ollama", f"found but not responding: {exc}")
        return

    models = [ln.split()[0] for ln in out.stdout.strip().splitlines()[1:] if ln.strip()]
    line(OK, "ollama", f"{len(models)} model(s) pulled")
    for m in models:
        print(f"         - {m}")
    # only qwen2.5:7b is ever called through Ollama. Projects 02 and 06 use
    # Qwen/Qwen2.5-0.5B-Instruct via transformers, which is a different artifact
    # from Ollama's qwen2.5:0.5b -- this used to demand the Ollama one and send
    # people to pull 400 MB that nothing loads.
    for need in ("qwen2.5:7b",):
        if not any(m.startswith(need) for m in models):
            print(f"       -> missing '{need}':  ollama pull {need}")


if __name__ == "__main__":
    print("=" * 74)
    print("  GenAI interview-prep workspace - environment check")
    print("=" * 74)
    check_python()
    check_torch()
    check_libs()
    check_ollama()
    print("\nDone. Anything marked [FAIL] will block a project; [WARN] is per-project.")
