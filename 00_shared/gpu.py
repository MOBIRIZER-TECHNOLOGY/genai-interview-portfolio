"""
Tiny GPU helpers shared by every project. No project depends on this file to
run -- it exists so the VRAM/timing code isn't copy-pasted five times.

    from gpu import pick_device, pick_dtype, vram, timer
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass


def pick_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(device: str | None = None):
    """bf16 on modern NVIDIA, fp16 on older CUDA, fp32 on CPU.

    bf16 is preferred over fp16 for *training*: same speed on Ampere+, but the
    wider exponent range means no gradient-scaler and no silent inf/NaN blowups.
    """
    import torch

    device = device or pick_device()
    if device != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@dataclass
class VramSnapshot:
    allocated_gb: float
    reserved_gb: float
    peak_gb: float
    free_gb: float
    total_gb: float

    def __str__(self) -> str:
        return (
            f"VRAM alloc={self.allocated_gb:.2f}GB reserved={self.reserved_gb:.2f}GB "
            f"peak={self.peak_gb:.2f}GB free={self.free_gb:.2f}GB / {self.total_gb:.1f}GB"
        )


def vram() -> VramSnapshot:
    import torch

    if not torch.cuda.is_available():
        return VramSnapshot(0, 0, 0, 0, 0)
    free, total = torch.cuda.mem_get_info()
    return VramSnapshot(
        allocated_gb=torch.cuda.memory_allocated() / 1024**3,
        reserved_gb=torch.cuda.memory_reserved() / 1024**3,
        peak_gb=torch.cuda.max_memory_allocated() / 1024**3,
        free_gb=free / 1024**3,
        total_gb=total / 1024**3,
    )


def reset_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@contextlib.contextmanager
def timer(label: str = "elapsed"):
    """Wall-clock timer that synchronises CUDA first.

    Without the synchronize() calls you are timing kernel *launches*, not kernel
    *execution* -- a classic way to report benchmark numbers that are 10x too good.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = {"seconds": 0.0}
    try:
        yield result
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result["seconds"] = time.perf_counter() - t0
        print(f"[{label}] {result['seconds']:.3f}s")
