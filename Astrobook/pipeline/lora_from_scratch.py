"""
LoRA from scratch -- the whole mechanism, in about sixty lines.

    python pipeline/lora_from_scratch.py          # demo + parameter accounting
    python pipeline/tests/test_lora_module.py     # equivalence vs peft

THE IDEA
--------
Fine-tuning updates W to W + dW. dW has the same shape as W, so storing it costs
as much as the model itself. LoRA's claim is that the useful part of dW is
LOW RANK -- it can be factored as B @ A where A is (r x in) and B is (out x r),
with r tiny (8, 16, 32) next to the layer's dimensions.

    full :  y = (W + dW) x                 dW is (out x in)      -- huge
    LoRA :  y =  W x + (B @ A) x * (a/r)   A,B are thin          -- ~0.5%

W is frozen; only A and B receive gradients. Two consequences follow, and both
are asserted in the test file:

  1. B is initialised to ZERO, so B@A = 0 and the adapter is an exact identity
     at step 0. Training departs from the unmodified base model gradually
     instead of injecting noise into every layer at once.

  2. Because the update is a separate additive term, it can be folded back into
     W afterwards (`merged_linear`). Inference then costs exactly what the base
     model cost -- LoRA adds no latency once merged.

The a/r scaling makes a chosen at one rank behave comparably at another, so you
can change r without re-tuning the learning rate. Convention is alpha = 2r.

QLoRA is this exact module, unchanged, with `base` held in 4-bit NF4 instead of
bf16. That is the entire difference -- see 04_train_hf.py.
"""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be >= 1")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Freeze the base. Without this you are doing a full fine-tune with
        # extra steps -- test_only_lora_requires_grad guards it.
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # peft uses Kaiming-uniform(a=sqrt(5)) on A and zeros on B. Match it
        # exactly, or test_matches_peft is comparing two different functions.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        #        frozen path              trainable low-rank path
        return self.base(x) + (
            self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        ) * self.scaling

    @torch.no_grad()
    def delta_weight(self):
        """The effective dW this adapter represents: (out x in)."""
        return (self.lora_B @ self.lora_A) * self.scaling

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        """Fold the adapter into a plain Linear -- zero inference overhead."""
        merged = nn.Linear(self.base.in_features, self.base.out_features,
                           bias=self.base.bias is not None)
        merged.weight.copy_(self.base.weight + self.delta_weight())
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged

    def extra_repr(self):
        return (f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:g}, "
                f"in={self.base.in_features}, out={self.base.out_features}")


def inject_lora(model: nn.Module, target_modules, r=8, alpha=16, dropout=0.0,
                freeze_base=True):
    """Replace every nn.Linear whose attribute name is in target_modules.

    Matches on the LAST path segment, so "q_proj" hits layers.0.self_attn.q_proj
    at every depth -- the same convention peft uses.

    freeze_base freezes the ENTIRE model before injecting. This is not optional
    in practice: wrapping a layer only freezes that layer, so every untargeted
    Linear, LayerNorm and embedding would keep requires_grad=True and quietly
    train alongside the adapters -- a partial full fine-tune wearing a LoRA
    costume. peft's get_peft_model freezes everything for the same reason.
    (test_inject_and_count caught exactly this.)
    """
    if freeze_base:
        for p in model.parameters():
            p.requires_grad_(False)

    targets = set(target_modules)
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(module, child_name,
                        LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
    return model


def trainable_parameters(model: nn.Module):
    """(trainable, total). The ratio is the number that makes LoRA click."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _demo():
    torch.manual_seed(0)

    class Block(nn.Module):
        """A stand-in for one transformer attention+MLP block."""
        def __init__(self, d=768, ff=3072):
            super().__init__()
            self.q_proj = nn.Linear(d, d)
            self.k_proj = nn.Linear(d, d)
            self.v_proj = nn.Linear(d, d)
            self.o_proj = nn.Linear(d, d)
            self.up_proj = nn.Linear(d, ff)
            self.down_proj = nn.Linear(ff, d)

        def forward(self, x):
            a = self.o_proj(self.v_proj(self.k_proj(self.q_proj(x))))
            return self.down_proj(torch.relu(self.up_proj(a)))

    model = Block()
    x = torch.randn(2, 768)
    before = model(x).clone()
    _, total_before = trainable_parameters(model)

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
    for r in (8, 32):
        m = inject_lora(Block(), targets, r=r, alpha=2 * r)
        trn, tot = trainable_parameters(m)
        print(f"  r={r:<3} trainable {trn:>9,} / {tot:>11,}  "
              f"= {100 * trn / tot:5.2f}%   "
              f"(adapter on disk ~{trn * 2 / 1e6:.1f} MB in fp16)")

    model = inject_lora(model, targets, r=32, alpha=64)
    model.eval()
    after = model(x)
    print(f"\n  identity at init: {torch.equal(before, after)}   "
          "<- B=0 makes the adapter an exact no-op")
    print(f"  full fine-tune would train {total_before:,} params; "
          f"LoRA trains {trainable_parameters(model)[0]:,}")

    print("\n  one wrapped layer:")
    print("   ", model.q_proj)


if __name__ == "__main__":
    print("LoRA from scratch -- parameter accounting on a 768-dim block\n")
    _demo()
