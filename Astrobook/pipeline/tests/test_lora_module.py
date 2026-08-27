"""
Phase 0, written BEFORE the implementation.

These assertions ARE the specification for lora_from_scratch.LoRALinear. The
peft-equivalence test is the real one: if our forward pass matches the reference
implementation bit-for-bit on identical weights, we have understood LoRA rather
than merely produced something that trains.

    python pipeline/tests/test_lora_module.py

Runs on CPU. No GPU needed, no pytest needed (works under pytest too).
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lora_from_scratch import LoRALinear, inject_lora, trainable_parameters

torch.manual_seed(0)
IN, OUT, R, ALPHA = 64, 32, 8, 16


def _base():
    torch.manual_seed(0)
    return nn.Linear(IN, OUT, bias=True)


def test_identity_at_init():
    """B is zero-initialised, so the adapter is an EXACT no-op at step 0.

    This is why attaching LoRA never damages the base model -- training starts
    from the unmodified function and departs from it gradually. If B were
    randomly initialised, step 0 would inject noise into every layer at once.
    """
    base = _base()
    lora = LoRALinear(_base(), r=R, alpha=ALPHA)
    x = torch.randn(4, IN)
    assert torch.equal(lora(x), base(x)), "adapter is not identity at init"


def test_b_is_zero_a_is_not():
    lora = LoRALinear(_base(), r=R, alpha=ALPHA)
    assert torch.count_nonzero(lora.lora_B) == 0, "B must init to zeros"
    assert torch.count_nonzero(lora.lora_A) > 0, "A must init non-zero"


def test_shapes():
    """A is (r x in), B is (out x r) -- the low-rank bottleneck."""
    lora = LoRALinear(_base(), r=R, alpha=ALPHA)
    assert lora.lora_A.shape == (R, IN), lora.lora_A.shape
    assert lora.lora_B.shape == (OUT, R), lora.lora_B.shape


def test_scaling_is_alpha_over_r():
    """delta = (B @ A) x * (alpha / r). Doubling alpha doubles the delta."""
    base = _base()
    x = torch.randn(4, IN)
    outs = []
    for alpha in (ALPHA, ALPHA * 2):
        m = LoRALinear(_base(), r=R, alpha=alpha)
        with torch.no_grad():
            m.lora_B.copy_(torch.randn(OUT, R) * 0.1)
        outs.append(m(x) - base(x))
    assert torch.allclose(outs[1], outs[0] * 2, atol=1e-5), \
        "delta did not scale linearly with alpha"


def test_matches_peft():
    """THE test: identical weights must produce identical outputs to peft."""
    from peft import LoraConfig, get_peft_model

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _base()

        def forward(self, x):
            return self.proj(x)

    ref = get_peft_model(Wrap(), LoraConfig(r=R, lora_alpha=ALPHA,
                                            lora_dropout=0.0,
                                            target_modules=["proj"]))
    # find peft's LoRA layer without depending on its attribute path
    layer = next(m for m in ref.modules() if hasattr(m, "lora_A")
                 and hasattr(m, "lora_B"))
    pa = layer.lora_A["default"].weight
    pb = layer.lora_B["default"].weight
    with torch.no_grad():                       # peft inits B to 0; randomise
        pb.copy_(torch.randn_like(pb) * 0.1)

    mine = LoRALinear(_base(), r=R, alpha=ALPHA)
    with torch.no_grad():
        mine.lora_A.copy_(pa)
        mine.lora_B.copy_(pb)

    x = torch.randn(8, IN)
    ref.eval(); mine.eval()
    a, b = ref(x), mine(x)
    assert torch.allclose(a, b, atol=1e-6), \
        f"max abs diff vs peft: {(a - b).abs().max().item():.3e}"


def test_only_lora_requires_grad():
    """The base weight is frozen. If it were not, this is a full fine-tune."""
    lora = LoRALinear(_base(), r=R, alpha=ALPHA)
    assert not lora.base.weight.requires_grad, "base weight is trainable!"
    if lora.base.bias is not None:
        assert not lora.base.bias.requires_grad, "base bias is trainable!"
    assert lora.lora_A.requires_grad and lora.lora_B.requires_grad


def test_gradients_reach_only_lora():
    lora = LoRALinear(_base(), r=R, alpha=ALPHA)
    with torch.no_grad():
        lora.lora_B.copy_(torch.randn(OUT, R) * 0.1)
    lora(torch.randn(4, IN)).sum().backward()
    assert lora.lora_A.grad is not None and lora.lora_B.grad is not None
    assert lora.base.weight.grad is None, "gradient leaked into the base weight"


def test_merge_matches_unmerged():
    """Merging folds BA*scale into W, so inference costs nothing extra."""
    lora = LoRALinear(_base(), r=R, alpha=ALPHA)
    with torch.no_grad():
        lora.lora_B.copy_(torch.randn(OUT, R) * 0.1)
    x = torch.randn(4, IN)
    lora.eval()
    before = lora(x)
    merged = lora.merged_linear()
    assert torch.allclose(before, merged(x), atol=1e-5), \
        "merged weights changed the function"


def test_inject_and_count():
    """Injection replaces targeted Linears and leaves the rest alone."""
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(IN, IN)
            self.v_proj = nn.Linear(IN, IN)
            self.other = nn.Linear(IN, IN)

        def forward(self, x):
            return self.other(self.v_proj(self.q_proj(x)))

    net = inject_lora(Net(), ["q_proj", "v_proj"], r=R, alpha=ALPHA)
    assert isinstance(net.q_proj, LoRALinear)
    assert isinstance(net.v_proj, LoRALinear)
    assert isinstance(net.other, nn.Linear) and not isinstance(net.other, LoRALinear)

    trn, tot = trainable_parameters(net)
    expected = 2 * (R * IN + IN * R)          # A and B for two layers
    assert trn == expected, f"{trn} trainable, expected {expected}"
    assert trn / tot < 0.15, f"{100*trn/tot:.1f}% trainable -- too high for LoRA"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}\n          {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
