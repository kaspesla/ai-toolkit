"""Check that attention_backend 'metal_flash' works on this Mac before a real run.

Compares UMFA forward + gradients against PyTorch's MPS SDPA on Wan- and
Minimax-H3-like shapes, times a fwd+bwd step for each, and prints UMFA's
dispatch counters so you can see the Metal kernels ran (fp32_autograd) rather
than the PyTorch fallback.

    python testing/check_metal_flash_attention.py [--seq 4096] [--dtype bf16]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from toolkit.util.metal_flash_attention import install_metal_flash_attention, uninstall_metal_flash_attention

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def step(q, k, v, grad_out, **kwargs):
    for t in (q, k, v):
        t.grad = None
    out = F.scaled_dot_product_attention(q, k, v, **kwargs)
    out.backward(grad_out)
    torch.mps.synchronize()
    return out.detach(), [t.grad.detach() for t in (q, k, v)]


def timed(q, k, v, grad_out, iters, **kwargs):
    step(q, k, v, grad_out, **kwargs)  # warmup / kernel compile
    start = time.perf_counter()
    for _ in range(iters):
        step(q, k, v, grad_out, **kwargs)
    return (time.perf_counter() - start) / iters * 1000


def max_rel_err(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()
    dtype = DTYPES[args.dtype]

    # (name, batch, heads, head_dim, sdpa kwargs)
    cases = [
        ("wan2.1 1.3b self-attn", 1, 12, 128, {}),
        ("wan 14b self-attn", 1, 40, 128, {}),
        ("minimax h3, no padding", 1, 24, 128, {}),
        ("causal", 1, 16, 64, {"is_causal": True}),
    ]

    import metal_sdpa_extension as ext

    for name, b, h, d, kwargs in cases:
        torch.manual_seed(0)
        q, k, v = (torch.randn(b, h, args.seq, d, device="mps", dtype=dtype, requires_grad=True) for _ in range(3))
        grad_out = torch.randn(b, h, args.seq, d, device="mps", dtype=dtype)

        ref_out, ref_grads = step(q, k, v, grad_out, **kwargs)
        ref_ms = timed(q, k, v, grad_out, args.iters, **kwargs)

        install_metal_flash_attention()
        ext.reset_dispatch_stats()
        out, grads = step(q, k, v, grad_out, **kwargs)
        stats = ext.get_dispatch_stats()
        mfa_ms = timed(q, k, v, grad_out, args.iters, **kwargs)
        uninstall_metal_flash_attention()

        errs = [max_rel_err(out, ref_out)] + [max_rel_err(g, r) for g, r in zip(grads, ref_grads)]
        print(f"\n{name}: B={b} H={h} S={args.seq} D={d} {args.dtype}")
        print("  rel err  out={:.2e} dq={:.2e} dk={:.2e} dv={:.2e}".format(*errs))
        print(f"  fwd+bwd  torch {ref_ms:.1f} ms | umfa {mfa_ms:.1f} ms ({ref_ms / mfa_ms:.2f}x)")
        print(f"  umfa path: fp32_autograd={stats['fp32_autograd']} pytorch_fallback={stats['pytorch_fallback']}")


if __name__ == "__main__":
    main()
