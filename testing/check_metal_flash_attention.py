"""Check that attention_backend 'metal_flash' works on this Mac before a real run.

For Wan- and Minimax-H3-like shapes, runs a fwd+bwd step with PyTorch's MPS
SDPA and with UMFA, scores both against an fp32 reference (explicit
softmax(QK^T)V, which no SDPA override can intercept), times both, and prints
UMFA's dispatch counters so you can see the Metal kernels ran (fp32_autograd)
rather than the PyTorch fallback.

A model-shaped training case runs a few fwd+bwd steps the way the models call
SDPA (q/k/v chunked from a fused projection that is still queued on the GPU,
gradients through matmuls) and checks the grads stay finite and correct. It
catches ordering races between PyTorch's MPS queue and UMFA's.

Importing UMFA replaces MPS SDPA for the whole process, so every PyTorch
measurement is taken before the extension is imported.

    python testing/check_metal_flash_attention.py [--seq 4096] [--dtype bf16]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# (name, batch, heads, head_dim, is_causal)
CASES = [
    ("wan2.1 1.3b self-attn", 1, 12, 128, False),
    ("wan 14b self-attn", 1, 40, 128, False),
    ("minimax h3, no padding", 1, 24, 128, False),
    ("causal", 1, 16, 64, True),
]

# Inference (no grad) takes UMFA's in-stream path, which reads strided inputs
# directly. "bshd" builds (B, S, H, D) and transposes to BHSD the way Minimax H3
# and diffusers call SDPA, so the kernel sees non-contiguous q/k/v.
# "qkv" chunks one fused projection like Minimax H3's qkv_proj, so k and v also
# start at a non-zero offset into the shared buffer.
# (name, layout, padding mask)
NOGRAD_CASES = [
    ("inference, contiguous", "bhsd", False),
    ("inference, transposed (as models call it)", "bshd", False),
    ("inference, fused qkv chunks (Minimax H3)", "qkv", False),
    ("inference, transposed + padding mask", "bshd", True),
]


def make_nograd_inputs(seq, dtype, layout, masked, b=1, h=24, d=128):
    torch.manual_seed(0)
    if layout == "qkv":
        qkv = torch.randn(b, seq, 3 * h * d, device="mps", dtype=dtype)
        q, k, v = (t.view(b, seq, h, d).transpose(1, 2) for t in qkv.chunk(3, dim=-1))
    else:
        shape = (b, h, seq, d) if layout == "bhsd" else (b, seq, h, d)
        q, k, v = (torch.randn(shape, device="mps", dtype=dtype) for _ in range(3))
        if layout == "bshd":
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    mask = None
    if masked:
        # (B, 1, 1, S) bool, True = attend, last 1/8 of keys padded out (Minimax H3 style)
        mask = torch.ones(b, 1, 1, seq, dtype=torch.bool, device="mps")
        mask[..., -seq // 8:] = False
    return q, k, v, mask


# (name, heads, head_dim, is_causal); batch 1
MODEL_CASES = [
    ("training through fused qkv projection (Minimax H3)", 24, 128, False),
]
MODEL_STEPS = 3


def make_model_inputs(seq, dtype, h, d):
    torch.manual_seed(0)
    dim = h * d
    x = torch.randn(1, seq, dim, device="mps", dtype=dtype, requires_grad=True)
    w_qkv = (torch.randn(dim, 3 * dim, device="mps", dtype=dtype) * dim ** -0.5).requires_grad_()
    w_out = (torch.randn(dim, dim, device="mps", dtype=dtype) * dim ** -0.5).requires_grad_()
    return x, w_qkv, w_out


def model_step(attn, x, w_qkv, w_out, h, is_causal):
    """fwd+bwd with no syncs between the projections and attention."""
    for t in (x, w_qkv, w_out):
        t.grad = None
    b, s, _ = x.shape
    q, k, v = (t.view(b, s, h, -1).transpose(1, 2) for t in (x @ w_qkv).chunk(3, dim=-1))
    out = attn(q, k, v, is_causal=is_causal).transpose(1, 2).reshape(b, s, -1) @ w_out
    out.float().square().mean().backward()
    torch.mps.synchronize()
    return [out.detach().float()] + [t.grad.detach().float() for t in (x, w_qkv, w_out)]


def make_inputs(b, h, d, seq, dtype):
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, h, seq, d, device="mps", dtype=dtype, requires_grad=True) for _ in range(3))
    grad_out = torch.randn(b, h, seq, d, device="mps", dtype=dtype)
    return q, k, v, grad_out


def step(attn, q, k, v, grad_out, is_causal):
    for t in (q, k, v):
        t.grad = None
    out = attn(q, k, v, is_causal=is_causal)
    out.backward(grad_out.to(out.dtype))
    torch.mps.synchronize()
    return [out.detach().float()] + [t.grad.detach().float() for t in (q, k, v)]


def timed_ms(attn, q, k, v, grad_out, is_causal, iters):
    step(attn, q, k, v, grad_out, is_causal)  # warmup / kernel compile
    start = time.perf_counter()
    for _ in range(iters):
        step(attn, q, k, v, grad_out, is_causal)
    return (time.perf_counter() - start) / iters * 1000


def reference_attention(q, k, v, is_causal=False, attn_mask=None):
    """fp32 softmax(QK^T * scale)V, plain matmuls only."""
    q, k, v = q.float(), k.float(), v.float()
    scores = (q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5
    if is_causal:
        s = q.shape[-2]
        mask = torch.ones(s, s, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf"))
    return scores.softmax(-1) @ v


def rel_errs(got, ref):
    return [((g - r).abs().max() / r.abs().max().clamp_min(1e-6)).item() for g, r in zip(got, ref)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--inference-only", action="store_true", help="skip the fwd+bwd training cases")
    args = parser.parse_args()
    dtype = DTYPES[args.dtype]
    cases = [] if args.inference_only else CASES

    assert "metal_sdpa_extension" not in sys.modules, "UMFA must not be imported before the torch baseline"
    nograd_results = {}
    with torch.no_grad():
        for name, layout, masked in NOGRAD_CASES:
            q, k, v, mask = make_nograd_inputs(args.seq, dtype, layout, masked)
            ref = reference_attention(q, k, v, attn_mask=mask)
            torch_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).float()
            nograd_results[name] = (ref, rel_errs([torch_out], [ref])[0])
            del q, k, v, mask, torch_out
            torch.mps.empty_cache()

    results = {}
    for name, b, h, d, causal in cases:
        q, k, v, grad_out = make_inputs(b, h, d, args.seq, dtype)
        ref = step(reference_attention, q, k, v, grad_out, causal)
        torch_res = step(F.scaled_dot_product_attention, q, k, v, grad_out, causal)
        torch_ms = timed_ms(F.scaled_dot_product_attention, q, k, v, grad_out, causal, args.iters)
        results[name] = (ref, rel_errs(torch_res, ref), torch_ms)
        del q, k, v, grad_out, torch_res
        torch.mps.empty_cache()

    model_results = {}
    for name, h, d, causal in [] if args.inference_only else MODEL_CASES:
        x, w_qkv, w_out = make_model_inputs(args.seq, dtype, h, d)
        ref = model_step(reference_attention, x, w_qkv, w_out, h, causal)
        torch_res = model_step(F.scaled_dot_product_attention, x, w_qkv, w_out, h, causal)
        model_results[name] = (ref, rel_errs(torch_res, ref))
        del x, w_qkv, w_out, torch_res
        torch.mps.empty_cache()

    from toolkit.util.metal_flash_attention import install_metal_flash_attention

    ext = install_metal_flash_attention()
    for name, layout, masked in NOGRAD_CASES:
        ref, torch_err = nograd_results[name]
        q, k, v, mask = make_nograd_inputs(args.seq, dtype, layout, masked)
        ext.reset_dispatch_stats()
        with torch.no_grad():
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        torch.mps.synchronize()
        stats = ext.get_dispatch_stats()
        umfa_err = rel_errs([out.float()], [ref])[0]
        ok = "OK" if umfa_err < max(4 * torch_err, 2e-2) else "WRONG OUTPUT"
        print(f"\n{name}: B=1 H=24 S={args.seq} D=128 {args.dtype}  [{ok}]")
        print(f"  rel err vs fp32  torch {torch_err:.2e} | umfa {umfa_err:.2e}")
        print(
            f"  umfa path: instream={stats['fp32_instream']} direct={stats['fp32_direct']} "
            f"pytorch_fallback={stats['pytorch_fallback']}"
        )
        del q, k, v, mask, out
        torch.mps.empty_cache()

    for name, b, h, d, causal in cases:
        ref, torch_errs, torch_ms = results[name]
        q, k, v, grad_out = make_inputs(b, h, d, args.seq, dtype)
        ext.reset_dispatch_stats()
        umfa_errs = rel_errs(step(F.scaled_dot_product_attention, q, k, v, grad_out, causal), ref)
        stats = ext.get_dispatch_stats()
        umfa_ms = timed_ms(F.scaled_dot_product_attention, q, k, v, grad_out, causal, args.iters)

        print(f"\n{name}: B={b} H={h} S={args.seq} D={d} {args.dtype}{' causal' if causal else ''}")
        print("  rel err vs fp32   out      dq       dk       dv")
        print("    torch        " + " ".join(f"{e:.2e}" for e in torch_errs))
        print("    umfa         " + " ".join(f"{e:.2e}" for e in umfa_errs))
        print(f"  fwd+bwd  torch {torch_ms:.1f} ms | umfa {umfa_ms:.1f} ms ({torch_ms / umfa_ms:.2f}x)")
        print(f"  umfa path: fp32_autograd={stats['fp32_autograd']} pytorch_fallback={stats['pytorch_fallback']}")
        del q, k, v, grad_out
        torch.mps.empty_cache()

    for name, h, d, causal in [] if args.inference_only else MODEL_CASES:
        ref, torch_errs = model_results[name]
        x, w_qkv, w_out = make_model_inputs(args.seq, dtype, h, d)
        ext.reset_dispatch_stats()
        # Worst error over several steps: a race shows up on some steps, not all.
        steps = [rel_errs(model_step(F.scaled_dot_product_attention, x, w_qkv, w_out, h, causal), ref)
                 for _ in range(MODEL_STEPS)]
        umfa_errs = [max(errs) for errs in zip(*steps)]
        stats = ext.get_dispatch_stats()
        bad = any(e != e or e > max(4 * t, 2e-2) for e, t in zip(umfa_errs, torch_errs))
        print(f"\n{name}: B=1 H={h} S={args.seq} D={d} {args.dtype}, worst of {MODEL_STEPS} steps  "
              f"[{'WRONG GRADIENTS' if bad else 'OK'}]")
        print("  rel err vs fp32   out      dx       dw_qkv   dw_out")
        print("    torch        " + " ".join(f"{e:.2e}" for e in torch_errs))
        print("    umfa         " + " ".join(f"{e:.2e}" for e in umfa_errs))
        print(f"  umfa path: fp32_autograd={stats['fp32_autograd']} pytorch_fallback={stats['pytorch_fallback']}")
        del x, w_qkv, w_out
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
