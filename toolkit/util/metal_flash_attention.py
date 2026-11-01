"""Metal Flash Attention on Apple Silicon via Universal Metal Flash Attention
(UMFA, https://github.com/bghira/universal-metal-flash-attention).

Enable with `train.attention_backend: metal_flash`. Importing UMFA's
`metal_sdpa_extension` registers its kernel for aten::scaled_dot_product_attention
on the MPS dispatch key, so from then on every MPS SDPA call in the process
(diffusers' "native" backend, extension models calling
F.scaled_dot_product_attention, references bound at import time) runs on UMFA.
CPU/CUDA calls are untouched. This can't be undone without restarting the process.

UMFA does its own routing: MPS fp32/fp16/bf16 attention runs on the Metal
kernels, forward and backward, and anything it can't run (dropout, a real
attention mask while training, odd shapes or dtypes) goes to PyTorch's native
SDPA.

Needs macOS 15+ and the extension built into this venv:
    cd universal-metal-flash-attention && swift build -c release
    cd examples/pytorch-custom-op-ffi && python -m pip install --no-build-isolation .
"""
import torch

_ext = None


def install_metal_flash_attention():
    """Route MPS scaled_dot_product_attention through UMFA. Idempotent."""
    global _ext
    if _ext is not None:
        return _ext
    if not torch.backends.mps.is_available():
        raise RuntimeError("attention_backend 'metal_flash' requires an MPS device (Apple Silicon)")
    try:
        import metal_sdpa_extension
    except ImportError as e:
        raise ImportError(
            "attention_backend 'metal_flash' requires the UMFA metal_sdpa_extension. Build it with "
            "`swift build -c release` in universal-metal-flash-attention, then "
            "`python -m pip install --no-build-isolation .` in examples/pytorch-custom-op-ffi "
            "using this venv's python."
        ) from e
    if not metal_sdpa_extension.is_metal_available():
        raise RuntimeError("UMFA reports Metal is not supported on this device")

    _ext = metal_sdpa_extension
    print("Using Metal Flash Attention (UMFA) for scaled_dot_product_attention")
    return _ext
