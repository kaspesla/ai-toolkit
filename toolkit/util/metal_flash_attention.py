"""Metal Flash Attention on Apple Silicon via Universal Metal Flash Attention
(UMFA, https://github.com/bghira/universal-metal-flash-attention).

Enable with `train.attention_backend: metal_flash`. This swaps
torch.nn.functional.scaled_dot_product_attention for UMFA's entry point, so it
covers diffusers models (their "native" backend calls the torch function at
call time) and the extension models that call F.scaled_dot_product_attention
directly.

UMFA does its own routing: MPS fp32/fp16/bf16 attention runs on the Metal
kernels, forward and backward, and anything it can't run (dropout, a real
attention mask while training, odd shapes or dtypes) goes to PyTorch's native
SDPA. Non-MPS tensors never reach UMFA.

Needs macOS 15+ and the UMFA `metal_sdpa_extension` built into this venv:
    cd universal-metal-flash-attention && swift build -c release
    cd examples/pytorch-custom-op-ffi && python setup.py install
"""
import torch
import torch.nn.functional as F

_original_sdpa = None
_ext = None


def _metal_flash_sdpa(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    if query.device.type != "mps":
        return _original_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    return _ext.metal_scaled_dot_product_attention(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
    )


def install_metal_flash_attention():
    """Route scaled_dot_product_attention through UMFA. Idempotent."""
    global _original_sdpa, _ext
    if _original_sdpa is not None:
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError("attention_backend 'metal_flash' requires an MPS device (Apple Silicon)")
    try:
        import metal_sdpa_extension
    except ImportError as e:
        raise ImportError(
            "attention_backend 'metal_flash' requires the UMFA metal_sdpa_extension. Build it with "
            "`swift build -c release` in universal-metal-flash-attention, then "
            "`python setup.py install` in examples/pytorch-custom-op-ffi using this venv's python."
        ) from e
    if not metal_sdpa_extension.is_metal_available():
        raise RuntimeError("UMFA reports Metal is not supported on this device")

    _ext = metal_sdpa_extension
    _original_sdpa = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _metal_flash_sdpa
    print("Using Metal Flash Attention (UMFA) for scaled_dot_product_attention")


def uninstall_metal_flash_attention():
    global _original_sdpa
    if _original_sdpa is None:
        return
    F.scaled_dot_product_attention = _original_sdpa
    _original_sdpa = None
