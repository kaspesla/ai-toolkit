"""Find the first layer that turns finite inputs into NaN/Inf.

Enabled by run.py when NAN_HUNT=1 is set:
    NAN_HUNT=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python run.py config/your_config.yaml

Checks the output of every nn.Module forward (VAE encode during caching,
text encoder, transformer, LoRA layers). Reports the first module whose inputs
were all finite but whose output was not, then stops the run. Slow; use only
for a step or two.
"""
import os
import sys

import torch
from torch.nn.modules.module import register_module_forward_hook

_names = {}


def _tensors(x):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, (list, tuple)):
        for i in x:
            yield from _tensors(i)
    elif isinstance(x, dict):
        for i in x.values():
            yield from _tensors(i)


def _bad(x):
    for t in _tensors(x):
        if torch.is_floating_point(t) and t.numel() and not torch.isfinite(t).all():
            return t
    return None


def _name(module):
    if not _names:
        # lazily index names from every module tree seen so far
        import gc
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.nn.Module):
                    for n, m in obj.named_modules():
                        if n and id(m) not in _names:
                            _names[id(m)] = f"{type(obj).__name__}.{n}"
            except Exception:
                pass
    return _names.get(id(module), "?")


def _desc(x):
    out = []
    for t in _tensors(x):
        if torch.is_floating_point(t) and t.numel():
            f = t.detach().float()
            fin = f[torch.isfinite(f)]
            rng = f"[{fin.min().item():.3g}, {fin.max().item():.3g}]" if fin.numel() else "none finite"
            out.append(f"{tuple(t.shape)} {t.dtype} {t.device} nan={torch.isnan(f).sum().item()} "
                       f"inf={torch.isinf(f).sum().item()} finite range {rng}")
    return "\n      ".join(out) or "(no float tensors)"


def _hook(module, args, output):
    if _bad(output) is None:
        return
    print("\n" + "=" * 70, file=sys.stderr)
    if _bad(args) is not None:
        # no earlier module was flagged (we exit on the first report), so the
        # NaN was created by plain tensor code, or was already in the input data
        print(f"NAN_HUNT: NaN/Inf was ALREADY IN THE INPUT to {type(module).__name__} "
              f"({_name(module)}) - it came from non-module code or the source data",
              file=sys.stderr)
    else:
        print(f"NAN_HUNT: first NaN/Inf produced by {type(module).__name__} ({_name(module)})",
              file=sys.stderr)
    print(f"  grad enabled: {torch.is_grad_enabled()}", file=sys.stderr)
    print(f"  inputs:\n      {_desc(args)}", file=sys.stderr)
    print(f"  output:\n      {_desc(output)}", file=sys.stderr)
    params = [(n, p) for n, p in module.named_parameters(recurse=False)]
    for n, p in params:
        print(f"  param {n}: {tuple(p.shape)} {p.dtype} finite={torch.isfinite(p).all().item()}",
              file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    os._exit(1)


def install():
    register_module_forward_hook(_hook)
    print("NAN_HUNT enabled: will stop at the first layer that produces NaN/Inf")
