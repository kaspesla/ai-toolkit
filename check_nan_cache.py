"""Scan ai-toolkit cache folders (latents + text embeddings) for NaN/Inf.

Usage:  python check_nan_cache.py /path/to/your/dataset
"""
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

root = Path(sys.argv[1])
files = [p for p in root.rglob("*.safetensors") if "_cache" in str(p.parent)]
if not files:
    print("No cache files found under", root)
bad = 0
for f in files:
    for k, t in load_file(str(f)).items():
        if not torch.is_floating_point(t):
            continue
        n_nan = torch.isnan(t).sum().item()
        n_inf = torch.isinf(t).sum().item()
        if n_nan or n_inf:
            bad += 1
            print(f"BAD  {f.relative_to(root)} [{k}] shape={tuple(t.shape)} nan={n_nan} inf={n_inf}")
print(f"checked {len(files)} files, {bad} bad tensors")
