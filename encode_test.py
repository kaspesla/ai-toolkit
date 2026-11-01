"""Find an MPS setup that encodes correctly, by comparing against the CPU.

Run from the ai-toolkit folder with the venv active:
    python encode_test.py /path/to/one/of/your/images.png
    (add --vae /path/to/minimax_h3_video_vae_fp16.safetensors if not found)

Encodes a 512x512 version of the image once on the CPU (float32, the
reference) and then on MPS with each combination of precision and the
contiguous-memory fix. A match is cosine similarity ~0.99+.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from PIL import Image

from toolkit.paths import MODELS_PATH
import extensions_built_in.diffusion_models.minimax_h3.src.vae as vae_mod
from extensions_built_in.diffusion_models.minimax_h3.src.vae import MiniMaxH3VideoVAE

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--vae", default=os.path.join(MODELS_PATH, "vae", "minimax_h3_video_vae_fp16.safetensors"))
a = ap.parse_args()
if not os.path.isfile(a.vae):
    sys.exit(f"VAE not found at {a.vae} - pass --vae /path/to/minimax_h3_video_vae_fp16.safetensors")

img = Image.open(a.image).convert("RGB").resize((a.size, a.size), Image.Resampling.LANCZOS)
px = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 127.5 - 1.0  # (3,H,W) [-1,1]
px = px[None, :, None]  # (1,3,1,H,W)

vae = MiniMaxH3VideoVAE.load_model(a.vae, dtype=torch.float32).eval()


def encode(device, dtype, contiguous):
    vae_mod.FORCE_CONTIGUOUS = contiguous
    vae.to(device, dtype)
    t0 = time.time()
    with torch.no_grad():
        z = vae.encode(px.to(device), sample=False)
    if device == "mps":
        torch.mps.synchronize()
    return z.float().cpu(), time.time() - t0


print("CPU float32 reference ...", flush=True)
ref, secs = encode("cpu", torch.float32, True)
print(f"  done in {secs:.1f}s  std={ref.std():.3f}")

for dtype in (torch.bfloat16, torch.float32):
    for contiguous in (False, True):
        name = f"MPS {str(dtype).split('.')[-1]:8s} contiguous={'on ' if contiguous else 'off'}"
        try:
            z, secs = encode("mps", dtype, contiguous)
        except Exception as e:
            print(f"{name}: ERROR {type(e).__name__}: {e}")
            continue
        if not torch.isfinite(z).all():
            print(f"{name}: NaN/Inf  ({secs:.1f}s)")
            continue
        cos = torch.nn.functional.cosine_similarity(z.flatten(), ref.flatten(), dim=0).item()
        verdict = "OK" if cos > 0.99 else "WRONG"
        print(f"{name}: cosine={cos:.4f}  max diff={(z - ref).abs().max():.3f}  {secs:.1f}s  {verdict}")
