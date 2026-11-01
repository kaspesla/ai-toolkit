"""Decode cached MiniMax-H3 latents back to PNGs, to see whether the cache
holds real images or finite-but-garbage values.

Run from the ai-toolkit folder with the venv active:
    python decode_cache.py /path/to/your/dataset            # first 4 files, CPU
    python decode_cache.py /path/to/your/dataset -n 8 --device mps

Writes <name>_decoded.png into a _decode_check folder inside the dataset.
CPU is slow (minutes per image) but avoids MPS entirely, so it's the
trustworthy check.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

import torch
from PIL import Image
from safetensors.torch import load_file

from toolkit.paths import MODELS_PATH
from extensions_built_in.diffusion_models.minimax_h3.src.vae import MiniMaxH3VideoVAE

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("-n", type=int, default=4)
ap.add_argument("--device", default="cpu")
ap.add_argument("--vae", default=os.path.join(MODELS_PATH, "vae", "minimax_h3_video_vae_fp16.safetensors"))
a = ap.parse_args()

if not os.path.isfile(a.vae):
    sys.exit(f"VAE not found at {a.vae} - pass --vae /path/to/minimax_h3_video_vae_fp16.safetensors")

vae = MiniMaxH3VideoVAE.load_model(a.vae, dtype=torch.float32).to(a.device).eval()

root = Path(a.dataset)
out_dir = root / "_decode_check"
out_dir.mkdir(exist_ok=True)
files = sorted((root / "_latent_cache").glob("*.safetensors"))[: a.n]
for f in files:
    z = load_file(str(f))["latent"].float()
    print(f"{f.name}: shape={tuple(z.shape)} mean={z.mean():.3f} std={z.std():.3f} "
          f"min={z.min():.2f} max={z.max():.2f} finite={torch.isfinite(z).all().item()}")
    if not torch.isfinite(z).all():
        continue
    with torch.no_grad():
        px = vae.decode(z.unsqueeze(0).to(a.device), autocast_fp16=False)  # (1,3,T,H,W) in [-1,1]
    img = ((px[0, :, 0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    out = out_dir / (f.name.split("__")[0] + "_decoded.png")
    Image.fromarray(img).save(out)
    print(f"  -> {out}")
