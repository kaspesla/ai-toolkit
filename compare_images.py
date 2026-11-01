"""List every cached latent as GOOD/BAD next to its source image's details.

Usage:  python compare_images.py /path/to/your/dataset
"""
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

root = Path(sys.argv[1])
images = {p.stem: p for p in root.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")}
rows = []
for f in sorted((root / "_latent_cache").glob("*.safetensors")):
    t = load_file(str(f)).get("latent")
    if t is None:
        continue
    stem = next((s for s in images if f.name.startswith(s + "_")), None)
    status = "BAD " if not torch.isfinite(t).all() else "good"
    info = "image not found"
    if stem:
        im = Image.open(images[stem])
        info = f"{im.size[0]}x{im.size[1]} mode={im.mode} bytes={images[stem].stat().st_size}"
    rows.append((status, f"latent {tuple(t.shape[-2:])}", stem or f.name, info))

for r in sorted(rows):
    print("  ".join(r))
print(f"\n{sum(r[0] == 'BAD ' for r in rows)} bad / {len(rows)} total")
