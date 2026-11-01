"""Find which encoder layer / op computes differently on MPS vs CPU.

Run from the ai-toolkit folder with the venv active:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python layer_diff.py /path/to/image.png [--vae ...]

Part 1 tests the individual tensor ops the encoder uses, on random data.
Part 2 runs the real encoder (untiled, 256x256, float32) on CPU and MPS with
the SAME inputs fed to every layer, and lists the layers whose output differs.
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from toolkit.paths import MODELS_PATH
from extensions_built_in.diffusion_models.minimax_h3.src.vae import MiniMaxH3VideoVAE

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--vae", default=os.path.join(MODELS_PATH, "vae", "minimax_h3_video_vae_fp16.safetensors"))
a = ap.parse_args()


def cos(x, y):
    x, y = x.float().cpu().flatten(), y.float().cpu().flatten()
    if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
        return float("nan")
    return F.cosine_similarity(x, y, dim=0).item()


def report(name, c, extra=""):
    flag = "OK   " if c > 0.9999 else "WRONG"
    print(f"  {flag} {name:55s} cosine={c:.6f} {extra}", flush=True)


# ---------------------------------------------------------------- part 1
print("Part 1: individual ops (random data, float32)")
torch.manual_seed(0)
x = torch.randn(1, 64, 3, 40, 40)
w3 = torch.randn(64, 64, 3, 3, 3) * 0.05
b = torch.randn(64) * 0.05
ops = {
    "reflect pad 5D (1,1,1,1,0,0)": lambda t: F.pad(t, (1, 1, 1, 1, 0, 0), mode="reflect"),
    "reflect pad 5D asym (0,1,0,1,0,0)": lambda t: F.pad(t, (0, 1, 0, 1, 0, 0), mode="reflect"),
    "zero pad time (0,0,0,0,2,0)": lambda t: F.pad(t, (0, 0, 0, 0, 2, 0)),
    "conv3d k3 stride 1": lambda t: F.conv3d(t, w3.to(t.device), b.to(t.device)),
    "conv3d k3 stride (1,2,2)": lambda t: F.conv3d(t, w3.to(t.device), b.to(t.device), stride=(1, 2, 2)),
    "conv3d k3 stride (2,2,2)": lambda t: F.conv3d(t, w3.to(t.device), b.to(t.device), stride=(2, 2, 2)),
    "group_norm 32 (frames folded)": lambda t: F.group_norm(
        t.permute(0, 2, 1, 3, 4).reshape(-1, 64, 40, 40), 32).reshape(1, 3, 64, 40, 40).permute(0, 2, 1, 3, 4),
    "silu": F.silu,
}
for name, fn in ops.items():
    try:
        report(name, cos(fn(x), fn(x.to("mps"))))
    except Exception as e:
        print(f"  ERROR {name}: {type(e).__name__}: {e}")

# ---------------------------------------------------------------- part 2
if not os.path.isfile(a.vae):
    sys.exit(f"VAE not found at {a.vae} - pass --vae /path/to/minimax_h3_video_vae_fp16.safetensors")
print(f"\nPart 2: real encoder layers ({a.size}x{a.size}, untiled, float32)")
img = Image.open(a.image).convert("RGB").resize((a.size, a.size), Image.Resampling.LANCZOS)
px = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 127.5 - 1.0
px = px[None, :, None]

print('  loading VAE ...', flush=True)
vae = MiniMaxH3VideoVAE.load_model(a.vae, dtype=torch.float32).eval()
print('  running CPU pass ...', flush=True)
vae.use_tiling = False

# record each leaf-ish module's CPU input and output
records = []  # (name, module, cpu_args, cpu_out)
def rec_hook(name):
    def h(m, args, out):
        records.append((name, m, [t.clone() if torch.is_tensor(t) else t for t in args], out.clone()))
    return h

handles = []
targets = [(n, m) for n, m in vae.encoder.named_modules()
           if n and type(m).__name__ in ("CausalConv3d", "FrameGroupNorm", "Downsample3d", "ResnetBlock3d")]
for n, m in targets:
    handles.append(m.register_forward_hook(rec_hook(n)))
with torch.no_grad():
    vae.encode(px, sample=False)
for h in handles:
    h.remove()
print(f'  recorded {len(records)} layers; replaying each on MPS ...', flush=True)

vae.to("mps")
bad = 0
with torch.no_grad():
    for i, (name, m, args, out) in enumerate(records):
        if i % 10 == 0:
            print(f'  ... layer {i}/{len(records)}', flush=True)
        mps_out = m(*[t.to("mps") if torch.is_tensor(t) else t for t in args])
        c = cos(out, mps_out)
        kind = type(m).__name__
        shape = tuple(args[0].shape)
        if c <= 0.9999:
            bad += 1
            extra = f"{kind} in={shape}"
            if kind == "CausalConv3d":
                extra += f" k={tuple(m.kernel_size)} stride={tuple(m.stride)} pad={m.spatial_pad}"
            report(name, c, extra)
print(f"\n{bad} of {len(records)} layers differ (same inputs, so each WRONG line is itself broken)")
print("Torch:", torch.__version__)
