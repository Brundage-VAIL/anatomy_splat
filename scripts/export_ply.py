"""Convert a gsplat training checkpoint into a standard Gaussian-splat .ply file.

The .ply opens in SuperSplat (https://superspl.at/editor) and most other splat viewers.
Works for both 3DGS (simple_trainer.py) and 2DGS (simple_trainer_2dgs.py) checkpoints.

Usage:
    python export_ply.py RESULT_DIR_OR_CKPT --out my_specimen.ply
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch


def find_checkpoint(path):
    if path.is_file():
        return path
    ckpts = list((path / "ckpts").glob("ckpt_*.pt")) or list(path.glob("ckpt_*.pt"))
    if not ckpts:
        raise SystemExit(f"No ckpt_*.pt files found in {path}")
    # Newest = highest step number (ckpt_29999_rank0.pt or ckpt_29999.pt)
    return max(ckpts, key=lambda p: int(re.search(r"ckpt_(\d+)", p.name).group(1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="training result folder or a ckpt_*.pt file")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ckpt_path = find_checkpoint(args.source)
    print(f"Reading {ckpt_path}")
    s = torch.load(ckpt_path, map_location="cpu", weights_only=False)["splats"]

    def arr(k):
        return s[k].detach().cpu().float().numpy()

    means = arr("means")                          # (N,3)
    scales = arr("scales")                        # (N,3) log-space
    quats = arr("quats")                          # (N,4) w,x,y,z
    opac = arr("opacities").reshape(-1, 1)        # (N,1) logit-space
    sh0 = arr("sh0")                              # (N,1,3)
    shN = arr("shN")                              # (N,K,3)

    n = means.shape[0]
    f_dc = sh0.reshape(n, 3)
    f_rest = shN.transpose(0, 2, 1).reshape(n, -1)  # channel-major, as the 3DGS .ply format expects
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    normals = np.zeros((n, 3), dtype=np.float32)

    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.concatenate([means, normals, f_dc, f_rest, opac, scales, quats], axis=1).astype("<f4")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {n}\n".encode())
        for nm in names:
            f.write(f"property float {nm}\n".encode())
        f.write(b"end_header\n")
        f.write(data.tobytes())
    print(f"Wrote {n:,} splats to {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
