# Instructor notes

How the pipeline is put together, for whoever maintains it.

## Pinned versions
The notebook reproduces the setup used for the equine heart splat (Colab, Python 3.12):

| Component | Version | Notes |
|---|---|---|
| gsplat | 1.5.3 (PyPI) + `v1.5.3` tag of the repo for `examples/` | CUDA kernels JIT-compile on first use (a few minutes) |
| pycolmap | `rmbrualla/pycolmap@cc7ea4b` | gsplat's COLMAP loader. Patched by `scripts/patch_pycolmap.py` for NumPy 2 (`np.uint64(-1)`) |
| fused-ssim | `rahul-goel/fused-ssim@328dc98` | built from source on Colab |
| nerfview | `@4538024` | same pin as gsplat 1.5.3 `examples/requirements.txt` |
| COLMAP | Ubuntu apt package (CPU-only) | `run_colmap.py` detects the `SiftExtraction`/`FeatureExtraction` option rename across COLMAP versions |

To upgrade gsplat, change `GSPLAT_VERSION` in `scripts/colab_setup.sh` and compare its `examples/requirements.txt` with the pip list in that script.

## Colab quirks the setup works around
- Colab's Hugging Face `datasets` package shadows `gsplat/examples/datasets`, so setup uninstalls it and adds an `__init__.py`.
- The gsplat examples pin `numpy<2`. Downgrading NumPy on Colab breaks other packages, so setup keeps NumPy 2 and patches pycolmap instead.
- Training runs with `--disable_viewer`. Otherwise the trainer starts a viser server after training and the cell never finishes.
- Training length uses `--steps_scaler` instead of `--max-steps`, so evaluation, checkpoints, densification and the trajectory video all scale with it.

## Differences from the original equine heart notebook
- The original began from a zip with COLMAP already run. This version starts from video (`extract_frames.py` → `run_colmap.py`, following the layout of Inria's `convert.py`: OPENCV camera, single camera, undistorted to `images/` + `sparse/0`).
- Frames are capped at 1600 px on the long side by default. The heart was trained on roughly 4300×3800 images, which took about 1 h on a large GPU and would not fit a T4 comfortably.
- `.ply` export uses the same manual writer as the original notebook (`export_ply.py`) for both 3DGS and 2DGS checkpoints.
- The mesh step follows the original (statistical outlier removal → normals → Poisson depth 9 → trim lowest 2% density). The normal radius is now relative to the specimen size (4% of the bounding-box diagonal) instead of a fixed 0.03, so it no longer depends on SuperSplat's scale.

## Running locally (not supported for students)
The scripts in `scripts/` are plain Python and also run on a Linux or Windows machine with an NVIDIA GPU, COLMAP and the same packages installed. On Windows, also run `patch_pycolmap.py` (it fixes the 8-byte `L` struct issue).
