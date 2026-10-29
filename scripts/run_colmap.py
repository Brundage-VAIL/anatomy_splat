"""Run COLMAP structure-from-motion on a folder of frames.

Works out where each frame's camera was, then undistorts the images.
Writes the layout gsplat expects:

    WORK/
      input/         frames from extract_frames.py (read only)
      distorted/     COLMAP database + raw reconstruction
      images/        undistorted images      <- used for training
      sparse/0/      cameras.bin, images.bin, points3D.bin

Usage:
    python run_colmap.py --work WORK [--matcher sequential] [--gpu]
"""

import argparse
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path


def colmap(*args, log):
    cmd = ["colmap", *map(str, args)]
    print("  $ " + " ".join(cmd[:2]) + " ...", flush=True)
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(Path(log).read_text()[-3000:])
        raise SystemExit(f"COLMAP step '{args[0]}' failed (full log: {log})")


def option_prefix(command, new_name, old_name):
    """COLMAP renamed some options across versions (e.g. SiftExtraction -> FeatureExtraction); use whichever exists."""
    help_text = subprocess.run(["colmap", command, "-h"], capture_output=True, text=True).stdout
    return new_name if f"--{new_name}." in help_text else old_name


def count_bin(path):
    # COLMAP .bin files start with a uint64 count of entries.
    with open(path, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True, help="project work folder containing input/")
    ap.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive",
                    help="exhaustive is most reliable; sequential is faster for 300+ frames from one video")
    ap.add_argument("--gpu", action="store_true", help="use GPU SIFT (only if your COLMAP was built with CUDA)")
    args = ap.parse_args()

    if shutil.which("colmap") is None:
        raise SystemExit("colmap is not installed (on Colab, run the setup cell first)")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no display on Colab

    work = args.work
    frames = work / "input"
    n_frames = len(list(frames.glob("*.jpg"))) + len(list(frames.glob("*.png")))
    if n_frames == 0:
        raise SystemExit(f"No frames found in {frames}; run extract_frames.py first")

    distorted = work / "distorted"
    for d in (distorted, work / "images", work / "sparse"):
        shutil.rmtree(d, ignore_errors=True)
    (distorted / "sparse").mkdir(parents=True)
    db = distorted / "database.db"
    log = work / "colmap_log.txt"
    log.write_text("")
    gpu = "1" if args.gpu else "0"

    print(f"[1/4] Finding features in {n_frames} frames")
    extract = option_prefix("feature_extractor", "FeatureExtraction", "SiftExtraction")
    colmap("feature_extractor",
           "--database_path", db, "--image_path", frames,
           "--ImageReader.single_camera", "1",          # one phone/camera for every frame
           "--ImageReader.camera_model", "OPENCV",
           f"--{extract}.use_gpu", gpu, log=log)

    print(f"[2/4] Matching features between frames ({args.matcher})")
    match = option_prefix(f"{args.matcher}_matcher", "FeatureMatching", "SiftMatching")
    extra = ["--SequentialMatching.overlap", "20", "--SequentialMatching.quadratic_overlap", "1"] \
        if args.matcher == "sequential" else []
    colmap(f"{args.matcher}_matcher", "--database_path", db, f"--{match}.use_gpu", gpu, *extra, log=log)

    print("[3/4] Solving camera positions (this is the slow part)")
    colmap("mapper", "--database_path", db, "--image_path", frames, "--output_path", distorted / "sparse", log=log)

    models = [m for m in (distorted / "sparse").iterdir() if (m / "images.bin").exists()]
    if not models:
        raise SystemExit("COLMAP could not reconstruct the scene. See the troubleshooting section of the README.")
    # COLMAP may split a scene into several pieces; keep the one with the most cameras.
    best = max(models, key=lambda m: count_bin(m / "images.bin"))
    registered = count_bin(best / "images.bin")

    print("[4/4] Undistorting images")
    colmap("image_undistorter", "--image_path", frames, "--input_path", best,
           "--output_path", work, "--output_type", "COLMAP", log=log)
    # The undistorter writes WORK/sparse/*.bin; gsplat expects WORK/sparse/0/.
    sparse = work / "sparse"
    (sparse / "0").mkdir(exist_ok=True)
    for f in sparse.iterdir():
        if f.is_file():
            f.rename(sparse / "0" / f.name)

    points = count_bin(sparse / "0" / "points3D.bin")
    pct = 100 * registered / n_frames
    print(f"\nCOLMAP placed {registered} of {n_frames} frames ({pct:.0f}%) and found {points} 3D points.")
    if len(models) > 1:
        print(f"Note: the scene split into {len(models)} pieces; only the largest is used.")
    if pct < 70:
        print("WARNING: fewer than 70% of frames were placed. The splat may have holes or blurry areas.\n"
              "Consider re-filming with slower movement and more overlap (see README troubleshooting).")


if __name__ == "__main__":
    sys.exit(main())
