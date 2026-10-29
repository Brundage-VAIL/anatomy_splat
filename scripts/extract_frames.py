"""Pick sharp, evenly spaced frames from one or more videos (or a folder of photos).

Usage:
    python extract_frames.py VIDEO_OR_FOLDER [MORE ...] --out WORK/input --num-frames 150

The output folder is what COLMAP reads in the next step.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def resize_to_max(img, max_size):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def sharpness(img):
    # Variance of the Laplacian: higher = sharper. Computed on a small copy for speed.
    small = resize_to_max(img, 640)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def open_video(path, tmp_dir):
    """Open a video with OpenCV, transcoding with ffmpeg first if OpenCV can't decode it (e.g. some HEVC .MOV files)."""
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened() and cap.read()[0]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return cap
    cap.release()
    print(f"  OpenCV can't read {path.name} directly, converting with ffmpeg...")
    converted = Path(tmp_dir) / (path.stem + "_h264.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-crf", "18", "-an", str(converted)],
        check=True,
    )
    cap = cv2.VideoCapture(str(converted))
    if not cap.isOpened():
        raise RuntimeError(f"Could not read video {path}")
    return cap


def extract_from_video(path, num_frames, out_dir, max_size, prefix, tmp_dir):
    cap = open_video(path, tmp_dir)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if total <= 0:
        raise RuntimeError(f"Could not count frames in {path}")
    num_frames = min(num_frames, total)
    # Only score ~8 candidate frames per output frame so long 4K videos stay fast.
    stride = max(1, total // (num_frames * 8))
    print(f"  {path.name}: {total} frames ({total / fps:.0f} s), keeping the sharpest of each of {num_frames} windows")

    best = {}  # window index -> (score, image)
    saved = 0
    idx = 0
    current_bin = 0

    def flush(b):
        nonlocal saved
        if b in best:
            _, img = best.pop(b)
            cv2.imwrite(str(out_dir / f"{prefix}_{saved:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

    while True:
        if not cap.grab():
            break
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if ok:
                b = min(idx * num_frames // total, num_frames - 1)
                if b != current_bin:
                    flush(current_bin)
                    current_bin = b
                frame = resize_to_max(frame, max_size)
                score = sharpness(frame)
                if b not in best or score > best[b][0]:
                    best[b] = (score, frame)
        idx += 1
    flush(current_bin)
    cap.release()
    return saved


def copy_images(folder, out_dir, max_size, prefix):
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for i, p in enumerate(files):
        img = cv2.imread(str(p))
        if img is None:
            print(f"  skipping unreadable image {p.name}")
            continue
        cv2.imwrite(str(out_dir / f"{prefix}_{i:04d}.jpg"), resize_to_max(img, max_size), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="video files and/or folders of videos or photos")
    ap.add_argument("--out", type=Path, required=True, help="folder to write frames into")
    ap.add_argument("--num-frames", type=int, default=150, help="total frames to keep across all videos")
    ap.add_argument("--max-size", type=int, default=1600, help="longest image side in pixels")
    args = ap.parse_args()

    videos, image_dirs = [], []
    for p in args.inputs:
        if p.is_dir():
            videos += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS)
            if any(f.suffix.lower() in IMAGE_EXTS for f in p.iterdir()):
                image_dirs.append(p)
        elif p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)
        else:
            raise SystemExit(f"Don't know what to do with {p} (expected a video or a folder)")
    if not videos and not image_dirs:
        raise SystemExit("No videos or images found in: " + ", ".join(map(str, args.inputs)))

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    # Split the frame budget between videos in proportion to their length.
    lengths = []
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        lengths.append(max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1))
        cap.release()

    total_saved = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, v in enumerate(videos):
            n = max(10, round(args.num_frames * lengths[i] / sum(lengths)))
            total_saved += extract_from_video(v, n, args.out, args.max_size, f"v{i:02d}", tmp)
    for i, d in enumerate(image_dirs):
        n = copy_images(d, args.out, args.max_size, f"p{i:02d}")
        print(f"  {d}: copied {n} photos")
        total_saved += n

    print(f"Wrote {total_saved} images to {args.out}")


if __name__ == "__main__":
    main()
