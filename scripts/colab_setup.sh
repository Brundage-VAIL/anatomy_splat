#!/usr/bin/env bash
# Installs everything the notebook needs on a Google Colab GPU runtime.
# Versions are pinned to the combination used to build the equine heart splat.
set -euo pipefail

GSPLAT_VERSION=1.5.3
GSPLAT_DIR=/content/gsplat-repo
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==> Installing COLMAP and ffmpeg"
apt-get -qq update > /dev/null
DEBIAN_FRONTEND=noninteractive apt-get -qq install -y colmap ffmpeg > /dev/null

echo "==> Installing gsplat ${GSPLAT_VERSION} and training dependencies"
pip install -q "gsplat==${GSPLAT_VERSION}"
pip install -q \
  "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" \
  "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787" \
  viser "imageio[ffmpeg]" scikit-learn tqdm "torchmetrics[image]" opencv-python "tyro>=0.8.8" \
  tensorboard tensorly pyyaml matplotlib splines plyfile

echo "==> Building fused-ssim (takes a few minutes)"
pip install -q \
  "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5"

# Colab ships the Hugging Face 'datasets' package, which hides gsplat's examples/datasets folder.
pip uninstall -q -y datasets || true

echo "==> Fetching gsplat ${GSPLAT_VERSION} example trainers"
if [ ! -d "${GSPLAT_DIR}" ]; then
  git clone -q --depth 1 --branch "v${GSPLAT_VERSION}" https://github.com/nerfstudio-project/gsplat.git "${GSPLAT_DIR}"
fi
touch "${GSPLAT_DIR}/examples/datasets/__init__.py"

python "${HERE}/patch_pycolmap.py"
echo "==> Setup complete"
