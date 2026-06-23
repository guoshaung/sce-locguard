#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/watermark_exps"
REPO_URL="https://github.com/guoshaung/sce-locguard.git"
ENV_NAME="sce-locguard"

echo "[Stage 9 setup] Creating project root: ${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

if [ ! -d ".git" ]; then
  echo "[Stage 9 setup] Cloning repository..."
  git clone "${REPO_URL}" .
else
  echo "[Stage 9 setup] Repository already exists. Pulling latest main..."
  git fetch origin
  git checkout main
  git pull --ff-only origin main
fi

echo "[Stage 9 setup] Creating conda environment: ${ENV_NAME}"
if command -v conda >/dev/null 2>&1; then
  conda create -n "${ENV_NAME}" python=3.10 -y || true
  echo "Run manually if needed: conda activate ${ENV_NAME}"
else
  echo "WARNING: conda was not found. Install Miniconda/Anaconda first."
fi

echo "[Stage 9 setup] Checking NVIDIA driver..."
nvidia-smi || echo "WARNING: nvidia-smi failed. Check GPU driver/CUDA installation."

echo "[Stage 9 setup] Installing Python dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "[Stage 9 setup] Checking PyTorch CUDA..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

cat <<'TXT'

Next steps:
1. Upload dataset.zip to /data/watermark_exps/.
2. Run: cd /data/watermark_exps && unzip dataset.zip
3. Verify:
   /data/watermark_exps/dataset/valAGE-Set
   /data/watermark_exps/dataset/valAGE-Set-Mask
4. Create server branch:
   git checkout -b stage9-server
5. Start with 5-sample smoke test, then 50, 200, and finally 1000 samples.

TXT

