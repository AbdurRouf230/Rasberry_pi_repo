#!/usr/bin/env bash
# Debian 13 (Trixie) / Raspberry Pi OS — YOLO + CSI camera only.
# No ROS 2. No Gazebo. CPU PyTorch only (a Pi has no NVIDIA GPU).
#
#   cd ~/Rasberry_pi_repo/Debian\ OS\ Yolo
#   sed -i 's/\r$//' install_deps.sh
#   bash install_deps.sh
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Run this on the Raspberry Pi, not Windows."
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. /etc/os-release
echo "============================================================="
echo " Debian OS Yolo — install only"
echo " OS: ${PRETTY_NAME:-unknown}"
echo " $HERE"
echo "============================================================="

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y \
  python3 python3-pip python3-venv python3-dev \
  python3-opencv python3-libcamera python3-picamera2 \
  libopenblas-dev libcamera-tools \
  curl ca-certificates

if [[ ! -f "$HERE/requirements.txt" ]]; then
  echo "Missing $HERE/requirements.txt"
  exit 1
fi

# Use Debian python so apt picamera2 is importable.
# A pyenv 3.11 venv cannot see Debian 3.13's python3-picamera2.
PY="${PYTHON:-/usr/bin/python3}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi
echo "venv python: $PY ($("$PY" -V 2>&1))"

# Default pip torch on aarch64 is CUDA (~2 GB) and fills a Pi SD card.
# Always install the CPU wheel from pytorch.org.
install_cpu_torch() {
  echo "Installing CPU PyTorch 2.7.1 (no CUDA). Do not use torch 2.8+ on Debian 13;"
  echo "those wheels need OpenBLAS sbgemm_ which this Pi does not provide."
  python -m pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.7.1" "torchvision==0.22.1"
}

if [[ -d "$HERE/.venv" ]]; then
  echo "Removing previous venv (may contain a partial CUDA torch download)..."
  rm -rf "$HERE/.venv"
fi

# Drop incomplete CUDA wheels if a previous run filled the disk.
rm -rf "${HOME}/.cache/pip"
sudo apt-get clean

avail_kb="$(df -Pk "$HERE" | awk 'NR==2 {print $4}')"
if [[ "${avail_kb:-0}" -lt 2500000 ]]; then
  echo "Need about 2.5 GB free. This filesystem has $((avail_kb / 1024)) MB."
  echo "Free space, then rerun. Typical cleanup:"
  echo "  rm -rf \"$HERE/.venv\" ~/.cache/pip"
  echo "  sudo apt-get clean"
  echo "  df -h"
  exit 1
fi

"$PY" -m venv --system-site-packages "$HERE/.venv"
# shellcheck disable=SC1091
source "$HERE/.venv/bin/activate"
python -m pip install -U pip wheel
install_cpu_torch
# Never `pip install ultralytics` with deps: that upgrades torch to CUDA on aarch64.
# --ignore-installed: Debian apt numpy is on sys.path (--system-site-packages)
# and pip would skip installing; mixing apt numpy + pip torch fails the numpy sanity check.
python -m pip install --ignore-installed --no-cache-dir "numpy>=1.24.0"
python -m pip install \
  "Pillow>=10.0.0" \
  matplotlib pyyaml requests scipy tqdm psutil
python -m pip install --no-deps "ultralytics>=8.3.0"
python -m pip install --no-deps ultralytics-thop || true
python - <<'PY'
import numpy
import torch
print("numpy", numpy.__version__, numpy.__file__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if numpy.__file__.startswith("/usr/lib"):
    raise SystemExit("Still using apt numpy; pip numpy must shadow it in the venv")
if "+cpu" not in torch.__version__ and torch.cuda.is_available():
    raise SystemExit("Refusing CUDA torch on Raspberry Pi")
PY
deactivate

echo
echo "DONE. Nothing was launched."
echo "  source \"$HERE/.venv/bin/activate\""
echo "  python3 check_camera.py"
echo "  python3 bench_camera.py --imgsz 320 --conf 0.35"
echo "  python3 bench_images.py --imgsz 320 --conf 0.35"
