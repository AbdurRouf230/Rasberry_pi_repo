#!/usr/bin/env bash
# Debian 13 (Trixie) / Raspberry Pi OS — YOLO + CSI camera only.
# No ROS 2. No Gazebo.
#
#   cd ~/Debian\ OS\ Yolo    # or this folder
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

# picamera2 is an apt package; the venv must see system site-packages.
python3 -m venv --system-site-packages "$HERE/.venv"
# shellcheck disable=SC1091
source "$HERE/.venv/bin/activate"
pip install -U pip wheel
pip install -r "$HERE/requirements.txt"
deactivate

echo
echo "DONE. Nothing was launched."
echo "  source \"$HERE/.venv/bin/activate\""
echo "  python3 check_camera.py"
echo "  python3 bench_camera.py --imgsz 320 --conf 0.35"
echo "  python3 bench_images.py --imgsz 320 --conf 0.35"
