#!/usr/bin/env bash
# Install Raspberry Pi 4B dependencies only. Does not run YOLO or the sim.
#
#   cd Rasberry-Repo    # or ~/Rasberry-Repo on the Pi
#   sed -i 's/\r$//' install_deps.sh
#   bash install_deps.sh
#
# Ubuntu 22.04 64-bit. Options:
#   SKIP_YOLO=1   skip camera + ultralytics
#   SKIP_ROS=1    skip ROS 2 Humble
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Run this on the Raspberry Pi (Ubuntu 22.04 64-bit), not Windows."
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YOLO_DIR="$ROOT/Only yolo"
SIM_DIR="$ROOT/Simulation"

echo "============================================================="
echo " Rasberry-Repo dependency install (no run)"
echo " $ROOT"
echo "============================================================="

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y \
  software-properties-common curl gnupg lsb-release ca-certificates \
  python3 python3-pip python3-venv python3-dev \
  libopenblas-dev libatlas-base-dev \
  locales

sudo locale-gen en_US.UTF-8 >/dev/null 2>&1 || true

# --- Only yolo: CSI camera + ultralytics ---
if [[ "${SKIP_YOLO:-0}" != "1" ]]; then
  echo
  echo "=== Only yolo (camera + PyTorch YOLO) ==="
  sudo apt-get install -y python3-opencv || true
  sudo apt-get install -y python3-libcamera python3-picamera2 || \
    echo "WARN: picamera2/libcamera not in apt. CSI camera may need Raspberry Pi OS packages."

  if [[ ! -f "$YOLO_DIR/requirements.txt" ]]; then
    echo "WARN: missing $YOLO_DIR/requirements.txt — skip venv"
  else
    python3 -m venv "$YOLO_DIR/.venv"
    # shellcheck disable=SC1091
    source "$YOLO_DIR/.venv/bin/activate"
    pip install -U pip wheel
    pip install -r "$YOLO_DIR/requirements.txt"
    deactivate
    echo "YOLO venv: $YOLO_DIR/.venv"
    echo "  source \"$YOLO_DIR/.venv/bin/activate\""
  fi
else
  echo "SKIP_YOLO=1"
fi

# --- Simulation companion: ROS 2 Humble (no Gazebo on the Pi) ---
if [[ "${SKIP_ROS:-0}" != "1" ]]; then
  echo
  echo "=== ROS 2 Humble (Pi companion only) ==="
  . /etc/os-release
  if [[ "${VERSION_ID:-}" != "22.04" ]]; then
    echo "WARN: ROS 2 Humble is for Ubuntu 22.04. This OS is ${VERSION_ID:-unknown}."
  fi

  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu ${UBUNTU_CODENAME} main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y \
    ros-humble-ros-base \
    python3-colcon-common-extensions \
    ros-humble-rmw-fastrtps-cpp \
    ros-humble-std-msgs \
    ros-humble-rclpy

  if ! grep -q "source /opt/ros/humble/setup.bash" "$HOME/.bashrc" 2>/dev/null; then
    echo "source /opt/ros/humble/setup.bash" >> "$HOME/.bashrc"
  fi
  echo "ROS 2: source /opt/ros/humble/setup.bash"
else
  echo "SKIP_ROS=1"
fi

echo
echo "DONE. Dependencies only — nothing was launched."
echo "  YOLO:  $YOLO_DIR/STEPS_RASPBERRY_PI.txt"
echo "  ROS2:  $SIM_DIR/STEPS_RASPBERRY_PI.txt"
