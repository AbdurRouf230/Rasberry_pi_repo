# Rasberry_pi_repo

Raspberry Pi 4B pack for IntelleSwarm box YOLO and (later) ROS 2.

- **Debian OS Yolo** — run this on Raspberry Pi OS **Debian 13 (trixie)**. CSI camera + `best.pt`. No ROS 2.
- **Only yolo** — same YOLO bench (generic).
- **Simulation** — Gazebo on the ground PC + Pi ROS 2 companion. ROS 2 Humble needs **Ubuntu 22.04**.

On Debian 13:

```bash
cd ~/DebianOSYolo
sed -i 's/\r$//' install_deps.sh
bash install_deps.sh
source .venv/bin/activate
python3 check_camera.py
python3 bench_camera.py --imgsz 320 --conf 0.35
```
