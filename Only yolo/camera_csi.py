"""Grab RGB frames from the Raspberry Pi CSI camera (picamera2), USB fallback."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

Size = Tuple[int, int]


def open_camera(size: Size = (640, 480)) -> Tuple[str, Any]:
    """Return (backend, handle). Prefer picamera2 on a Pi CSI ribbon."""
    try:
        from picamera2 import Picamera2

        cam = Picamera2()
        cfg = cam.create_preview_configuration(
            main={"size": size, "format": "RGB888"},
        )
        cam.configure(cfg)
        cam.start()
        return "picamera2", cam
    except Exception as exc:
        print(f"picamera2 not available ({exc}); trying OpenCV /dev/video0")

    import cv2

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    if not cap.isOpened():
        raise RuntimeError("No CSI picamera2 and no OpenCV camera on /dev/video0")
    return "cv2", cap


def read_rgb(kind: str, cam: Any) -> Optional[np.ndarray]:
    if kind == "picamera2":
        frame = cam.capture_array()
        if frame is None:
            return None
        if frame.ndim == 3 and frame.shape[2] >= 3:
            return np.ascontiguousarray(frame[:, :, :3])
        return frame
    ok, bgr = cam.read()
    if not ok or bgr is None:
        return None
    import cv2

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def close_camera(kind: str, cam: Any) -> None:
    try:
        if kind == "picamera2":
            cam.stop()
            cam.close()
        else:
            cam.release()
    except Exception:
        pass
