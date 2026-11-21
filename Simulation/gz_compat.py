"""Keep Gazebo image protobufs working with newer pip protobuf."""

from __future__ import annotations

import os


def apply_protobuf_compat() -> str:
    """Must run before google.protobuf is imported."""
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    return os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]


def import_gz_camera():
    apply_protobuf_compat()
    try:
        from gz.transport13 import Node as GzNode
        from gz.msgs10.image_pb2 import Image as GzImage, PixelFormatType
        return GzNode, GzImage, PixelFormatType, True
    except Exception:
        return None, None, None, False
