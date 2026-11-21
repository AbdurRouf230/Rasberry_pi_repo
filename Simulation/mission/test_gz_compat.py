"""Flight node must survive a Gazebo protobuf TypeError and still take off."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gz_compat  # noqa: E402


class ProtobufCompatTests(unittest.TestCase):
    def test_apply_sets_python_implementation(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
            value = gz_compat.apply_protobuf_compat()
            self.assertEqual(value, "python")
            self.assertEqual(os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"], "python")

    def test_gz_typeerror_does_not_abort_import(self):
        def boom(name, *args, **kwargs):
            raise TypeError("Descriptors cannot be created directly.")

        with patch.dict(sys.modules, {"gz.transport13": None, "gz.msgs10.image_pb2": None}):
            with patch("builtins.__import__", side_effect=boom):
                node, image, fmt, ok = gz_compat.import_gz_camera()
        self.assertFalse(ok)
        self.assertIsNone(node)
        self.assertIsNone(image)
        self.assertIsNone(fmt)


if __name__ == "__main__":
    unittest.main()
