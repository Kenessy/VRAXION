"""Tests for non-destructive VRX sync block patching."""

from __future__ import annotations

import unittest

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

from tools.vrx_sync_linear_projects import SYNC_BEGIN, SYNC_END, patch_sync_block


class TestVrxSyncBlock(unittest.TestCase):
    def test_append_when_missing(self) -> None:
        body = "hello\nworld\n"
        out = patch_sync_block(body, "SNAP")
        self.assertIn(SYNC_BEGIN, out)
        self.assertIn(SYNC_END, out)
        self.assertIn("hello\nworld", out)
        self.assertIn("SNAP", out)

    def test_replace_when_present(self) -> None:
        body = f"top\n{SYNC_BEGIN}\nOLD\n{SYNC_END}\nbottom\n"
        out = patch_sync_block(body, "NEW")
        self.assertIn("top\n", out)
        self.assertIn("\nbottom\n", out)
        self.assertIn("NEW", out)
        self.assertNotIn("OLD", out)

    def test_ambiguous_markers_raise(self) -> None:
        body = f"{SYNC_BEGIN}\nA\n{SYNC_END}\n{SYNC_BEGIN}\nB\n{SYNC_END}\n"
        with self.assertRaises(Exception):
            patch_sync_block(body, "X")

