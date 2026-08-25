"""Packaging and public-import smoke tests."""

from importlib.metadata import version
import unittest

import qrtlib


class TestPackage(unittest.TestCase):
    def test_version_has_a_single_source(self) -> None:
        self.assertEqual(qrtlib.__version__, version("QRTlib"))

    def test_public_api_is_importable(self) -> None:
        self.assertEqual(
            qrtlib.__all__, ["QCTGate", "QHTGate", "QSTGate", "__version__"]
        )
        self.assertEqual(qrtlib.QCTGate.__name__, "QCTGate")
        self.assertEqual(qrtlib.QHTGate.__name__, "QHTGate")
        self.assertEqual(qrtlib.QSTGate.__name__, "QSTGate")
