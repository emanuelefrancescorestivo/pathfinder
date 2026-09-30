"""The documents' result tables match results/*.json (invariant: no hand-typed numbers)."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_documents_are_rendered_from_results():
    if not (ROOT / "results" / "final.json").exists():
        pytest.skip("results not generated")
    out = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "render_readme.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
