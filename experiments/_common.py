"""Shared helpers: every experiment writes its numbers to results/<name>.json.

The README's result tables are rendered from these files (experiments/render_readme.py),
so no number in the documentation is typed by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))


def save(name: str, payload: dict[str, Any]) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def load(name: str) -> dict[str, Any]:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} is missing: run the experiment that writes it")
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
