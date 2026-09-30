"""Run every experiment in order, then render the documents.

The OULAD experiment runs only if its files are present.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STEPS = [
    "00_is_it_synthetic.py",
    "01_select.py",
    "02_final_test.py",
    "03_audit_original.py",
    "04_figures.py",
    "06_feature_questions.py",
    "07_admin_views.py",
    "08_story_figures.py",
]
for step in STEPS:
    print(f"== {step}")
    runpy.run_path(str(HERE / step), run_name="__main__")

if (HERE.parent / "data" / "raw" / "oulad" / "studentInfo.csv").exists():
    for step in ("05_early_warning_oulad.py", "09_time_oulad.py"):
        print(f"== {step}")
        runpy.run_path(str(HERE / step), run_name="__main__")
else:
    print("== 05 and 09 (OULAD) skipped: data/raw/oulad/ is empty")

runpy.run_path(str(HERE / "render_readme.py"), run_name="__main__")
