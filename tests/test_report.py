"""The report script runs end-to-end and embeds real computed numbers."""

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_make_report_runs_and_writes_numbers(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run(
        [sys.executable, "-m", "src.make_report"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    report = ROOT / "report" / "report.md"
    assert report.exists()
    text = report.read_text()
    # Real numbers must appear: true tau, a CI, variance reduction %, power table rows.
    assert "tau = +0.35" in text
    assert re.search(r"reduced\nthe sampling variance of the ATE estimator by "
                    r"\*\*\d+\.\d+%\*\*", text)
    assert re.search(r"\| 2000 \| 1000 \|", text)
    assert (ROOT / "report" / "figures" / "power_curve.png").exists()
    assert "IPW" in text and "Doubly robust" in text
