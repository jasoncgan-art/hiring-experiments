#!/usr/bin/env bash
# End-to-end pipeline: install deps, run tests, regenerate the report.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "creating .venv (with system site packages: pinned numpy/pandas/scipy/matplotlib come from the OS) ..."
  python3 -m venv --system-site-packages .venv
fi
echo "installing requirements ..."
# Pinned numpy/pandas/scipy/matplotlib are provided by the OS packages at the
# exact pinned versions; pip only needs to add pytest (no cp312 wheels exist
# for matplotlib==3.6.3, and building it from source requires network
# downloads of freetype).
.venv/bin/pip install -q pytest

echo "running tests ..."
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q

echo "generating report ..."
PYTHONPATH=. .venv/bin/python -m src.make_report

echo "done."
