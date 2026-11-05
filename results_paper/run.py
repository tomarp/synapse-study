"""Run the analyses of the SYNAPSE results paper and draw its figures.

Usage (from this folder):  ../../../.venv-results/bin/python -I -B run.py [--only A B D E] [--no-figures]

Blind by default (condition labels scrambled, outputs in <out>/blind/); SYNAPSE_UNBLIND=1 runs on the true labels.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
logging.getLogger("pymc").setLevel(logging.ERROR)

from rp import analyses, common  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="analysis families: A B D E")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    print(f"{'UNBLINDED' if common.UNBLIND else 'BLIND'} run -> {common.PAPER}")
    analyses.run(a.only)
    if not a.no_figures and a.only is None:
        from rp import figures
        figures.run()


if __name__ == "__main__":
    main()
