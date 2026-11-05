"""Technical validation of the SYNAPSE dataset: figures, tables and the numbers quoted in the Data Descriptor.

Usage (from this folder):  ../../../.venv/bin/python run.py [--only fig1 fig3 ...]
Reads output/dataset/SYNAPSE_dataset/ and writes output/deliverables/01_data_descriptor/{figures,generated}/.
"""
from __future__ import annotations

import argparse
import warnings

import pandas as pd

from ddval import common, eeg, fig_behaviour, fig_design, fig_eeg, fig_environment, fig_example, fig_physio, fig_quality, tables

warnings.filterwarnings("ignore", category=RuntimeWarning)


def eeg_metrics(recompute: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    q, s = common.GEN / "eeg_epoch_quality.csv", common.GEN / "eeg_baseline_psd.csv"
    if recompute or not (q.exists() and s.exists()):
        return eeg.session_metrics()
    return pd.read_csv(q), pd.read_csv(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--recompute-eeg", action="store_true")
    a = ap.parse_args()
    want = lambda k: a.only is None or k in a.only                    # noqa: E731
    common.style()
    eq, spectra = eeg_metrics(a.recompute_eeg)
    if want("fig1"):
        fig_design.make()
    if want("fig2"):
        fig_example.make()
    if want("fig3"):
        fig_quality.make(eq)
    if want("fig4"):
        fig_behaviour.make()
    if want("fig5"):
        fig_eeg.make(eq, spectra)
    if want("fig6"):
        fig_physio.make()
    if want("fig7"):
        fig_environment.make()
    if want("tables"):
        tables.make()
    common.NUM.write()
    print(f"figures: {common.FIG}\nnumbers: {common.GEN / 'numbers.tex'}")


if __name__ == "__main__":
    main()
