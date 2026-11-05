"""Figure 5: EEG — usable data per channel, baseline spectra and alpha peaks."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import C, CHANNELS, DOUBLE, GEN, NUM, panel, save, tsv
from .eeg import PTP_UV, alpha_peaks

CH_COL = {"TP9": C["red"], "AF7": C["cool"], "AF8": C["sky"], "TP10": C["traffic"]}
MIN_EPOCHS = 30            # >= 60 s of usable baseline for a session to enter the spectra


def make(eq: pd.DataFrame, spectra: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(DOUBLE, 62 / 25.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.25, 1.25], wspace=0.38)

    # a: usable fraction per channel (baseline vs whole recording)
    ax = fig.add_subplot(gs[0])
    for k, scope in enumerate(["baseline", "recording"]):
        d = eq[eq.scope == scope]
        for j, ch in enumerate(CHANNELS):
            v = 100 * d[d.channel == ch].usable_fraction.to_numpy()
            x = j * 2.6 + k
            ax.boxplot(v, positions=[x], widths=0.7, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor="white" if k else CH_COL[ch], edgecolor=CH_COL[ch], lw=0.6),
                       medianprops=dict(color="black", lw=0.7), whiskerprops=dict(color=CH_COL[ch], lw=0.6),
                       capprops=dict(color=CH_COL[ch], lw=0.6))
            ax.scatter(np.full(len(v), x) + np.random.default_rng(j + 10 * k).uniform(-0.2, 0.2, len(v)), v, s=2,
                       color="black", lw=0, zorder=3)
    ax.set_xticks([j * 2.6 + 0.5 for j in range(4)], CHANNELS)
    ax.set_ylabel("Usable 2-s epochs (%)")
    ax.set_ylim(-3, 103)
    ax.text(0.02, 1.02, "filled: baseline   open: whole recording", transform=ax.transAxes, fontsize=5)
    panel(ax, "a", x=-0.3)

    # b: baseline PSD per channel (sessions with enough usable epochs)
    ax = fig.add_subplot(gs[1])
    used = {}
    for ch in CHANNELS:
        d = spectra[(spectra.channel == ch) & (spectra.n_epochs >= MIN_EPOCHS)]
        P = d.pivot_table(index=["participant_id", "session_id"], columns="freq", values="psd_uv2_hz")
        used[ch] = len(P)
        if not len(P):
            continue
        logP = 10 * np.log10(P.to_numpy())
        f = P.columns.to_numpy()
        med, q1, q3 = np.median(logP, 0), *np.percentile(logP, [25, 75], axis=0)
        ax.fill_between(f, q1, q3, color=CH_COL[ch], alpha=0.18, lw=0)
        ax.plot(f, med, color=CH_COL[ch], lw=0.9, label=f"{ch} (n = {len(P)})")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Baseline PSD (dB re 1 µV²/Hz)")
    ax.legend(loc="lower left", fontsize=5, handlelength=1.2)
    panel(ax, "b", x=-0.22)

    # c: distribution of epoch peak-to-peak amplitudes (whole recordings), with the usability criterion
    ax = fig.add_subplot(gs[2])
    hist = pd.read_csv(GEN / "eeg_ptp_histogram.csv")
    for ch in CHANNELS:
        h = hist[hist.channel == ch].groupby(["bin_lo", "bin_hi"])["count"].sum().reset_index()
        centres = np.sqrt(h.bin_lo * h.bin_hi)
        frac = h["count"] / h["count"].sum()
        ax.plot(centres, 100 * frac, color=CH_COL[ch], lw=0.9, label=ch)
    ax.axvline(PTP_UV, color="black", lw=0.6, ls="--")
    ax.text(PTP_UV * 1.08, ax.get_ylim()[1] * 0.95, f"{PTP_UV:.0f} µV", fontsize=5.5, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("Epoch peak-to-peak (µV); railed at right")
    ax.set_xlim(10, 6000)
    ax.set_ylabel("Epochs (%)")
    ax.legend(loc="upper left", fontsize=5, handlelength=1.2)
    panel(ax, "c", x=-0.22)
    save(fig, "fig5_eeg")

    peaks = alpha_peaks(spectra[spectra.n_epochs >= MIN_EPOCHS])
    for ch in CHANNELS:
        NUM.set(f"PSDSessions{ch}", used[ch])
        p = peaks[peaks.channel == ch]
        NUM.set(f"AlphaPeak{ch}", int(p.has_peak.sum()))
        if p.has_peak.any():
            NUM.set(f"AlphaFreq{ch}", p[p.has_peak].peak_hz.median(), "{:.1f}")
    eqr = tsv("derivatives/quality_control/eeg_quality.tsv")
    NUM.set("EEGHours", eqr.n_samples.sum() / 256 / 3600, "{:.1f}")
    rails = eqr[[f"rail_fraction_{c}" for c in CHANNELS]]
    NUM.set("RailSessions", int((rails >= 0.10).any(axis=1).sum()))
