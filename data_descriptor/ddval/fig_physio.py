"""Figure 6: wrist PPG validated against the device heart rate; thermal inversion validated against the camera spot meter;
within-session trajectories of wrist and facial temperature."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import BIDS, C, DOUBLE, NUM, SESSIONS, facetemp, panel, physio, save, session_window, tsv
from .physio import heart_rate_agreement


def _bland_altman(ax, a, b, color, s=3):
    m, d = (a + b) / 2, a - b
    bias, sd = d.mean(), d.std(ddof=1)
    ax.scatter(m, d, s=s, color=color, lw=0, alpha=0.5, rasterized=True)
    for y, ls in [(bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")]:
        ax.axhline(y, color="black", lw=0.6, ls=ls)
    return bias, sd


def _trajectories(ax, kind: str, color: str) -> None:
    curves = []
    grid = np.arange(-5, 46, 1.0)
    for sub, ses in SESSIONS:
        t0, t1 = session_window(sub, ses)
        if kind == "wrist":
            d = physio(sub, ses, "skintemp")
            if d is None:
                continue
            t, v = d.time_unix.to_numpy(), d.skin_temperature.to_numpy(float)
        else:
            d = facetemp(sub, ses)
            d = d[d.valid.astype(bool)]
            t, v = d.time_unix.to_numpy(), d.T_median_minus_bg.to_numpy(float)
        tm = (t - t0) / 60
        base = np.nanmedian(v[(tm >= 0) & (tm <= 5)])
        if not np.isfinite(base):
            continue
        y = np.interp(grid, tm, v - base, left=np.nan, right=np.nan)
        y[grid > (t1 - t0) / 60] = np.nan
        curves.append(y)
        ax.plot(grid, y, color=color, lw=0.4, alpha=0.35)
    Y = np.array(curves)
    ax.plot(grid, np.nanmedian(Y, axis=0), color="black", lw=1.0)
    ax.axvspan(0, 5, color="#EEEEEE", lw=0, zorder=0)
    ax.axhline(0, color="black", lw=0.4, ls=":")
    ax.set_xlabel("Time from baseline onset (min)")
    ax.set_xlim(-5, 45)
    return len(curves)


def make() -> None:
    hr = heart_rate_agreement()
    v = hr.dropna(subset=["hr_ppg"])
    fig = plt.figure(figsize=(DOUBLE, 100 / 25.4))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(v.hr_device, v.hr_ppg, s=3, color=C["red"], lw=0, alpha=0.5, rasterized=True)
    lim = [40, 115]
    ax.plot(lim, lim, color="black", lw=0.6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Device heart rate (bpm)")
    ax.set_ylabel("Heart rate from raw PPG (bpm)")
    r = np.corrcoef(v.hr_device, v.hr_ppg)[0, 1]
    ax.text(0.04, 0.92, f"r = {r:.2f}, n = {len(v)} min", transform=ax.transAxes, fontsize=5.5)
    panel(ax, "a", x=-0.3)

    ax = fig.add_subplot(gs[0, 1])
    bias, sd = _bland_altman(ax, v.hr_ppg.to_numpy(), v.hr_device.to_numpy(), C["red"])
    ax.set_xlabel("Mean of the two (bpm)")
    ax.set_ylabel("Raw PPG − device (bpm)")
    panel(ax, "b", x=-0.3)
    NUM.set("HRMinutes", len(v))
    NUM.set("HRMinutesAll", int(hr.groupby(["participant_id", "session_id"]).n_session_minutes.first().sum()))
    NUM.set("HRMinutesDevice", len(hr))
    NUM.set("HRr", r, "{:.2f}")
    NUM.set("HRBias", bias, "{:.1f}")
    NUM.set("HRSD", sd, "{:.1f}")
    NUM.set("HRLoALo", bias - 1.96 * sd, "{:.1f}")
    NUM.set("HRLoAHi", bias + 1.96 * sd, "{:.1f}")
    NUM.set("HRWithinFive", 100 * (np.abs(v.hr_ppg - v.hr_device) <= 5).mean(), "{:.0f}")

    ax = fig.add_subplot(gs[0, 2])
    fr = tsv("derivatives/facial_temperature/quality_control/thermal_frames.tsv")
    fr = fr[fr.inversion_ok.astype(bool) & fr.ocr_spot_ok.astype(bool)].dropna(subset=["inverted_spot", "ocr_spot"])
    bg = fr.spot_rel_position < 0.3
    for mask, col, lab in [(bg, C["grey"], "background"), (~bg, C["warm"], "warm target")]:
        ax.scatter((fr.inverted_spot[mask] + fr.ocr_spot[mask]) / 2, fr.inverted_spot[mask] - fr.ocr_spot[mask], s=2,
                   color=col, lw=0, alpha=0.5, label=lab, rasterized=True)
    d = fr.inverted_spot - fr.ocr_spot
    tb, ts = d.mean(), d.std(ddof=1)
    for y, ls in [(tb, "-"), (tb - 1.96 * ts, "--"), (tb + 1.96 * ts, "--")]:
        ax.axhline(y, color="black", lw=0.6, ls=ls)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel("Mean of the two (°C)")
    ax.set_ylabel("Inverted − camera spot (°C)")
    ax.legend(loc="lower left", fontsize=5, markerscale=2, handletextpad=0.2)
    panel(ax, "c", x=-0.3)
    NUM.set("ThermN", len(fr))
    NUM.set("ThermBias", tb, "{:.2f}")
    NUM.set("ThermLoALo", tb - 1.96 * ts, "{:.2f}")
    NUM.set("ThermLoAHi", tb + 1.96 * ts, "{:.2f}")
    dw = d[~bg]
    NUM.set("ThermWarmN", len(dw))
    NUM.set("ThermWarmMedian", dw.median(), "{:.2f}")

    ax = fig.add_subplot(gs[1, 0:1])
    n = _trajectories(ax, "wrist", C["warm"])
    ax.set_ylabel("Wrist skin temp.\nchange (°C)")
    panel(ax, "d", x=-0.3)
    NUM.set("SkinSessions", n)

    ax = fig.add_subplot(gs[1, 1:2])
    n = _trajectories(ax, "face", C["traffic"])
    ax.set_ylabel("Face − background\nchange (K)")
    panel(ax, "e", x=-0.3)
    NUM.set("FaceSessions", n)

    ax = fig.add_subplot(gs[1, 2])
    h = pd.concat([tsv(f"derivatives/wristband_vendor_metrics/{s}/{e}/{s}_{e}_task-cogbattery_desc-hrv_timeseries.tsv")
                   for s, e in SESSIONS
                   if (BIDS / f"derivatives/wristband_vendor_metrics/{s}/{e}").exists()])
    ax.hist(h.hrv_reliability.dropna(), bins=20, color=C["grey"], ec="white", lw=0.3)
    ax.set_xlabel("Device HRV reliability (%)")
    ax.set_ylabel("300-s windows")
    panel(ax, "f", x=-0.3)
    NUM.set("HRVRelMedian", h.hrv_reliability.median(), "{:.0f}")
    NUM.set("HRVRMSSDMedian", h.rmssd.median(), "{:.0f}")
    save(fig, "fig6_physiology")
