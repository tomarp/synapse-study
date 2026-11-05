"""Figure 2: all data streams of one session on the common timeline (sub-P04, session A)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import spectrogram

from . import eeg
from .common import C, DOUBLE, TASK_LABEL, behaviour, facetemp, panel, physio, room, save, session_window, timeline, tsv, vendor

SUB, SES = "sub-P04", "ses-A"
EEG_CH = "AF8"


def _phases(ax, t0, show_labels=False):
    t = timeline()
    t = t[(t.participant_id == SUB) & (t.session_id == SES)]
    for r in t[t.phase.isin(["baseline", "T1", "T2", "T3"])].itertuples():
        ax.axvspan((r.onset_unix - t0) / 60, (r.offset_unix - t0) / 60, color="#EBEBEB", lw=0, zorder=0)
    for r in t[t.phase == "block"].itertuples():
        ax.axvline((r.onset_unix - t0) / 60, color="#BDBDBD", lw=0.4, zorder=0)


def make() -> None:
    t0, t1 = session_window(SUB, SES)
    t = timeline()
    t = t[(t.participant_id == SUB) & (t.session_id == SES)]
    fig, axes = plt.subplots(7, 1, figsize=(DOUBLE, 150 / 25.4), sharex=True,
                             gridspec_kw={"height_ratios": [0.55, 1.2, 0.9, 0.8, 0.8, 0.8, 0.8], "hspace": 0.18})
    m = lambda x: (np.asarray(x, float) - t0) / 60                                       # noqa: E731

    # a: protocol
    ax = axes[0]
    for r in t[t.phase.isin(["baseline", "T1", "T2", "T3"])].itertuples():
        ax.barh(0, m(r.offset_unix) - m(r.onset_unix), left=m(r.onset_unix), color="#BDBDBD", ec="white", lw=0.4)
        ax.text((m(r.onset_unix) + m(r.offset_unix)) / 2, 0, r.phase.replace("baseline", "Base."), ha="center", va="center", fontsize=5)
    for r in t[t.phase == "block"].itertuples():
        col = C[r.soundscape] if r.soundscape in C else C["sky"]
        ax.barh(0, m(r.offset_unix) - m(r.onset_unix), left=m(r.onset_unix), color=col, ec="white", lw=0.4, alpha=0.85)
        ax.text((m(r.onset_unix) + m(r.offset_unix)) / 2, 0, TASK_LABEL[r.task].split(" (")[0].replace("Shape comparison", "Shape")
                .replace("Visual counting", "Count").replace("Arrow flanker", "Flanker").replace("Digit flanker", "Flanker")
                .replace("Word Stroop", "Stroop").replace("Emoji Stroop", "Stroop"), ha="center", va="center", fontsize=4.8, color="white")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.5, 0.5)
    ax.text(0.0, 1.08, "Task blocks with road traffic", color=C["traffic"], transform=ax.transAxes, fontsize=5.5)
    ax.text(0.17, 1.08, "with babble", color=C["people"], transform=ax.transAxes, fontsize=5.5)
    panel(ax, "a", x=-0.075, y=0.9)

    # b: EEG spectrogram
    ax = axes[1]
    raw = eeg.read(SUB, SES).load_data().filter(1.0, 45.0, verbose="ERROR")
    ev = tsv(f"{SUB}/{SES}/eeg/{SUB}_{SES}_task-cogbattery_events.tsv")
    rec0 = pd.Timestamp(ev.onset_time.iloc[0]).timestamp() - ev.onset.iloc[0]
    x = raw.get_data(picks=[EEG_CH])[0] * 1e6
    f, ts, S = spectrogram(x, fs=raw.info["sfreq"], nperseg=1024, noverlap=512)
    keep = f <= 40
    im = ax.pcolormesh(m(rec0 + ts), f[keep], 10 * np.log10(S[keep] + 1e-12), shading="auto", cmap="magma",
                       vmin=np.percentile(10 * np.log10(S[keep] + 1e-12), 5), vmax=np.percentile(10 * np.log10(S[keep] + 1e-12), 99),
                       rasterized=True)
    ax.set_ylabel(f"EEG {EEG_CH}\n(Hz)")
    cb = fig.colorbar(im, ax=ax, pad=0.005, fraction=0.015)
    cb.set_label("dB µV²/Hz", fontsize=5)
    cb.ax.tick_params(labelsize=5)
    panel(ax, "b", x=-0.075, y=0.85)

    # c: trials (accuracy per trial over time)
    ax = axes[2]
    b = behaviour()
    b = b[(b.participant_id == SUB) & (b.session_id == SES)]
    tt = m(b.submitted_unix)
    ax.scatter(tt, b.rt_ms / 1000, s=2.5, c=np.where(b.correct == 1, C["cool"], C["red"]), lw=0, rasterized=True)
    ax.set_yscale("log")
    ax.set_ylabel("RT (s)")
    ax.text(0.005, 0.86, "correct", color=C["cool"], fontsize=5, transform=ax.transAxes)
    ax.text(0.055, 0.86, "error", color=C["red"], fontsize=5, transform=ax.transAxes)
    panel(ax, "c", x=-0.075, y=0.85)

    # d: heart rate
    ax = axes[3]
    hr = vendor(SUB, SES, "heartrate")
    hr = hr[(hr.time_unix >= t0 - 300) & (hr.time_unix <= t1 + 300)]
    ax.plot(m(hr.time_unix + 30), hr.heart_rate_bpm, color=C["red"], marker="o", ms=1.5, lw=0.7)
    ax.set_ylabel("HR (bpm)")
    panel(ax, "d", x=-0.075, y=0.85)

    # e: skin temperature (wrist) and facial temperature
    ax = axes[4]
    sk = physio(SUB, SES, "skintemp")
    sk = sk[(sk.time_unix >= t0 - 300) & (sk.time_unix <= t1 + 300)]
    ax.plot(m(sk.time_unix), sk.skin_temperature, color=C["warm"], lw=0.8)
    ax.set_ylabel("Wrist skin\n(°C)")
    panel(ax, "e", x=-0.075, y=0.85)

    ax = axes[5]
    ft = facetemp(SUB, SES)
    ft = ft[ft.valid.astype(bool)]
    ax.plot(m(ft.time_unix), ft.T_median_minus_bg, color=C["traffic"], lw=0.6, marker="o", ms=0.8)
    ax.set_ylabel("Face−bg.\n(K, rel.)")
    panel(ax, "f", x=-0.075, y=0.85)

    # g: room
    ax = axes[6]
    r = room()
    r = r[(r.time_unix >= t0 - 300) & (r.time_unix <= t1 + 300)]
    ax.plot(m(r.time_unix), r.co2, color=C["grey"], lw=0.8)
    ax.set_ylabel("CO$_2$ (ppm)")
    ax2 = ax.twinx()
    ax2.plot(m(r.time_unix), r.air_temperature, color=C["people"], lw=0.8)
    ax2.set_ylabel("Air temp. (°C)", color=C["people"])
    ax2.tick_params(axis="y", colors=C["people"], labelsize=6)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(C["people"])
    panel(ax, "g", x=-0.075, y=0.85)

    for ax in axes[1:]:
        _phases(ax, t0)
    axes[-1].set_xlabel(f"Time from baseline onset (min), {SUB} {SES}; grey bands: baseline and questionnaires")
    axes[-1].set_xlim(-2, m(t1) + 2)
    save(fig, "fig2_example_session")
