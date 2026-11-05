"""Figure 1: measurement set-up (a), session protocol with observed durations (b) and counterbalancing (c)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .common import C, DOUBLE, NUM, TASK_LABEL, panel, save, sessions, timeline, tsv


def _box(ax, x, y, w, h, text, fc="white", ec="black", fs=6, bold=False, lw=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            linespacing=1.15)


def _arrow(ax, a, b, color="black", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=6, lw=0.6, color=color, ls=ls,
                                 shrinkA=0, shrinkB=0))


def setup_diagram(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fs = 5.3
    ax.add_patch(FancyBboxPatch((0.0, 0.0), 0.6, 1.0, boxstyle="round,pad=0,rounding_size=0.01",
                                fc="#F7F7F7", ec=C["grey"], lw=0.6))
    ax.text(0.015, 0.975, "Field laboratory (DTU)", fontsize=5.8, fontweight="bold", va="top")
    _box(ax, 0.02, 0.80, 0.56, 0.10, "Tunable-white LED luminaires\nwarm (2014 K) or cool (6250 K) preset", fc="#FFF4D6", fs=fs)
    _box(ax, 0.02, 0.63, 0.27, 0.13, "FLIR AX8 thermal\ncamera, both seats", fc="#FBE3D6", fs=fs)
    _box(ax, 0.31, 0.63, 0.27, 0.13, "ATMOCUBE room\nmonitor, 1-min", fc="#E3F1E9", fs=fs)
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.56, 0.57, boxstyle="round,pad=0,rounding_size=0.01",
                                fc="white", ec="black", lw=0.6, ls="--"))
    ax.text(0.035, 0.575, "Per participant (seat L, R or C)", fontsize=5.6, fontweight="bold", va="top")
    _box(ax, 0.04, 0.29, 0.25, 0.20, "Muse S EEG\nTP9 AF7 AF8 TP10\n256 Hz", fc="#DCEBF7", fs=fs)
    _box(ax, 0.31, 0.29, 0.25, 0.20, "ActiGraph LEAP\nPPG, acceleration,\nskin temperature", fc="#DCEBF7", fs=fs)
    _box(ax, 0.04, 0.05, 0.25, 0.20, "Laptop, browser\ntask platform\ntasks, T0-T3", fc="#EDE3F3", fs=fs)
    _box(ax, 0.31, 0.05, 0.25, 0.20, "ANC headphones\nbabble or\nroad traffic", fc="#EDE3F3", fs=fs)
    xs, w = 0.67, 0.33
    _box(ax, xs, 0.78, w, 0.16, "Raw archive\n(read-only, SHA-256)", fs=fs, bold=True)
    _box(ax, xs, 0.40, w, 0.28, "Curation pipeline\nde-duplication,\nre-scoring, coding,\nlinkage, timeline, QC", fc="#F2F2F2", fs=fs)
    _box(ax, xs, 0.04, w, 0.26, "BIDS 1.10 dataset\nbeh, eeg, physio,\nphenotype, environment,\nderivatives", fs=fs, bold=True)
    _arrow(ax, (0.6, 0.86), (xs, 0.86))
    _arrow(ax, (xs + w / 2, 0.78), (xs + w / 2, 0.68))
    _arrow(ax, (xs + w / 2, 0.40), (xs + w / 2, 0.30))


def protocol_durations() -> tuple[pd.DataFrame, pd.DataFrame]:
    t = timeline()
    phases = t[t.phase.isin(["baseline", "T1", "round1", "T2", "round2", "T3"])]
    dur = phases.groupby("phase").duration_s.median() / 60
    blocks = t[t.phase == "block"]
    bdur = blocks.groupby(["round", "task"]).duration_s.median() / 60
    return dur, bdur


def protocol_bar(ax) -> None:
    dur, bdur = protocol_durations()
    s = sessions()
    total = (s.session_end - s.session_start).dt.total_seconds() / 60
    order = [("Baseline", dur["baseline"]), ("T1", dur["T1"]), ("Round 1", dur["round1"]),
             ("T2", dur["T2"]), ("Round 2", dur["round2"]), ("T3", dur["T3"])]
    grey, y, h = "#D9D9D9", 0.62, 0.30
    ax.barh(y, 3.0, left=-3.0, height=h, color="white", ec="black", lw=0.5, hatch="//////")
    ax.text(-1.5, y + h / 2 + 0.04, "T0", ha="center", va="bottom", fontsize=5.5)
    x = 0.0
    for name, d in order:
        rnd = name.startswith("Round")
        ax.barh(y, d, left=x, height=h, color=C["sky"] if rnd else ("#BDBDBD" if name == "Baseline" else grey),
                ec="black", lw=0.5)
        if d >= 4:
            ax.text(x + d / 2, y, f"{name}\n{d:.1f} min", ha="center", va="center", fontsize=5.5)
        else:
            ax.text(x + d / 2, y + h / 2 + 0.04, f"{name}\n{d:.1f}", ha="center", va="bottom", fontsize=5.2)
        if rnd:
            r = 1 if name == "Round 1" else 2
            tasks = ["nbackletter", "stroopword", "visualcount", "flankerarrow"] if r == 1 else \
                    ["nbackdigit", "stroopemoji", "visualshape", "flankerdigit"]
            short = {"nbackletter": "n-back\nletters", "stroopword": "Word\nStroop", "visualcount": "Visual\ncounting",
                     "flankerarrow": "Arrow\nflanker", "nbackdigit": "n-back\ndigits", "stroopemoji": "Emoji\nStroop",
                     "visualshape": "Shape\ncomp.", "flankerdigit": "Digit\nflanker"}
            b = np.array([bdur.get((float(r), k), np.nan) for k in tasks])
            xb = x
            for k, bw in zip(tasks, b):
                ax.barh(y - 0.36, bw, left=xb, height=0.24, color="white", ec=C["cool"], lw=0.5)
                ax.text(xb + bw / 2, y - 0.36, short[k], ha="center", va="center", fontsize=4.7)
                xb += bw
            ax.text(x + d / 2, y + h / 2 + 0.04, f"soundscape {r}", ha="center", va="bottom", fontsize=5.5,
                    color=C["people"], style="italic")
        x += d
    ax.set_xlim(-3.4, x + 0.4)
    ax.set_ylim(0.05, 1.05)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks(np.arange(0, x + 1, 5))
    ax.set_xlabel("Time from baseline onset (min; median phase durations over 20 sessions; block widths are median task durations)")
    NUM.set("SessionMedianMin", np.median(total), "{:.0f}")
    NUM.set("SessionMinMin", total.min(), "{:.0f}")
    NUM.set("SessionMaxMin", total.max(), "{:.0f}")
    for k, name in [("baseline", "Baseline"), ("round1", "RoundOne"), ("round2", "RoundTwo"), ("T1", "TOne"), ("T2", "TTwo"), ("T3", "TThree")]:
        NUM.set(f"Dur{name}Min", dur[k], "{:.1f}")


SND = {"people": "B", "traffic": "T"}


def counterbalancing(ax) -> None:
    p = tsv("participants.tsv")
    s = sessions().set_index(["participant_id", "session_id"])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    xs = [0.02, 0.30, 0.47, 0.62, 0.82]
    for x, c in zip(xs, ["Participant", "Group", "Seat", "Ses. A", "Ses. B"]):
        ax.text(x, 0.985, c, fontsize=5.6, fontweight="bold", va="top")
    for i, r in enumerate(p.itertuples()):
        y = 0.885 - i * 0.082
        for x, v in zip(xs[:3], (r.participant_id, r.group, r.seat)):
            ax.text(x, y, v, fontsize=5.5, va="center")
        for j, ses in enumerate(["ses-A", "ses-B"]):
            q = s.loc[(r.participant_id, ses)]
            ax.add_patch(plt.Rectangle((xs[3 + j], y - 0.028), 0.05, 0.056, color=C[q.light], lw=0))
            ax.text(xs[3 + j] + 0.065, y, f"{SND[q.soundscape_round1]}→{SND[q.soundscape_round2]}", fontsize=5.5, va="center")
    ax.add_patch(plt.Rectangle((0.02, 0.025), 0.04, 0.045, color=C["warm"], lw=0))
    ax.text(0.07, 0.047, "warm", fontsize=5.2, va="center")
    ax.add_patch(plt.Rectangle((0.2, 0.025), 0.04, 0.045, color=C["cool"], lw=0))
    ax.text(0.25, 0.047, "cool", fontsize=5.2, va="center")
    ax.text(0.37, 0.047, "B babble, T road traffic", fontsize=5.2, va="center")


def make() -> None:
    fig = plt.figure(figsize=(DOUBLE, 125 / 25.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1.6, 0.85], hspace=0.28, wspace=0.10)
    ax_a = fig.add_subplot(gs[0, 0])
    setup_diagram(ax_a)
    panel(ax_a, "a", x=0.0, y=0.98)
    ax_c = fig.add_subplot(gs[0, 1])
    counterbalancing(ax_c)
    panel(ax_c, "c", x=-0.04, y=1.0)
    ax_b = fig.add_subplot(gs[1, :])
    protocol_bar(ax_b)
    panel(ax_b, "b", x=-0.02, y=1.0)
    save(fig, "fig1_design")
