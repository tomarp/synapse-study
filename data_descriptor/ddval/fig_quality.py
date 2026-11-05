"""Figure 3: completeness and quality of every data stream per session."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .common import BIDS, C, CHANNELS, DOUBLE, GEN, NUM, SESSIONS, behaviour, panel, save, tsv
from .physio import wrist_coverage

LINK_ORDER = {"single_participant": 3, "causally_unique": 3, "moderate": 2, "weak": 1}


def quality_table(eeg_quality: pd.DataFrame) -> pd.DataFrame:
    keys = pd.DataFrame(SESSIONS, columns=["participant_id", "session_id"])
    eq = tsv("derivatives/quality_control/eeg_quality.tsv")[["participant_id", "session_id", "session_coverage_fraction"]]
    us = eeg_quality[eeg_quality.scope == "recording"].pivot_table(index=["participant_id", "session_id"], columns="channel",
                                                                     values="usable_fraction").reset_index()
    us = us.rename(columns={c: f"usable_{c}" for c in CHANNELS})
    th = tsv("derivatives/facial_temperature/thermal_coverage.tsv")
    th["face_valid"] = th.n_valid / th.n_snapshots
    links = tsv("derivatives/quality_control/questionnaire_linkage.tsv")
    lk = links.pivot_table(index=["participant_id", "session_id"], columns="questionnaire", values="link_quality",
                           aggfunc="first").reset_index()
    b = behaviour().groupby(["participant_id", "session_id"]).size().rename("trials").reset_index()
    q = (keys.merge(b, how="left").merge(eq, how="left").merge(us, how="left").merge(wrist_coverage(), how="left")
         .merge(th[["participant_id", "session_id", "face_valid", "baseline_usable"]], how="left").merge(lk, how="left"))
    q.to_csv(GEN / "session_quality.csv", index=False)
    return q


def make(eeg_quality: pd.DataFrame) -> pd.DataFrame:
    q = quality_table(eeg_quality)
    cols = [("session_coverage_fraction", "EEG\ncoverage"), *[(f"usable_{c}", f"EEG {c}\nusable") for c in CHANNELS],
            ("ppg_coverage", "Wrist PPG\ncoverage"), ("face_valid", "Face\nvalid")]
    M = q[[c for c, _ in cols]].to_numpy(float)
    cmap = LinearSegmentedColormap.from_list("q", ["#B2182B", "#F4A582", "#F7F7F7", "#92C5DE", "#2166AC"])
    fig = plt.figure(figsize=(DOUBLE, 112 / 25.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 4.2, 1.15], wspace=0.05)
    labels = [f"{a}/{b}" for a, b in zip(q.participant_id, q.session_id)]
    y = np.arange(len(q))

    ax0 = fig.add_subplot(gs[0])
    ax0.barh(y, q.trials, color=C["sky"], height=0.7)
    ax0.set_yticks(y, labels, fontsize=5.5)
    ax0.invert_yaxis()
    ax0.set_xlabel("Trials")
    ax0.set_title("Analysable trials", fontsize=6)
    panel(ax0, "a", x=-0.9, y=1.01)

    ax = fig.add_subplot(gs[1], sharey=ax0)
    nr, nc = M.shape
    im = ax.pcolormesh(np.arange(nc + 1) - 0.5, np.arange(nr + 1) - 0.5, np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1,
                       edgecolors="white", linewidth=0.3)
    for i in range(nr):
        for j in range(nc):
            v = M[i, j]
            ax.text(j, i, "–" if np.isnan(v) else f"{100 * v:.0f}", ha="center", va="center", fontsize=4.8,
                    color="white" if (not np.isnan(v) and (v < 0.15 or v > 0.85)) else "black")
    ax.set_xticks(range(nc), [l for _, l in cols], fontsize=5.5)
    ax.xaxis.tick_top()
    ax.set_xlim(-0.5, nc - 0.5)
    ax.tick_params(axis="y", left=False, labelleft=False)
    for i, ok in enumerate(q.baseline_usable):
        if not ok:
            ax.text(nc - 0.5 + 0.08, i, "†", ha="left", va="center", fontsize=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cax = ax.inset_axes([0.0, -0.075, 1.0, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("% of session (EEG and PPG coverage; usable 2-s EEG epochs) or of thermal snapshots (face valid); † no usable thermal baseline",
                 fontsize=5.3)
    cb.ax.tick_params(labelsize=5)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "25", "50", "75", "100"])
    panel(ax, "b", x=-0.02, y=1.075)

    from matplotlib.lines import Line2D
    ax2 = fig.add_subplot(gs[2], sharey=ax0)
    style = {3: (C["cool"], C["cool"]), 2: (C["warm"], "white"), 1: (C["red"], "white")}
    for j, qn in enumerate(["T1", "T2", "T3"]):
        for i, v in enumerate(q[qn]):
            col, face = style[LINK_ORDER[v]]
            ax2.plot(j, i, marker="o", color=col, mfc=face, ms=3.2, mew=0.8)
    ax2.set_xticks([0, 1, 2], ["T1", "T2", "T3"], fontsize=5.5)
    ax2.xaxis.tick_top()
    ax2.set_xlim(-0.6, 2.6)
    ax2.tick_params(axis="y", left=False, labelleft=False)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    handles = [Line2D([], [], marker="o", ls="", color=c, mfc=f, ms=3.2, mew=0.8, label=l)
               for (c, f), l in [(style[3], "unique"), (style[2], "moderate"), (style[1], "weak")]]
    ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), fontsize=5, handletextpad=0.1,
               title="Questionnaire link", title_fontsize=5.3)
    panel(ax2, "c", x=-0.05, y=1.075)
    save(fig, "fig3_quality")

    rec = eeg_quality[eeg_quality.scope == "recording"]
    base = eeg_quality[eeg_quality.scope == "baseline"]
    for ch in CHANNELS:
        NUM.set(f"UsableRec{ch}", 100 * rec[rec.channel == ch].usable_fraction.median(), "{:.0f}")
        NUM.set(f"UsableBase{ch}", 100 * base[base.channel == ch].usable_fraction.median(), "{:.0f}")
    NUM.set("EEGCovMin", 100 * q.session_coverage_fraction.min(), "{:.0f}")
    NUM.set("EEGCovFull", int((q.session_coverage_fraction >= 0.99).sum()))
    NUM.set("FaceValidMin", 100 * q.face_valid.min(), "{:.0f}")
    NUM.set("FaceValidMax", 100 * q.face_valid.max(), "{:.0f}")
    NUM.set("PPGCovMedian", 100 * q.loc[q.ppg_coverage > 0, "ppg_coverage"].median(), "{:.0f}")
    links = tsv("derivatives/quality_control/questionnaire_linkage.tsv")
    NUM.set("LinksTotal", len(links))
    NUM.set("LinksUnique", int(links.link_quality.isin(["causally_unique", "single_participant"]).sum()))
    NUM.set("LinksModerate", int((links.link_quality == "moderate").sum()))
    NUM.set("LinksWeak", int((links.link_quality == "weak").sum()))
    return q
