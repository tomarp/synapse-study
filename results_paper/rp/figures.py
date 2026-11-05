"""Figures and LaTeX tables of the results paper, drawn from ``generated/`` (and the cached fits for the posterior
predictive checks). Design figures use the true allocation (it is part of the published design, not an outcome)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from . import data, models
from .common import (AFFECT, C, DOUBLE, GEN, ITEM_LABEL, NUM, ONEHALF, PERCEPTION, SINGLE, SUBS, TASK_LABEL, TASKS,
                     panel, save, style)

FACTOR = {"bL": ("Lighting: cool − warm", C["cool"]), "bS": ("Soundscape: traffic − babble", C["traffic"]),
          "dE_L": ("Lighting: cool − warm", C["cool"]), "dE_S": ("Soundscape: traffic − babble", C["traffic"])}


def _res() -> pd.DataFrame:
    return pd.read_csv(GEN / "results.tsv", sep="\t")


def _forest(ax, rows: pd.DataFrame, labels: list[str], terms: list[str], rope=None, log=False, xlabel="") -> None:
    """Rows of estimates (median, 95 % CrI); filled marker when the CrI excludes zero (one on a ratio scale)."""
    off = np.linspace(-0.18, 0.18, len(terms)) if len(terms) > 1 else [0.0]
    null = 1.0 if log else 0.0
    if rope is not None:
        ax.axvspan(*rope, color="0.92", lw=0, zorder=0)
    ax.axvline(null, color="0.4", lw=0.6, zorder=1)
    for j, term in enumerate(terms):
        lab, col = FACTOR[term]
        for i, key in enumerate(labels):
            r = rows[(rows.key == key) & (rows.term == term)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = i + off[j]
            ax.plot([r.lo, r.hi], [y, y], color=col, lw=1.1, solid_capstyle="butt", zorder=2)
            filled = r.verdict == "credible"
            ax.plot(r["median"], y, "o", ms=3.6, mec=col, mfc=col if filled else "white", mew=0.9, zorder=3,
                    label=lab if i == 0 else None)
    ax.set_yticks(range(len(labels)))
    ax.set_ylim(len(labels) - 0.5, -0.5)
    if log:
        ax.set_xscale("log")
        ticks = [t for t in (0.25, 0.5, 1, 2, 4, 8) if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel(xlabel)


# ----------------------------------------------------------------------------------------------- figure 1
def fig_design() -> None:
    s = data.sessions_true()
    t = data.timeline()
    med = t.groupby("phase").duration_s.median() / 60
    fig = plt.figure(figsize=(DOUBLE, 72 / 25.4))
    ax = fig.add_axes([0.04, 0.74, 0.94, 0.19])
    seg = [("T0", 1.5, "0.85"), ("Baseline\n(5 min)", med["baseline"], "0.75"), ("T1", med["T1"], "0.85"),
           ("Round 1 · soundscape 1\nn-back (letters) · word Stroop\nvisual counting · arrow flanker", med["round1"], C["sky"]),
           ("T2", med["T2"], "0.85"),
           ("Round 2 · soundscape 2\nn-back (digits) · emoji Stroop\nshape comparison · digit flanker", med["round2"], "#9AD0EC"),
           ("T3", med["T3"], "0.85")]
    x = 0
    for lab, w, col in seg:
        ax.add_patch(Rectangle((x, 0), w, 1, fc=col, ec="white", lw=1))
        ax.text(x + w / 2, 0.5, lab, ha="center", va="center", fontsize=5.6 if w > 4 else 5.2)
        x += w
    ax.set_xlim(0, x)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time from session start (min; median phase durations)")
    for sp in ("left",):
        ax.spines[sp].set_visible(False)
    panel(ax, "a", x=-0.035, y=1.08)
    ax.set_title("One session (two per participant, same day, 10–15 min apart; lighting differs between the sessions)",
                 fontweight="normal", fontsize=6, loc="left", x=0.0)

    ax = fig.add_axes([0.06, 0.07, 0.46, 0.45])
    for i, p in enumerate(SUBS):
        for j in (1, 2):
            x0 = s[(s.participant_id == p) & (s.session_order == j)].iloc[0]
            ax.add_patch(Rectangle((j - 1 + 0.03, i + 0.1), 0.94, 0.8, fc=C[x0.light], ec="none", alpha=0.85))
            ax.plot([j - 0.5, j - 0.5], [i + 0.1, i + 0.9], color="white", lw=1.2)
            for k, rr in enumerate((1, 2)):
                snd = x0[f"soundscape_round{rr}"]
                ax.text(j - 1 + 0.27 + 0.46 * k, i + 0.5, "B" if snd == "people" else "T", ha="center", va="center",
                        fontsize=5.5, color="white", fontweight="bold")
    ax.set_xlim(0, 2)
    ax.set_ylim(10, 0)
    ax.set_yticks(np.arange(10) + 0.5, [p.replace("sub-", "") for p in SUBS])
    ax.set_xticks([0.5, 1.5], ["Session 1 (round 1 | round 2)", "Session 2 (round 1 | round 2)"])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, fc=C["warm"]), Rectangle((0, 0), 1, 1, fc=C["cool"])],
              labels=["warm, 2014 K", "cool, 6250 K"], ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    panel(ax, "b", x=-0.13, y=1.02)
    ax.set_title("Allocation (B, cafeteria babble; T, road traffic)", fontweight="normal", fontsize=6, loc="left")

    ax = fig.add_axes([0.58, 0.07, 0.40, 0.45])
    ax.axis("off")
    panel(ax, "c", x=-0.07, y=1.02)
    rows = [("Effect", "Contrast", "Free of"),
            ("Soundscape → ratings,\nphysiology", "rounds within a session,\norder reversed between sessions", "order, lighting"),
            ("Soundscape → performance", "round-1 vs round-2 task sets\nwithin participant", "order, lighting"),
            ("Lighting → all outcomes", "sessions within participant", "soundscape;\norder only between\nparticipants (6 vs 4)"),
            ("Lighting × soundscape", "between pairing groups (5 vs 4)", "weakly identified")]
    for i, r in enumerate(rows):
        y = 0.94 - i * 0.2
        for k, (xx, txt) in enumerate(zip((0.0, 0.36, 0.76), r)):
            ax.text(xx, y, txt, fontsize=5.4, va="top", fontweight="bold" if i == 0 else "normal")
    ax.set_title("What identifies each effect", fontweight="normal", fontsize=6, loc="left")
    save(fig, "fig1_design")


# ----------------------------------------------------------------------------------------------- figure 2
def fig_conditions() -> None:
    ph = data.physiology()
    vars_ = [("room_air_temperature", "Air temperature (°C)"), ("room_relative_humidity", "Relative humidity (%)"),
             ("room_co2", "CO$_2$ (ppm)"), ("room_sound_level", "Room sound level (dB(A))"),
             ("room_illuminance", "Room illuminance (lx, uncal.)")]
    fig, axs = plt.subplots(1, 5, figsize=(DOUBLE, 42 / 25.4), constrained_layout=True)
    rng = np.random.default_rng(1)
    groups = [("warm", ph.L < 0, C["warm"]), ("cool", ph.L > 0, C["cool"]), ("babble", ph.S < 0, C["people"]),
              ("traffic", ph.S > 0, C["traffic"])]
    for ax, (v, lab), letter in zip(axs, vars_, "abcde"):
        for i, (g, sel, col) in enumerate(groups):
            y = ph.loc[sel, v].dropna()
            ax.plot(i + rng.uniform(-0.15, 0.15, len(y)), y, "o", ms=2, color=col, alpha=0.7, mew=0)
            ax.plot([i - 0.25, i + 0.25], [y.mean()] * 2, color="k", lw=1)
        ax.set_xticks(range(4), ["warm", "cool", "babble", "traffic"], rotation=40)
        ax.set_title(lab, fontweight="normal")
        panel(ax, letter, x=-0.3)
    save(fig, "fig2_conditions")


# ----------------------------------------------------------------------------------------------- figure 3
def fig_performance(res: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 62 / 25.4), constrained_layout=True)
    keys = ["all"] + TASKS
    labels = ["All tasks (pooled)"] + [TASK_LABEL[t] for t in TASKS]
    for ax, model, rope, log, xl, letter in (
            (axs[0], "A1", (100 * (np.exp(-0.02) - 1), 100 * (np.exp(0.02) - 1)), False, "Change in response time (%)", "a"),
            (axs[1], "A2", (np.exp(-0.2), np.exp(0.2)), True, "Odds ratio of a correct response", "b")):
        r = res[(res.outcome == model) & res.analysis.isin(["primary", "task"]) & res.term.isin(["bL", "bS"])].copy()
        r["key"] = np.where(r.analysis == "primary", "all", r.task)
        _forest(ax, r, keys, ["bL", "bS"], rope=rope, log=log, xlabel=xl)
        ax.set_yticklabels(labels if ax is axs[0] else [""] * len(labels))
        ax.axhline(0.5, color="0.7", lw=0.5)
        panel(ax, letter, x=-0.1 if ax is axs[1] else -0.3)
    axs[0].set_title("Response speed (correct trials)", fontweight="normal")
    axs[1].set_title("Accuracy", fontweight="normal")
    axs[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
    save(fig, "fig3_performance")


# ----------------------------------------------------------------------------------------------- figure 4
def fig_perception(res: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 70 / 25.4), constrained_layout=True, sharex=True)
    r = res[(res.family == "B") & (res.analysis == "primary") & res.term.isin(["dE_L", "dE_S"])].copy()
    lat = res[(res.family == "B") & (res.analysis == "primary") & res.term.isin(["bL", "bS"])]
    # verdict from the latent effect (ANALYSIS_PLAN.md, section 5)
    v = {(o, t.replace("b", "dE_")): x for o, t, x in zip(lat.outcome, lat.term, lat.verdict)}
    r["verdict"] = [v[(o, t)] for o, t in zip(r.outcome, r.term)]
    r["key"] = r.outcome
    for ax, term, check, letter in ((axs[0], "dE_L", "light", "a"), (axs[1], "dE_S", "acoustic", "b")):
        for i, it in enumerate(PERCEPTION):
            if it.startswith(check):
                ax.axhspan(i - 0.5, i + 0.5, color="#FFF4D6", lw=0, zorder=0)
        _forest(ax, r, PERCEPTION, [term], xlabel="Difference in mean rating (scale points)")
        ax.set_yticklabels([ITEM_LABEL[i] for i in PERCEPTION] if ax is axs[0] else [""] * len(PERCEPTION))
        for b in (1.5, 3.5, 5.5):
            ax.axhline(b, color="0.75", lw=0.5)
        ax.set_title(FACTOR[term][0], fontweight="normal", color=FACTOR[term][1])
        for i, it in enumerate(PERCEPTION):
            x = r[(r.key == it) & (r.term == term)]
            if len(x):
                ax.text(1.0, i, f"pd {x.iloc[0].pd:.2f}", transform=ax.get_yaxis_transform(), fontsize=5, va="center",
                        ha="left", color="0.35")
        panel(ax, letter, x=-0.1 if ax is axs[1] else -0.55)
    save(fig, "fig4_perception")


# ----------------------------------------------------------------------------------------------- figure 5
def fig_affect_physiology(res: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(DOUBLE, 78 / 25.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1, 1], hspace=0.9, wspace=0.6, left=0.19, right=0.98, top=0.88, bottom=0.14)
    ax = fig.add_subplot(gs[:, 0])
    r = res[(res.family == "C") & (res.analysis == "primary") & res.term.isin(["dE_L", "dE_S"])].copy()
    lat = res[(res.family == "C") & (res.analysis == "primary") & res.term.isin(["bL", "bS"])]
    v = {(o, t.replace("b", "dE_")): x for o, t, x in zip(lat.outcome, lat.term, lat.verdict)}
    r["verdict"] = [v[(o, t)] for o, t in zip(r.outcome, r.term)]
    r["key"] = r.outcome
    _forest(ax, r, AFFECT, ["dE_L", "dE_S"], xlabel="Difference in mean rating (points)")
    ax.set_yticklabels([ITEM_LABEL[i] for i in AFFECT])
    ax.set_title("Affect and workload (T2, T3)", fontweight="normal")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.6, 1.04), ncol=2)
    panel(ax, "a", x=-0.75)
    phys = [("hr", "Heart rate (bpm)", ["bL", "bS"]), ("ln_rmssd", "Resting ln RMSSD", ["bL"]),
            ("skin_temp", "Wrist skin temp. (°C)", ["bL", "bS"]), ("face_temp_bg", "Facial temp. (°C, rel.)", ["bS"])]
    for k, (out, lab, terms) in enumerate(phys):
        ax = fig.add_subplot(gs[k // 2, 1 + k % 2])
        rr = res[(res.family == "D") & (res.analysis == "primary") & (res.outcome == out) & res.term.isin(terms)].copy()
        rr["key"] = rr.term
        for j, term in enumerate(terms):
            x = rr[rr.term == term].iloc[0]
            col = FACTOR[term][1]
            ax.plot([x.lo, x.hi], [j, j], color=col, lw=1.1)
            ax.plot(x["median"], j, "o", ms=3.6, mec=col, mfc=col if x.verdict == "credible" else "white", mew=0.9)
        ax.axvline(0, color="0.4", lw=0.6)
        ax.set_yticks(range(len(terms)), ["lighting" if t == "bL" else "soundscape" for t in terms])
        ax.set_ylim(len(terms) - 0.5, -0.5)
        ax.set_title(lab, fontweight="normal")
        panel(ax, "bcde"[k], x=-0.45)
    save(fig, "fig5_affect_physiology")


# ----------------------------------------------------------------------------------------------- figure 6
def fig_individual(res: pd.DataFrame) -> None:
    pe = pd.read_csv(GEN / "participant_effects_A1.tsv", sep="\t")
    fig, axs = plt.subplots(1, 2, figsize=(ONEHALF, 55 / 25.4), constrained_layout=True, sharey=True)
    for ax, term, letter in ((axs[0], "bL", "a"), (axs[1], "bS", "b")):
        lab, col = FACTOR[term]
        p = pe[pe.term == term].reset_index(drop=True)
        pop = res[(res.outcome == "A1") & (res.analysis == "primary") & (res.term == term)].iloc[0]
        ax.axvspan(pop.lo, pop.hi, color=col, alpha=0.15, lw=0)
        ax.axvline(pop["median"], color=col, lw=0.8)
        ax.axvline(0, color="0.4", lw=0.6)
        for i, x in p.iterrows():
            ax.plot([x.lo, x.hi], [i, i], color=col, lw=1)
            ax.plot(x["median"], i, "o", ms=3.2, mec=col, mfc=col if x.verdict == "credible" else "white", mew=0.9)
        sd = res[(res.outcome == "A1") & (res.analysis == "heterogeneity") & (res.term == ("sdL" if term == "bL" else "sdS"))].iloc[0]
        ax.text(0.02, 0.02, f"between-participant SD {100 * sd['median']:.1f} %\n[{100 * sd.lo:.1f}, {100 * sd.hi:.1f}]",
                transform=ax.transAxes, fontsize=5.2, va="bottom")
        ax.set_yticks(range(len(p)), [s.replace("sub-", "") for s in p.participant_id])
        ax.set_ylim(len(p) - 0.5, -0.5)
        ax.set_xlabel("Change in response time (%)")
        ax.set_title(lab, fontweight="normal", color=col)
        panel(ax, letter, x=-0.2 if ax is axs[0] else -0.05)
    save(fig, "fig6_individual")


# ----------------------------------------------------------------------------------------- supplementary
def fig_sensitivity(res: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 80 / 25.4), constrained_layout=True)
    order = ["primary", "S1", "S2", "S3", "S4", "S5", "S6", "S7 wide", "S7 narrow", "S8"]
    for ax, model, xl in ((axs[0], "A1", "Change in response time (%)"), (axs[1], "A2", "Odds ratio, correct")):
        r = res[(res.outcome == model) & res.term.isin(["bL", "bS"]) & res.analysis.isin(order)].copy()
        r["key"] = r.analysis
        an = [a for a in order if a in set(r.analysis)]
        _forest(ax, r, an, ["bL", "bS"], log=model == "A2", xlabel=xl)
        ax.set_yticklabels(an)
        ax.set_title("Response time" if model == "A1" else "Accuracy", fontweight="normal")
    ax = axs[2]
    r = res[(res.family == "B") & res.term.isin(["bL", "bS"]) & res.analysis.isin(order) & res.outcome.isin(PERCEPTION)].copy()
    rows, labels = [], []
    for it in PERCEPTION:
        for a in [a for a in order if a in set(r[r.outcome == it].analysis)]:
            rows.append(f"{it}|{a}")
            labels.append(f"{ITEM_LABEL[it].split(' (')[0]} · {a}" if a == "primary" else a)
    r["key"] = r.outcome + "|" + r.analysis
    _forest(ax, r, rows, ["bL", "bS"], xlabel="Effect (log-odds)")
    ax.set_yticklabels(labels, fontsize=3.8)
    ax.set_title("Perceived environment", fontweight="normal")
    axs[0].legend(loc="lower left", bbox_to_anchor=(0, 1.05), ncol=2)
    for a, l in zip(axs, "abc"):
        panel(a, l, x=-0.25)
    save(fig, "figS1_sensitivity")


def fig_ppc() -> None:
    t = data.trials()
    base = t[t.task.isin(TASKS) & t.rt_ok & ~t.strategy]
    f1 = models.performance(base[base.correct == 1], "log_rt", name="A1-primary")
    f2 = models.performance(base, "correct", name="A2-primary")
    fig, axs = plt.subplots(2, 3, figsize=(DOUBLE, 90 / 25.4), constrained_layout=True)
    for ax, fit, stat, lab in ((axs[0, 0], f1, "mean", "Mean log RT (centred)"), (axs[0, 1], f1, "sd", "SD of log RT"),
                               (axs[0, 2], f2, "mean", "Accuracy")):
        rep, obs = fit[f"ppc_{stat}"], fit[f"ppc_obs_{stat}"]
        lo, hi = np.percentile(rep, [2.5, 97.5], axis=0)
        x = np.arange(len(TASKS))
        ax.vlines(x, lo, hi, color="0.6", lw=3)
        ax.plot(x, obs, "o", color="k", ms=3)
        ax.set_xticks(x, [TASK_LABEL[k] for k in TASKS], rotation=40, ha="right")
        ax.set_title(lab, fontweight="normal")
    r = data.ratings()
    for k, it in enumerate(["thermal_sensation", "light_assessment", "acoustic_satisfaction"]):
        d = r[r["item"] == it].copy()
        d["y"] = d.value - 1
        fit = models.ordinal(d, 5, "B", name=f"B-{it}-primary")
        rep, obs = fit["ppc_prop"], fit["ppc_obs_prop"]
        lo, hi = np.percentile(rep, [2.5, 97.5], axis=0)
        ax = axs[1, k]
        ax.vlines(np.arange(1, 6), lo, hi, color="0.6", lw=3)
        ax.plot(np.arange(1, 6), obs, "o", color="k", ms=3)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("Rating")
        ax.set_title(ITEM_LABEL[it], fontweight="normal")
    for a, l in zip(axs.flat, "abcdef"):
        panel(a, l, x=-0.2)
    save(fig, "figS2_ppc")


# ------------------------------------------------------------------------------------------------ tables
def _fmt(x, scale: str) -> str:
    f = {"percent": "{:.1f}", "odds ratio": "{:.2f}"}.get(scale, "{:.2f}")
    s = f"{f.format(x['median'])} [{f.format(x.lo)}, {f.format(x.hi)}]"
    return s.replace("-", "\\textminus ")


def tables(res: pd.DataFrame) -> None:
    lines = []
    for model, lab in (("A1", "Response time (\\%)"), ("A2", "Accuracy (odds ratio)")):
        for term in ("bL", "bS"):
            x = res[(res.outcome == model) & (res.analysis == "primary") & (res.term == term)].iloc[0]
            lines.append(f"{lab} & {'Lighting' if term == 'bL' else 'Soundscape'} & {_fmt(x, x.scale)} & {x.pd:.2f} & {x.verdict} \\\\")
    lines.append("\\midrule")
    for it in PERCEPTION:
        for term in ("L", "S"):
            x = res[(res.outcome == it) & (res.analysis == "primary") & (res.term == f"dE_{term}")].iloc[0]
            lt = res[(res.outcome == it) & (res.analysis == "primary") & (res.term == f"b{term}")].iloc[0]
            lab = ITEM_LABEL[it].replace("→", "$\\to$") if term == "L" else ""
            lines.append(f"{lab} & {'Lighting' if term == 'L' else 'Soundscape'} & {_fmt(x, 'rating')} & {lt.pd:.2f} & {lt.verdict} \\\\")
    (GEN / "table_primary.tex").write_text("% Generated by output/code/results_paper; do not edit by hand.\n" + "\n".join(lines) + "\n\\bottomrule\n")
    dev = pd.read_csv(GEN / "deviations.tsv", sep="\t")
    (GEN / "table_deviations.tex").write_text("% Generated by output/code/results_paper; do not edit by hand.\n" + "\n".join(
        f"{x['item']} & {x.change} & {x.reason} \\\\" for _, x in dev.iterrows()).replace("%", "\\%").replace("_", "\\_") + "\n\\bottomrule\n")
    conv = pd.read_csv(GEN / "convergence.tsv", sep="\t")
    NUM.set("NModelsConverged", str(int(((conv.rhat_max <= 1.01) & (conv.ess_bulk_min >= 400) & (conv.divergences == 0)).sum())))


def run() -> None:
    style()
    res = _res()
    fig_design()
    fig_conditions()
    fig_performance(res)
    fig_perception(res)
    fig_affect_physiology(res)
    fig_individual(res)
    fig_sensitivity(res)
    fig_ppc()
    tables(res)
    NUM.write()
