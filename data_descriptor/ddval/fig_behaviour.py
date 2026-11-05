"""Figure 4: behavioural data — expected task effects as implementation checks, RT distributions, response strategies."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import C, DOUBLE, NUM, TASK_LABEL, TASKS, behaviour, panel, save

RNG = np.random.default_rng(20250624)
SHORT = {"nbackletter": "n-back\nletters", "nbackdigit": "n-back\ndigits", "stroopword": "Word\nStroop", "stroopemoji": "Emoji\nStroop",
         "visualcount": "Visual\ncount", "visualshape": "Shape\ncomp.", "flankerarrow": "Arrow\nflanker", "flankerdigit": "Digit\nflanker"}




def boot_ci(x: np.ndarray, n: int = 5000) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    m = RNG.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def _participant_lines(ax, d: pd.DataFrame, xcol: str, ycol: str, xs: list, color: str, offset: float = 0.0):
    p = d.groupby(["participant_id", xcol])[ycol].mean().unstack(xcol).reindex(columns=xs)
    xi = np.arange(len(xs)) + offset
    for _, row in p.iterrows():
        ax.plot(xi, row.to_numpy(), color=color, alpha=0.25, lw=0.5, marker="o", ms=1.4)
    mean = p.mean()
    ci = np.array([boot_ci(p[c].dropna()) for c in xs])
    ax.errorbar(xi, mean, yerr=[mean - ci[:, 0], ci[:, 1] - mean], color=color, lw=1.1, marker="o", ms=3, capsize=1.5,
                zorder=5)
    return p


def make() -> None:
    b = behaviour()
    fig = plt.figure(figsize=(DOUBLE, 108 / 25.4))
    gs = fig.add_gridspec(2, 4, hspace=0.62, wspace=0.55)

    # a, b: n-back load
    nb = b[b.task.isin(["nbackletter", "nbackdigit"])].copy()
    nb["rt_s"] = nb.rt_ms / 1000
    for k, (col, ylab, letter) in enumerate([("correct", "Accuracy", "a"), ("rt_s", "Median RT (s)", "b")]):
        ax = fig.add_subplot(gs[0, k])
        for task, color, off in [("nbackletter", C["cool"], -0.08), ("nbackdigit", C["warm"], 0.08)]:
            d = nb[nb.task == task]
            if col == "rt_s":
                d = d[d.correct == 1].groupby(["participant_id", "session_id", "level"]).rt_s.median().reset_index()
            p = _participant_lines(ax, d, "level", col, [1, 2, 3], color, off)
            if col == "rt_s":
                NUM.set("NbackRT" + {"nbackletter": "Letter", "nbackdigit": "Digit"}[task] + "One", p[1].mean(), "{:.2f}")
                NUM.set("NbackRT" + {"nbackletter": "Letter", "nbackdigit": "Digit"}[task] + "Three", p[3].mean(), "{:.2f}")
                NUM.set("NbackRTIncrease" + {"nbackletter": "Letter", "nbackdigit": "Digit"}[task], int((p[3] > p[1]).sum()))
                NUM.set("NbackRTN" + {"nbackletter": "Letter", "nbackdigit": "Digit"}[task], int(p[[1, 3]].dropna().shape[0]))
            else:
                NUM.set("NbackAcc" + {"nbackletter": "Letter", "nbackdigit": "Digit"}[task] + "One", 100 * p[1].mean(), "{:.1f}")
                NUM.set("NbackAcc" + {"nbackletter": "Letter", "nbackdigit": "Digit"}[task] + "Three", 100 * p[3].mean(), "{:.1f}")
        ax.set_xticks([0, 1, 2], ["1", "2", "3"])
        n3 = nb[nb.level == 3].participant_id.nunique()
        ax.text(2, 1.0, f"n = {n3}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5)
        ax.set_xlabel("n-back level")
        ax.set_ylabel(ylab)
        if k == 0:
            ax.text(0.03, 0.08, "letters", color=C["cool"], transform=ax.transAxes, fontsize=5.5)
            ax.text(0.03, 0.18, "digits", color=C["warm"], transform=ax.transAxes, fontsize=5.5)
        panel(ax, letter, x=-0.42)

    # c: flanker congruency (RT of correct trials, per participant)
    ax = fig.add_subplot(gs[0, 2:])
    conds = ["Congruent", "Neutral", "Incongruent"]
    for task, color, off, x0 in [("flankerarrow", C["cool"], 0, 0), ("flankerdigit", C["warm"], 0, 3.6)]:
        d = b[(b.task == task) & (b.correct == 1)]
        d = d.groupby(["participant_id", "session_id", "condition"]).rt_ms.median().reset_index()
        p = d.groupby(["participant_id", "condition"]).rt_ms.mean().unstack().reindex(columns=conds) / 1000
        xi = np.arange(3) + x0
        for _, row in p.iterrows():
            ax.plot(xi, row.to_numpy(), color=color, alpha=0.25, lw=0.5, marker="o", ms=1.4)
        ci = np.array([boot_ci(p[c].dropna()) for c in conds])
        ax.errorbar(xi, p.mean(), yerr=[p.mean() - ci[:, 0], ci[:, 1] - p.mean()], color=color, lw=1.1, marker="o", ms=3, capsize=1.5)
        diff = (p["Incongruent"] - p["Congruent"]) * 1000
        lo, hi = boot_ci(diff.dropna())
        name = "Arrow" if task == "flankerarrow" else "Digit"
        NUM.set(f"Flanker{name}Effect", diff.mean(), "{:.0f}")
        NUM.set(f"Flanker{name}EffectLo", lo, "{:.0f}")
        NUM.set(f"Flanker{name}EffectHi", hi, "{:.0f}")
        NUM.set(f"Flanker{name}Positive", int((diff > 0).sum()))
        ax.text(x0 + 1, ax.get_ylim()[1] if False else 0.02, "", transform=ax.get_xaxis_transform())
        ax.text(x0 + 1, 1.02, TASK_LABEL[task], color=color, ha="center", fontsize=6, transform=ax.get_xaxis_transform())
    ax.set_xticks([0, 1, 2, 3.6, 4.6, 5.6], ["Cong.", "Neutr.", "Incong."] * 2)
    ax.set_ylabel("Median RT, correct (s)")
    panel(ax, "c", x=-0.2)

    # d: RT distributions per task
    ax = fig.add_subplot(gs[1, :2])
    order = TASKS
    data = [np.log10(b[(b.task == t)].rt_ms.dropna() / 1000) for t in order]
    vp = ax.violinplot(data, positions=range(len(order)), widths=0.8, showextrema=False, showmedians=True)
    for body, t in zip(vp["bodies"], order):
        body.set_facecolor(C["sky"] if t in TASKS[:4] else C["warm"])
        body.set_alpha(0.6)
        body.set_edgecolor("none")
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(0.7)
    ax.set_xticks(range(len(order)), [SHORT[t] for t in order], fontsize=5)
    ax.set_yticks(np.log10([0.3, 1, 3, 10, 30, 100]), ["0.3", "1", "3", "10", "30", "100"])
    ax.set_ylabel("RT (s, log scale)")
    ax.set_ylim(np.log10(0.2), np.log10(150))
    ax.text(0.01, 0.95, "round 1", color=C["sky"], transform=ax.transAxes, fontsize=5.5, va="top")
    ax.text(0.12, 0.95, "round 2", color=C["warm"], transform=ax.transAxes, fontsize=5.5, va="top")
    panel(ax, "d", x=-0.1)

    # e: accuracy per session and task, with the response-strategy sessions highlighted
    ax = fig.add_subplot(gs[1, 2:])
    acc = b[b.task.isin(TASKS)].groupby(["task", "participant_id", "session_id"]).correct.mean().reset_index()
    e = b[b.task == "stroopemoji"].copy()
    e["word"] = e.response == e.word
    strat = e.groupby(["participant_id", "session_id"]).agg(w=("word", "mean"), a=("correct", "mean"))
    strat = set(strat[(strat.w >= 0.2) & ((strat.w + strat.a).round(3) == 1)].index)
    f = b[b.task == "flankerdigit"]
    rev = (f.response != f.expected_response).groupby([f.participant_id, f.session_id]).mean()
    rev = set(rev[rev >= 0.5].index)
    for i, t in enumerate(order):
        d = acc[acc.task == t]
        xs = i + RNG.uniform(-0.18, 0.18, len(d))
        flag = [(r.participant_id, r.session_id) in (strat if t == "stroopemoji" else rev if t == "flankerdigit" else set())
                for r in d.itertuples()]
        ax.scatter(xs, d.correct, s=4, c=[C["red"] if fl else C["grey"] for fl in flag], lw=0, zorder=3)
    ax.set_xticks(range(len(order)), [SHORT[t] for t in order], fontsize=5)
    ax.set_ylabel("Accuracy per session")
    ax.set_ylim(-0.03, 1.03)
    ax.text(0.99, 0.04, "red: response strategy (word naming / reversed mapping)", color=C["red"], fontsize=5,
            transform=ax.transAxes, ha="right")
    panel(ax, "e", x=-0.13)
    save(fig, "fig4_behaviour")

    for task, name in [("nbackletter", "Letter"), ("nbackdigit", "Digit")]:
        d = nb[nb.task == task]
        for lvl, word in [(2, "Two"), (3, "Three")]:
            NUM.set(f"NbackReach{name}{word}Sessions", d[d.level == lvl].groupby(["participant_id", "session_id"]).ngroups)
            NUM.set(f"NbackReach{name}{word}Participants", d[d.level == lvl].participant_id.nunique())
    vc = b[b.task == "visualcount"]
    NUM.set("CountAccNominal", 100 * vc.correct.mean(), "{:.0f}")
    NUM.set("EmojiStrategySessions", len(strat))
    NUM.set("RTFlagFast", int((b.rt_flag == "fast_lt150ms").sum()))
    NUM.set("RTFlagSlow", int(b.rt_flag.str.startswith("slow").sum()))
