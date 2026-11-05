"""Additional summaries for the manuscript text, added after unblinding.

Nothing here fits a new model or changes an estimate: the functions summarise the posteriors of the registered models
(loaded from the cache) and describe the data, e.g. how strongly the posterior argues against effects of a given size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, models
from .common import GEN, NUM, PERCEPTION, TASKS


def _p(x: float) -> str:
    return f"{x:.2f}"


def run() -> None:
    t = data.trials()
    base = t[t.task.isin(TASKS) & t.rt_ok & ~t.strategy]
    a1 = models.performance(base[base.correct == 1], "log_rt", name="A1-primary")
    a2 = models.performance(base, "correct", name="A2-primary")
    pct = lambda x: 100 * (np.exp(x) - 1)                                        # noqa: E731
    # how strongly the posterior argues against benefits of a practically relevant size
    NUM.set("PCoolFasterTwo", _p((a1["bL"] < np.log(0.98)).mean()))
    NUM.set("PCoolFaster", _p((a1["bL"] < 0).mean()))
    NUM.set("PSoundWithinFour", _p((np.abs(a1["bS"]) < np.log(1.04)).mean()))
    NUM.set("PSoundWithinTwo", _p((np.abs(a1["bS"]) < np.log(1.02)).mean()))
    for term, key in (("sdL", "SdL"), ("sdS", "SdS")):
        lo, med, hi = np.percentile(pct(a1[term]), [2.5, 50, 97.5])
        NUM.set(f"RT{key}", f"{med:.1f}")
        NUM.set(f"RT{key}Lo", f"{lo:.1f}")
        NUM.set(f"RT{key}Hi", f"{hi:.1f}")
    ratio = a1["sdL"] / np.abs(a1["bL"])
    NUM.set("RTHeteroRatio", f"{np.median(ratio):.1f}")
    pe = pd.read_csv(GEN / "participant_effects_A1.tsv", sep="\t")
    pl = pe[pe.term == "bL"]
    slower, faster = pl[(pl.lo > 0)], pl[(pl.hi < 0)]
    words = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    NUM.set("NSlowerCool", words[len(slower)])
    NUM.set("SlowerCoolMin", f"{slower['median'].min():.0f}")
    NUM.set("SlowerCoolMax", f"{slower['median'].max():.0f}")
    NUM.set("NFasterCoolProbable", words[int(((pl.pd >= 0.9) & (pl["median"] < 0)).sum())])
    NUM.set("PartLRangeMin", f"{pl['median'].min():.0f}".replace("-", "\\textminus "))
    NUM.set("PartLRangeMax", f"{pl['median'].max():.0f}")
    # order / practice in A1 and A2 (nuisance terms, registered models)
    lo, med, hi = np.percentile(pct(a1["bO"]), [2.5, 50, 97.5])
    NUM.set("RTOrder", f"{med:.1f}".replace("-", "\\textminus "))
    NUM.set("RTOrderLo", f"{lo:.1f}".replace("-", "\\textminus "))
    NUM.set("RTOrderHi", f"{hi:.1f}".replace("-", "\\textminus "))
    # descriptive: share of dissatisfied acoustic / thermal ratings at T1 vs T2-T3
    r = data.ratings()
    for item, key in (("acoustic_satisfaction", "Acoustic"), ("thermal_satisfaction", "Thermal")):
        d = r[r["item"] == item]
        NUM.set(f"Dissat{key}TOne", f"{100 * (d[d.time == 'T1'].value <= 2).mean():.0f}")
        NUM.set(f"Dissat{key}TTwoThree", f"{100 * (d[d.time != 'T1'].value <= 2).mean():.0f}")
    d = r[r["item"] == "thermal_preference"]
    NUM.set("PreferCoolerTOne", f"{100 * (d[d.time == 'T1'].value < 0).mean():.0f}")
    NUM.set("PreferCoolerTTwoThree", f"{100 * (d[d.time != 'T1'].value < 0).mean():.0f}")
    # room conditions by lighting (integrity of the non-manipulated domains)
    room = pd.read_csv(GEN / "room_by_condition.tsv", sep="\t").set_index(["variable", "condition"])
    for v, key, f in (("air_temperature", "T", "{:.1f}"), ("co2", "CO", "{:.0f}"), ("relative_humidity", "RH", "{:.0f}"),
                      ("illuminance", "Lux", "{:.0f}"), ("sound_level", "DB", "{:.1f}")):
        for c in ("warm", "cool"):
            NUM.set(f"Room{key}{c.capitalize()}", f.format(room.loc[(v, c), "mean"]))
    # joint perception model (S8): pooled cross-modal effect of lighting
    res = pd.read_csv(GEN / "results.tsv", sep="\t")
    x = res[(res.analysis == "S8") & (res.term == "muL")].iloc[0]
    NUM.set("JointMuL", f"{x['median']:.2f}".replace("-", "\\textminus "))
    NUM.set("JointMuLLo", f"{x.lo:.2f}".replace("-", "\\textminus "))
    NUM.set("JointMuLHi", f"{x.hi:.2f}".replace("-", "\\textminus "))
    NUM.set("JointMuLPd", f"{x.pd:.2f}")
    for an in ("bsnd",):
        for item, key in (("acoustic_satisfaction", "AcSat"), ("acoustic_assessment", "AcAss"),
                          ("thermal_satisfaction", "ThSat"), ("thermal_preference", "ThPref"), ("thermal_sensation", "ThSens")):
            y = res[(res.outcome == item) & res.analysis.isin(["primary", "pref primary"]) & (res.term == an)].iloc[0]
            NUM.set(f"Snd{key}", f"{y['median']:.2f}".replace("-", "\\textminus "))
            NUM.set(f"Snd{key}Lo", f"{y.lo:.2f}".replace("-", "\\textminus "))
            NUM.set(f"Snd{key}Hi", f"{y.hi:.2f}".replace("-", "\\textminus "))
    # range of the lighting effect on response time across the sensitivity analyses; accuracy in S2
    sens = res[(res.outcome == "A1") & (res.term == "bL") & res.analysis.str.match(r"S\d")]
    NUM.set("SensLMin", f"{sens['median'].min():.1f}")
    NUM.set("SensLMax", f"{sens['median'].max():.1f}")
    s2 = res[(res.outcome == "A2") & (res.term == "bL") & (res.analysis == "S2")].iloc[0]
    NUM.set("AccLightSTwo", f"{s2['median']:.2f}")
    NUM.set("AccLightSTwoLo", f"{s2.lo:.2f}")
    NUM.set("AccLightSTwoHi", f"{s2.hi:.2f}")
    # accuracy level and n
    NUM.set("AccOverall", f"{100 * base.correct.mean():.1f}")
    NUM.set("NStrategyBlocks", "six")
    assoc = pd.read_csv(GEN / "associations.tsv", sep="\t")
    hr = assoc[(assoc.x == "hr") & (assoc.y == "speed_z")].iloc[0]
    NUM.set("AssocHrSpeed", f"{hr.r:.2f}")
    NUM.set("AssocHrSpeedLo", f"{hr.lo:.2f}".replace("-", "\\textminus "))
    NUM.set("AssocHrSpeedHi", f"{hr.hi:.2f}")
    NUM.set("NItemsPerception", str(len(PERCEPTION)))
    NUM.write()
    supplement_tables(res)


def _cell(x, f="{:.2f}") -> str:
    return (f"{f.format(x['median'])} [{f.format(x.lo)}, {f.format(x.hi)}]").replace("-", "\\textminus ")


def supplement_tables(res: pd.DataFrame) -> None:
    """Tables S1 (convergence by family) and S2 (secondary outcomes)."""
    c = pd.read_csv(GEN / "convergence.tsv", sep="\t")
    c["family"] = c.model.str.split().str[0]
    refit = c.get("refit_longer", pd.Series(False, index=c.index)).fillna(False).astype(bool)
    lines = []
    for fam, g in c.groupby("family"):
        lines.append(f"{fam} & {len(g)} & {g.rhat_max.max():.3f} & {g.ess_bulk_min.min():.0f} & {int(g.divergences.sum())} & "
                     f"{int(refit[g.index].sum())} \\\\")
    (GEN / "table_convergence.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")
    from .common import AFFECT, ITEM_LABEL, PREFERENCE
    lines = []
    for item in PREFERENCE + AFFECT:
        an = "pref primary" if item in PREFERENCE else "primary"
        row = [ITEM_LABEL[item].replace("→", "$\\to$")]
        for term in ("dE_L", "dE_S"):
            x = res[(res.outcome == item) & (res.analysis == an) & (res.term == term)].iloc[0]
            row.append(_cell(x))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\midrule")
    unit = {"hr": ("Heart rate (bpm)", "{:.1f}"), "ln_rmssd": ("Resting ln RMSSD", "{:.2f}"),
            "skin_temp": ("Wrist skin temperature (\\textdegree C)", "{:.2f}"),
            "face_temp_bg": ("Facial temperature (\\textdegree C, relative)", "{:.2f}")}
    for out, (lab, f) in unit.items():
        row = [lab]
        for term in ("bL", "bS"):
            x = res[(res.outcome == out) & (res.analysis == "primary") & (res.term == term)]
            row.append(_cell(x.iloc[0], f) if len(x) else "--")
        lines.append(" & ".join(row) + " \\\\")
    (GEN / "table_secondary.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")
