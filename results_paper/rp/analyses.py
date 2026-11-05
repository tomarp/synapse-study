"""Registered analyses (ANALYSIS_PLAN.md, sections 4-7): families A-D, sensitivity analyses S1-S9, exploratory analyses.

Results are written to ``generated/``: ``results.tsv`` (one row per estimand and analysis), ``participant_effects.tsv``,
``room_by_condition.tsv``, ``associations.tsv``, ``convergence.tsv``, ``deviations.tsv`` and the number macros.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import data, models
from .common import AFFECT, GEN, NUM, PERCEPTION, PREFERENCE, SEED, SUBS, TASKS, UNBLIND
from .models import ROPE, summarise

ROWS: list[dict] = []
DIAG: list[dict] = []
DEVIATIONS = [
    ("D1 heart rate", "Inter-beat intervals are accepted when within 20 % of the local median of 11 intervals instead "
     "of the previous accepted interval", "the plan's rule locks onto an early artefact and then rejects all following "
     "beats (found on blind data)", "before unblinding"),
    ("Sampler", "1500 draws after 2000 tuning steps per chain, target acceptance 0.95 (0.99 for the physiology models)",
     "the plan's 1000/1000 gave a bulk ESS below 400 and divergent transitions in A2 and D1 (blind data)",
     "before unblinding"),
    ("Convergence", "Models that miss the convergence criteria are refitted once with twice the draws and tuning steps",
     "implements the plan's 'tuned longer' rule automatically", "before unblinding"),
    ("S4", "Questionnaires with link_quality 'weak' or 'moderate' are excluded", "the dataset README calls both levels "
     "'weak link'; clarification of the plan", "before unblinding"),
]

PCT = lambda x: 100 * (np.exp(x) - 1)                               # noqa: E731  log-ratio -> percent change
OR = np.exp


def _row(family, analysis, outcome, term, draws, rope, transform=None, scale="model", **extra) -> dict:
    s = summarise(draws, rope)
    if transform is not None:
        s.update(median=float(transform(s["median"])), lo=float(transform(s["lo"])), hi=float(transform(s["hi"])))
    r = dict(family=family, analysis=analysis, outcome=outcome, term=term, scale=scale, **s, **extra)
    ROWS.append(r)
    return r


def _diag(name: str, fit: dict) -> None:
    DIAG.append(dict(model=name, **fit["_diag"]))


# --------------------------------------------------------------------------------------------- family A
def performance() -> None:
    t = data.trials()
    base = t[t.task.isin(TASKS) & t.rt_ok]
    variants = {
        "primary": base[~base.strategy],
        "S1": base[~base.strategy & ~base.early],
        "S2": base,
    }
    for outcome, rope, tr, scale in (("log_rt", ROPE["log_rt"], PCT, "percent"), ("correct", ROPE["logit_acc"], OR, "odds ratio")):
        model = "A1" if outcome == "log_rt" else "A2"
        runs = [(k, d, {}) for k, d in variants.items()]
        if outcome == "log_rt":
            runs.append(("S3", variants["primary"], {}))
        runs += [("S5", variants["primary"], dict(extra="lux_z")), ("S6", variants["primary"], dict(extra="slot")),
                 ("S7 wide", variants["primary"], dict(prior_scale=2.0)), ("S7 narrow", variants["primary"], dict(prior_scale=0.5))]
        for analysis, d, kw in runs:
            if outcome == "log_rt" and analysis != "S3":
                d = d[d.correct == 1]
            fit = models.performance(d, outcome, name=f"{model}-{analysis}", **kw)
            _diag(f"{model} {analysis}", fit)
            for term in ("bL", "bS", "bLS", "bO"):
                _row("A", analysis, model, term, fit[term], rope, tr, scale, n=len(d))
            if analysis != "primary":
                continue
            for i, task in enumerate(TASKS):
                _row("A", "task", model, "bL", fit["bL"] + fit["gL"][:, i], rope, tr, scale, task=task)
                _row("A", "task", model, "bS", fit["bS"] + fit["gS"][:, i], rope, tr, scale, task=task)
            for term in ("sdL", "sdS"):
                _row("A", "heterogeneity", model, term, fit[term], rope, None, "model scale")
            pe = []
            for j, p in enumerate(SUBS):
                for term, u in (("bL", "uL"), ("bS", "uS")):
                    s = summarise(fit[term] + fit[u][:, j], rope)
                    pe.append(dict(model=model, participant_id=p, term=term, **{k: (float(tr(v)) if k in ("median", "lo", "hi") else v)
                                                                               for k, v in s.items()}))
            pd.DataFrame(pe).to_csv(GEN / f"participant_effects_{model}.tsv", sep="\t", index=False)
            NUM.set(f"N{model}Trials", f"{len(d):,}".replace(",", "{,}"))


# --------------------------------------------------------------------------------------------- family B, C
def _item_frame(r: pd.DataFrame, item: str) -> tuple[pd.DataFrame, int, np.ndarray]:
    d = r[r["item"] == item].copy()
    if item in PREFERENCE:
        d["y"], K, values = d.value + 1, 3, np.array([-1.0, 0.0, 1.0])
    elif item in AFFECT:
        K = 7 if item == "task_focus" else 9
        d["y"], values = d.value - 1, np.arange(1, K + 1, dtype=float)
    else:
        d["y"], K, values = d.value - 1, 5, np.arange(1, 6, dtype=float)
    return d, K, values


def ratings() -> None:
    r = data.ratings()
    for fam, items, keep_time in (("B", PERCEPTION + PREFERENCE, ("T1", "T2", "T3")), ("C", AFFECT, ("T2", "T3"))):
        rr = r[r.time.isin(keep_time)]
        for item in items:
            d, K, values = _item_frame(rr, item)
            runs = [("primary", d, {}), ("S4", d[~d.link_quality.isin(["weak", "moderate"])], {})]
            if item in PERCEPTION:
                runs += [("S5", d, dict(extra="lux_z")), ("S6", d, dict(extra="slot")),
                         ("S7 wide", d, dict(prior_scale=2.0)), ("S7 narrow", d, dict(prior_scale=0.5))]
            for analysis, dd, kw in runs:
                fit = models.ordinal(dd, K, fam, name=f"{fam}-{item}-{analysis}", **kw)
                _diag(f"{fam} {item} {analysis}", fit)
                sub = "pref" if item in PREFERENCE else analysis
                for term in ("bL", "bS", "bLS", "bR", "bO") + (("bsnd",) if fam == "B" else ()):
                    _row(fam, analysis if sub != "pref" else f"pref {analysis}", item, term, fit[term], ROPE["ordinal"], n=len(dd))
                if analysis == "primary":
                    for f_, lab in (("L", "dE_L"), ("S", "dE_S")):
                        de = models.expected_difference(fit, dd, f_, values)
                        _row(fam, "primary" if sub != "pref" else "pref primary", item, lab, de, 0.1, None, "rating points")
    d = r[r["item"].isin(PERCEPTION)].copy()
    d["y"] = d.value - 1
    fit = models.ordinal_joint(d, PERCEPTION, 5)
    _diag("B joint S8", fit)
    for i, item in enumerate(PERCEPTION):
        for term in ("bL", "bS"):
            _row("B", "S8", item, term, fit[term][:, i], ROPE["ordinal"])
    for term in ("muL", "muS", "tauL", "tauS"):
        _row("B", "S8", "cross-modal", term, fit[term], ROPE["ordinal"])


# --------------------------------------------------------------------------------------------- family D
def physiology() -> None:
    ph = data.physiology()
    specs = [("hr", "phase", 5.0, ROPE["hr"]), ("ln_rmssd", "baseline", 0.3, ROPE["ln_rmssd"]),
             ("skin_temp", "phase", 0.5, ROPE["temp"]), ("face_temp_bg", "within", 0.5, ROPE["temp"])]
    for outcome, design, eff, rope in specs:
        d = ph[ph.phase == "baseline"] if design == "baseline" else ph
        runs = [("primary", outcome, {})]
        if design != "within":
            runs.append(("S5", outcome, dict(extra="lux_z")))
        else:
            runs.append(("S9", "face_temp", {}))
        for analysis, out, kw in runs:
            fit = models.gaussian(d, out, design, eff, name=f"D-{out}-{analysis}", **kw)
            _diag(f"D {out} {analysis}", fit)
            for term in [k for k in fit if k.startswith("b")]:
                _row("D", analysis, outcome, term, fit[term], rope, n=int(d[out].notna().sum()))


# ----------------------------------------------------------------------------------------- exploratory
def room_by_condition() -> pd.DataFrame:
    ph = data.physiology()
    rows = []
    for k in ("room_air_temperature", "room_relative_humidity", "room_co2", "room_sound_level", "room_illuminance"):
        for grp, sel in (("warm", ph.L < 0), ("cool", ph.L > 0), ("babble", ph.S < 0), ("traffic", ph.S > 0),
                         ("baseline", ph.phase == "baseline")):
            v = ph.loc[sel, k]
            rows.append(dict(variable=k[5:], condition=grp, mean=v.mean(), sd=v.std(), min=v.min(), max=v.max(), n=v.count()))
    out = pd.DataFrame(rows)
    out.to_csv(GEN / "room_by_condition.tsv", sep="\t", index=False, float_format="%.3f")
    return out


def _rmcorr(df: pd.DataFrame, x: str, y: str, centre: str) -> dict:
    d = df.dropna(subset=[x, y]).copy()
    for v in (x, y):
        d[v] = d[v] - d.groupby(centre)[v].transform("mean")
    r = float(np.corrcoef(d[x], d[y])[0, 1]) if d[x].std() > 0 and d[y].std() > 0 else np.nan
    rng = np.random.default_rng(SEED)
    parts = d.participant_id.unique()
    boot = []
    for _ in range(2000):
        b = pd.concat([d[d.participant_id == p] for p in rng.choice(parts, len(parts))])
        if b[x].std() > 0 and b[y].std() > 0:
            boot.append(np.corrcoef(b[x], b[y])[0, 1])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(x=x, y=y, centred_within=centre, r=r, lo=lo, hi=hi, n_obs=len(d), n_participants=len(parts))


def associations() -> pd.DataFrame:
    """Round-level within-person associations between performance, ratings and physiology (exploratory)."""
    t = data.trials()
    a = t[t.task.isin(TASKS) & t.rt_ok & ~t.strategy & (t.correct == 1)].copy()
    a["z"] = (a.log_rt - a.groupby("task").log_rt.transform("mean")) / a.groupby("task").log_rt.transform("std")
    perf = a.groupby(["participant_id", "session_id", "round"]).z.mean().rename("speed_z").reset_index()
    perf["phase"] = "round" + perf["round"].astype(int).astype(str)
    r = data.ratings()
    rt = r[r.time.isin(["T2", "T3"])].pivot_table(index=["participant_id", "session_id", "time"], columns="item", values="value").reset_index()
    rt["phase"] = rt.time.map({"T2": "round1", "T3": "round2"})
    ph = data.physiology()
    m = perf.merge(rt, on=["participant_id", "session_id", "phase"]).merge(ph, on=["participant_id", "session_id", "phase"], how="left")
    m["session"] = m.participant_id + "/" + m.session_id
    pairs = [("acoustic_satisfaction", "speed_z", "participant_id"), ("cognitive_load", "speed_z", "participant_id"),
             ("task_focus", "speed_z", "participant_id"), ("hr", "speed_z", "participant_id"),
             ("hr", "emotional_arousal", "participant_id"), ("hr", "relaxation_alertness", "participant_id"),
             ("skin_temp", "thermal_sensation", "participant_id"), ("face_temp_bg", "thermal_sensation", "session"),
             ("acoustic_satisfaction", "emotional_valence", "participant_id"),
             ("thermal_satisfaction", "acoustic_satisfaction", "participant_id")]
    out = pd.DataFrame([_rmcorr(m, x, y, c) for x, y, c in pairs])
    out.to_csv(GEN / "associations.tsv", sep="\t", index=False, float_format="%.3f")
    return out


# ------------------------------------------------------------------------------------------- numbers
def _camel(s: str) -> str:
    return "".join(w.capitalize() for w in re.split(r"[^A-Za-z0-9]+", s) if w)


def register_numbers(res: pd.DataFrame) -> None:
    fmt = {"percent": "{:.1f}", "odds ratio": "{:.2f}", "model": "{:.2f}", "rating points": "{:.2f}"}
    prim = res[res.analysis.isin(["primary", "task", "pref primary"])]
    for _, x in prim.iterrows():
        f = fmt.get(x.scale, "{:.2f}")
        if x.family == "D":
            f = "{:.1f}" if x.outcome == "hr" else "{:.2f}"
        key = _camel(f"{x.family} {x.outcome} {x.get('task') if isinstance(x.get('task'), str) else ''} {x.term}")
        for stat in ("median", "lo", "hi"):
            NUM.set(key + {"median": "", "lo": "Lo", "hi": "Hi"}[stat], f.format(x[stat]).replace("-", "\\textminus "))
        NUM.set(key + "Pd", "{:.2f}".format(x.pd))
    descriptives()
    t = data.trials()
    NUM.set("NTrialsAnalysed", f"{int((t.task.isin(TASKS) & t.rt_ok).sum()):,}".replace(",", "{,}"))
    diag = pd.DataFrame(DIAG)
    NUM.set("RhatMax", "{:.3f}".format(diag.rhat_max.max()))
    NUM.set("EssMin", "{:.0f}".format(diag.ess_bulk_min.min()))
    NUM.set("Divergences", str(int(diag.divergences.sum())))
    NUM.set("NModels", str(len(diag)))
    NUM.set("Blind", "false" if UNBLIND else "true")


def descriptives() -> None:
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    ss = data.tsv("background_questionnaire/sample_summary.tsv").set_index("statistic").value
    NUM.set("SampleSex", f"{words[int(ss.female)]} women and {words[int(ss.male)]} men")
    NUM.set("SampleAge", f"{ss.age_mean_years:.1f} $\\pm$ {ss.age_sd_years:.1f} years (range {ss.age_min_years:.0f}--{ss.age_max_years:.0f})")
    s = data.sessions_true()
    dur = (pd.to_datetime(s.session_end) - pd.to_datetime(s.session_start)).dt.total_seconds() / 60
    NUM.set("SessionMedianMin", f"{dur.median():.0f}")
    t = data.timeline()
    for ph, key in (("round1", "RoundOneMin"), ("round2", "RoundTwoMin")):
        NUM.set(key, f"{t[t.phase == ph].duration_s.median() / 60:.1f}")
    ph = data.physiology()
    for k, key, f in (("room_air_temperature", "RoomT", "{:.1f}"), ("room_relative_humidity", "RoomRH", "{:.0f}"),
                      ("room_co2", "RoomCO", "{:.0f}"), ("room_sound_level", "RoomDB", "{:.1f}")):
        NUM.set(key + "Min", f.format(ph[k].min()))
        NUM.set(key + "Max", f.format(ph[k].max()))


def run(parts: list[str] | None = None) -> pd.DataFrame:
    GEN.mkdir(parents=True, exist_ok=True)
    want = lambda k: parts is None or k in parts                         # noqa: E731
    if want("A"):
        performance()
    if want("B"):
        ratings()
    if want("D"):
        physiology()
    if want("E"):
        room_by_condition()
        associations()
    res = pd.DataFrame(ROWS)
    res.to_csv(GEN / "results.tsv", sep="\t", index=False, float_format="%.4f")
    pd.DataFrame(DIAG).to_csv(GEN / "convergence.tsv", sep="\t", index=False, float_format="%.4f")
    pd.DataFrame(DEVIATIONS, columns=["item", "change", "reason", "when"]).to_csv(GEN / "deviations.tsv", sep="\t", index=False)
    if parts is None:
        register_numbers(res)
        NUM.write()
    return res
