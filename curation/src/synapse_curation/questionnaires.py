"""Questionnaires T0-T3: parsing, coding from the deployed form definitions, and linkage of the
anonymous T1-T3 forms.

Instruments (decision D22): three Google Forms embedded in the platform; definitions archived in
``raw/metadata/deployed_google_forms/form_*.json`` (retrieved 2026-10-27).  Item wording, option
labels and their order are read from those definitions; ordinal items are coded 1..k in the order
of the deployed options and every observed response label must exist among them.

T0 carries participant and (self-reported) session.  T1 (after the baseline), T2 (after round 1)
and T3 (after round 2) carried no identifier.  Because submitting an embedded form immediately
loaded the next platform page, each anonymous submission is linked to the participant whose next
platform activity starts shortly afterwards:
  T1 -> first round-1 activity (n-back instruction event or first n-back response)
  T2 -> first round-2 n-back (digit) response
  T3 -> last round-2 flanker (digit) response, which precedes T3
Within each concurrently tested pair the assignment minimising the summed squared timing
residual is chosen; residuals and the cost ratio of the best alternative are reported.
"""
from __future__ import annotations

import json
import re
from itertools import permutations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from . import behavior, paths
from .common import FIRST_SESSION, TZ, local_to_unix, sessions

RAW = {"T0": "surveys/surveyT0.csv", "T1": "surveys/surveyT1.csv", "T23": "surveys/surveyT2-T3.csv"}
FORMS = {"T0": "form_T0.json", "T1": "form_T1.json", "T23": "form_T2_T3.json"}

T0_NAMES = [
    "timestamp", "participant_raw", "session_raw",
    "health_hearing_problem", "health_vision_problem", "health_neurological_condition", "health_medication",
    "mental_alert_focused_now", "mental_energy", "concentration_difficulty_freq", "physically_ready",
    "physical_energy", "fatigue_during_day_freq", "positive_motivated_now", "emotional_energy",
    "overwhelmed_stressed_freq",
    "clothing_items", "caffeine_daily", "caffeine_last_intake", "meal_before_session",
    "cold_symptom_severity", "smell_change",
    "sleep_duration_h", "sleep_quality", "deep_sleep", "tired_on_waking", "woke_several_times",
    "insufficient_sleep",
    "ieq_temperature", "ieq_humidity", "ieq_air_quality", "ieq_lighting", "ieq_noise",
    "ieq_workspace_size", "ieq_seating_comfort", "ieq_screen_position", "ieq_overall_comfort",
    "ieq_suitable_for_cognitive_tasks",
]
HEALTH = ["health_hearing_problem", "health_vision_problem", "health_neurological_condition", "health_medication"]
IEQ = ["thermal_sensation", "thermal_satisfaction", "thermal_preference",
       "air_quality_assessment", "air_quality_satisfaction", "air_movement_preference",
       "light_assessment", "light_satisfaction", "light_preference",
       "acoustic_assessment", "acoustic_satisfaction", "acoustic_preference"]
STATE = ["task_difficulty", "task_focus", "relaxation_alertness", "emotional_stability",
         "task_engagement", "cognitive_load", "emotional_valence", "emotional_arousal"]
T1_NAMES = ["timestamp", *IEQ]
T23_NAMES = ["timestamp", *STATE, *IEQ]
# nominal 3-option preference items are coded -1 / 0 / +1 in this direction
PREF_CODES = {
    "thermal_preference": {"Cooler": -1, "No Change": 0, "Warmer": 1},
    "air_movement_preference": {"Less air movement": -1, "No change": 0, "More air movement": 1},
    "light_preference": {"Dimmer": -1, "No change": 0, "Brighter": 1},
    "acoustic_preference": {"Quiter": -1, "No change": 0, "Louder": 1},
}
NOT_CODED = {"timestamp", "participant_raw", "session_raw", "clothing_items", "sleep_duration_h"}


# --------------------------------------------------------------------------- form definitions
def form_items(key: str) -> list[dict]:
    """Answerable items of a deployed form: title, description, options, type."""
    raw = json.loads((paths.RAW / "metadata" / "deployed_google_forms" / FORMS[key]).read_text())
    out = []
    for it in raw["FB_PUBLIC_LOAD_DATA_"][1][1]:
        if it[3] in (6, 8, 11, 12) or not (len(it) > 4 and it[4]):
            continue
        sub = it[4][0]
        out.append({"title": (it[1] or "").strip(), "description": (it[2] or "").strip() if len(it) > 2 and it[2] else "",
                    "type": {0: "short_text", 2: "single_choice", 3: "dropdown", 4: "checkboxes"}.get(it[3], str(it[3])),
                    "options": [o[0] for o in (sub[1] or [])] if len(sub) > 1 and sub[1] else [],
                    "required": bool(sub[2]) if len(sub) > 2 else None})
    return out


def item_table(key: str, names: list[str]) -> pd.DataFrame:
    """Map export column names to form items (order-preserving, verified by option membership)."""
    items = form_items(key)
    cols = [n for n in names if n != "timestamp"]
    if len(items) != len(cols):
        raise ValueError(f"{key}: {len(items)} form items vs {len(cols)} export columns")
    t = pd.DataFrame(items)
    t.insert(0, "variable", cols)
    return t


def _code(series: pd.Series, var: str, options: list[str]) -> pd.Series:
    if var in PREF_CODES:
        return series.map(PREF_CODES[var]).astype("Int64")
    pos = {o: i + 1 for i, o in enumerate(options)}
    return series.map(pos).astype("Int64")


# --------------------------------------------------------------------------- loading
def _read(key: str, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(paths.RAW / RAW[key], dtype=str, keep_default_na=False)
    if len(d.columns) != len(names):
        raise ValueError(f"{key}: expected {len(names)} columns, found {len(d.columns)}")
    items = item_table(key, names)
    items["export_header"] = list(d.columns[1:])
    d.columns = names
    d = d.apply(lambda s: s.str.strip())
    d.insert(0, "src_file", "raw/" + RAW[key])
    d.insert(1, "src_row", np.arange(2, len(d) + 2))
    d["submitted_local"] = pd.to_datetime(d["timestamp"], format="%m/%d/%Y %H:%M:%S").dt.tz_localize(TZ)
    d["submitted_unix"] = local_to_unix(d["submitted_local"])
    # verify labels against deployed options and add numeric codes
    for r in items.itertuples():
        if r.variable in NOT_CODED or r.type in ("checkboxes", "short_text"):
            continue
        bad = set(d[r.variable]) - set(r.options) - {""}
        if bad:
            raise ValueError(f"{key}.{r.variable}: labels not in deployed form: {bad}")
        if r.variable not in HEALTH:
            d[r.variable + "_code"] = _code(d[r.variable], r.variable, r.options)
    return d, items


def load_t0() -> tuple[pd.DataFrame, pd.DataFrame]:
    d, items = _read("T0", T0_NAMES)
    d["participant"] = d["participant_raw"].str.upper().str.replace("O", "0", regex=False)   # 'PO4'
    d["session_self_reported"] = d["session_raw"].str.upper()
    d["exclusion"] = np.where(d["submitted_local"] < FIRST_SESSION, "pilot_or_test_day", "")
    # The session letter was self-reported and is wrong at least once (sub-P01 typed 'B' at 12:52
    # during session A): the session is assigned from the logged windows (T0 precedes tasks).
    s = sessions()
    d["session"] = pd.NA
    for i, r in d[d["exclusion"] == ""].iterrows():
        m = s[(s.participant == r["participant"]) & (r["submitted_unix"] >= s.log_start_unix - 1200)
              & (r["submitted_unix"] <= s.log_end_unix)]
        d.loc[i, "session"] = m["session"].iloc[0] if len(m) == 1 else pd.NA
    d["session_corrected"] = d["session"].ne(d["session_self_reported"]) & d["session"].notna()
    d["clothing_clo_est"], d["clothing_note"] = zip(*d["clothing_items"].map(clothing_clo))
    d["questionnaire"] = "T0"
    return d, items


def _anchors() -> dict[str, pd.DataFrame]:
    ev = behavior.load_events()
    ev = ev[(ev["exclusion"] == "") & (ev["event_type"] == "NBack_Level_1_Instructions")]
    t1 = ev.groupby(["participant", "session"])["start_unix"].min().rename("anchor")
    nb = behavior.load_nback()
    nb = nb[nb["exclusion"] == ""]
    # the event log does not distinguish letter/digit and misses some round-1 events (sub-P07/ses-A):
    # the T1 anchor is the earliest of the instruction onset and the first round-1 response
    t1_fb = nb[nb["round"] == 1].groupby(["participant", "session"])["submitted_unix"].min()
    t1 = pd.concat([t1, t1_fb.rename("anchor")], axis=1).min(axis=1).rename("anchor")
    t2 = nb[nb["round"] == 2].groupby(["participant", "session"])["submitted_unix"].min().rename("anchor")
    fl = behavior.load_flanker()
    fl = fl[(fl["exclusion"] == "") & (fl["round"] == 2)]
    t3 = fl.groupby(["participant", "session"])["submitted_unix"].max().rename("anchor")
    return {"T1": t1.reset_index(), "T2": t2.reset_index(), "T3": t3.reset_index()}


def _link(forms: pd.DataFrame, anchors: pd.DataFrame, which: str) -> pd.DataFrame:
    s = sessions()
    a = anchors.merge(s[["participant", "session", "group"]], on=["participant", "session"])
    a["slot"] = a["group"] + a["session"]
    for _, ga in a.groupby("slot"):
        lo, hi = ga["anchor"].min() - 900, ga["anchor"].max() + 900
        gf = forms[(forms["submitted_unix"] >= lo) & (forms["submitted_unix"] <= hi) & forms["participant"].isna()]
        if gf.empty:
            continue
        sign = -1 if which == "T3" else 1
        res = np.array([[sign * (an - fs) for an in ga["anchor"]] for fs in gf["submitted_unix"]])
        # squared residuals are order-preserving with same-signed residuals; causally impossible
        # orders (> 5 s tolerance) are forbidden
        cost = np.where(res < -5, 1e12, res ** 2)
        r, c = linear_sum_assignment(cost)
        best = cost[r, c].sum()
        alt = np.inf
        for rows in permutations(range(cost.shape[0]), min(cost.shape)):
            pairs = set(zip(rows, range(cost.shape[1])))
            if pairs != set(zip(r, c)):
                alt = min(alt, sum(cost[i, j] for i, j in pairs))
        ratio = alt / max(best, 1.0) if np.isfinite(alt) else np.inf
        for i, j in zip(r, c):
            if cost[i, j] >= 1e12:
                continue
            idx = gf.index[i]
            forms.loc[idx, ["participant", "session"]] = ga.iloc[j][["participant", "session"]].to_numpy()
            forms.loc[idx, "link_residual_s"] = res[i, j]
            forms.loc[idx, "link_alt_cost_ratio"] = min(ratio, 1e6)
            forms.loc[idx, "link_quality"] = ("single_participant" if len(ga) == 1 else
                                              "causally_unique" if ratio >= 1e6 else
                                              "strong" if ratio >= 4 else
                                              "moderate" if ratio >= 1.5 else "weak")
    return forms


def load_linked() -> dict[str, pd.DataFrame]:
    """Return {'T0','T1','T2','T3'} tables (all rows, with exclusion) and item tables."""
    anchors = _anchors()
    t1, items_t1 = _read("T1", T1_NAMES)
    t23, items_t23 = _read("T23", T23_NAMES)
    for d in (t1, t23):
        d["participant"] = pd.Series(pd.NA, index=d.index, dtype="object")
        d["session"] = pd.Series(pd.NA, index=d.index, dtype="object")
        d["link_residual_s"] = np.nan
        d["link_alt_cost_ratio"] = np.nan
        d["link_quality"] = pd.Series(pd.NA, index=d.index, dtype="object")
    t1 = _link(t1, anchors["T1"], "T1")
    t1["questionnaire"] = "T1"
    # T2 and T3 share one form: link T3 first (end of round 2), then T2
    t23["questionnaire"] = pd.NA
    t23 = _link(t23, anchors["T3"], "T3")
    t23.loc[t23["participant"].notna(), "questionnaire"] = "T3"
    t23 = _link(t23, anchors["T2"], "T2")
    t23.loc[t23["participant"].notna() & t23["questionnaire"].isna(), "questionnaire"] = "T2"
    for d in (t1, t23):
        d["exclusion"] = np.where(d["submitted_local"] < FIRST_SESSION, "pilot_or_test_day",
                                  np.where(d["participant"].isna(), "unlinked", ""))
    t0, items_t0 = load_t0()
    return {"T0": t0, "T1": t1, "T2": t23[t23.questionnaire.eq("T2") | t23.exclusion.ne("")],
            "T3": t23[t23.questionnaire.eq("T3")],
            "items_T0": items_t0, "items_T1": items_t1, "items_T23": items_t23}


# --------------------------------------------------------------------------- clothing insulation
# Garment insulation (clo) following ASHRAE 55-2020 Table 5.2.2.2B / ISO 9920 typical summer
# garments; the checklist does not distinguish garment weights, so mid-range values are used and
# the ensemble value is an estimate (about +-0.1 clo).
GARMENT_CLO = {
    "undergarments": 0.04, "t-shirt or short-sleeve shirt": 0.08, "t-shirt or long-sleeve shirt": 0.20,
    "shorts": 0.06, "cotton or synthetic trousers": 0.20, "denim jeans": 0.24, "hoodie": 0.34,
    "sweatshirt": 0.34, "socks": 0.02, "shoes": 0.02, "jacket": 0.36, "skirt": 0.14,
}
CHAIR_CLO = 0.10
UPPER = {"t-shirt or short-sleeve shirt", "t-shirt or long-sleeve shirt", "hoodie", "sweatshirt", "jacket"}
LOWER = {"shorts", "cotton or synthetic trousers", "denim jeans", "skirt"}


def clothing_clo(items: str) -> tuple[float, str]:
    """(estimated ensemble clo incl. office chair, note).  Checklists without at least one
    upper- and one lower-body garment are incomplete self-reports and yield NaN."""
    its = [x.strip().lower() for x in re.split(r",(?![^(]*\))", str(items)) if x.strip()]
    unknown = [i for i in its if i not in GARMENT_CLO]
    if not (set(its) & UPPER and set(its) & LOWER):
        return float("nan"), ";".join(unknown + ["incomplete_checklist"])
    return round(sum(GARMENT_CLO.get(i, 0.0) for i in its) + CHAIR_CLO, 2), ";".join(unknown)
