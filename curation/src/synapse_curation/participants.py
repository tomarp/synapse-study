"""Participants table and the pre-study background questionnaire (decision D08).

``raw/ParticipantInformation.xlsx`` holds 9 responses (10 participants) with direct identifiers
and no participant code.  It is therefore released only as (i) an aggregate description and
(ii) an unlinked, generalised table whose rows are shuffled with a fixed seed.  Direct identifiers
(name, e-mail), the free-text place of residence and free-text health conditions are never read
into the output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import paths
from .common import sessions

# column index in the export -> (output variable, generalisation)
BACKGROUND = {
    1: ("sex", None), 2: ("age_band", "age5"), 3: ("education_field", None),
    4: ("caffeine_frequency", None), 5: ("tobacco_frequency", None), 6: ("alcohol_frequency", None),
    7: ("time_in_denmark", None), 9: ("climate_difference_previous_country", None),
    10: ("climate_previous_country", None), 11: ("sensitive_ieq_parameters", None),
    12: ("usual_bedtime_workweek", None), 13: ("usual_wake_time_workweek", None),
    14: ("regular_sleep_days_per_workweek", "int"),
    15: ("preferred_thermal", None), 16: ("preferred_sound_environment", None),
    17: ("preferred_lighting", None), 18: ("preferred_air_quality", None),
    19: ("headphone_tolerance", None), 20: ("visual_impairment", "yesno"),
    21: ("hearing_impairment", "yesno"), 22: ("olfactory_impairment", "yesno"),
    23: ("self_rated_physical_health_0_10", "score"), 24: ("self_rated_mental_health_0_10", "score"),
    27: ("english_proficiency", "first_word"),
}
NEVER_READ = {0: "name", 8: "city/country of residence (free text)", 25: "physical health conditions (free text)",
              26: "mental health conditions (free text)", 28: "declaration", 29: "communication consent",
              30: "e-mail", 31: "form status"}


def _age_band(v) -> str:
    a = int(float(v))
    lo = 5 * (a // 5)
    return f"{lo}-{lo + 4}"


def _score(v):
    # 0-10 item anchored "0=Poor, 10=Excellent"; one respondent typed the anchor word itself
    s = str(v).strip()
    return 10 if s.lower() == "excellent" else int(float(s))


def background_raw() -> pd.DataFrame:
    return pd.read_excel(paths.RAW / "ParticipantInformation.xlsx", header=0, dtype=str)


def background_unlinked(seed: int = 20250624) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = background_raw()
    out = pd.DataFrame(index=raw.index)
    for i, (name, rule) in BACKGROUND.items():
        v = raw.iloc[:, i].astype(str).str.strip()
        if rule == "age5":
            v = v.map(_age_band)
        elif rule == "score":
            v = v.map(_score)
        elif rule == "int":
            v = v.map(lambda x: int(float(x)))
        elif rule == "yesno":
            v = v.map(lambda x: "no" if x.lower().startswith("no") else "yes")
        elif rule == "first_word":
            v = v.str.split(":").str[0]
        out[name] = v.to_numpy()
    out["sex"] = out["sex"].str.lower()
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    out.insert(0, "respondent", [f"R{i + 1}" for i in range(len(out))])
    ages = pd.to_numeric(raw.iloc[:, 2], errors="coerce")
    summary = pd.DataFrame([
        {"statistic": "participants_tested", "value": 10},
        {"statistic": "background_forms_received", "value": len(raw)},
        {"statistic": "female", "value": int((out.sex == "female").sum())},
        {"statistic": "male", "value": int((out.sex == "male").sum())},
        {"statistic": "age_mean_years", "value": round(float(ages.mean()), 1)},
        {"statistic": "age_sd_years", "value": round(float(ages.std(ddof=1)), 1)},
        {"statistic": "age_min_years", "value": int(ages.min())},
        {"statistic": "age_max_years", "value": int(ages.max())},
    ], dtype=object)
    return out, summary


def participants_table(avail: pd.DataFrame) -> pd.DataFrame:
    """One row per participant.  ``avail``: per participant-session availability flags."""
    s = sessions()
    first = s[s.session == "A"].set_index("participant_id")
    rows = []
    for pid, r in first.iterrows():
        rows.append({
            "participant_id": pid, "group": r.group,
            "co_tested_with": r.co_tested_with, "seat": r.seat,
            "lighting_order": f"{r.light}_first",
        })
    p = pd.DataFrame(rows)
    a = avail.groupby("participant_id").agg({c: "sum" for c in avail.columns if c.startswith("has_")})
    for c in a.columns:
        p["n_sessions_" + c[4:]] = p["participant_id"].map(a[c]).fillna(0).astype(int)
    return p
