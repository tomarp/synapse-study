"""Behavioural task exports: loading, session/round assignment, run reconstruction,
stimulus-based re-scoring and per-session task-configuration reconstruction.

Source: ``raw/dataforms/*-Responses.csv`` (Google-Forms sheets fed by the platform; byte-identical
copies exist in ``raw/tasks/``).  One row per response.  ``Timestamp`` is the Google server
receipt time (local CEST, 1-s resolution, arrival order); ``ReactionTime_ms`` is the client-side
``performance.now()`` difference between stimulus onset and click.  Every curated row keeps its
source file and line (``src_file``, ``src_row``).

Exclusion vocabulary (column ``exclusion``; empty = analysable):
  pilot_or_test_day     23 June 2025 (decision D02)
  test_identifier       participant ID 'test'
  before_first_session  study days, before 24 June 12:40
  no_matching_session   participant/session not in the session table
  superseded_run        an earlier, restarted run of the same block (the last run is kept)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import paths
from .common import FIRST_SESSION, STUDY_DATES, TZ, local_to_unix, sessions

SOURCES = {
    "nback": "dataforms/nBackTask-Responses.csv",
    "stroop": "dataforms/StroopTask-Responses.csv",
    "visual": "dataforms/VisualTask-Responses.csv",
    "flanker": "dataforms/FlankerTask-Responses.csv",
    "events": "dataforms/event-Logs.csv",
}

# raw variant label -> (BIDS task label, round).  Round 1 precedes T2, round 2 follows it.
VARIANT = {
    "nBackLetter": ("nbackletter", 1), "nBackDigit": ("nbackdigit", 2),
    "stroopWord": ("stroopword", 1), "stroopEmoji": ("stroopemoji", 2),
    # configuration of sessions sub-P01/ses-A and sub-P02/ses-A: emotional-word Stroop (round 1), emotional-image Stroop (round 2)
    "Positive": ("stroopemotionalword", 1), "Negative": ("stroopemotionalword", 1),
    "Neutral": ("stroopemotionalword", 1), "Image": ("stroopemotionalimage", 2),
    "visualCount": ("visualcount", 1), "visualShape": ("visualshape", 2),
    "Flanker-Arrow": ("flankerarrow", 1), "Flanker-Number": ("flankerdigit", 2),
}
FAMILY = {"nback": "n-back", "stroop": "Stroop", "visual": "visual", "flanker": "flanker"}
RT_FAST_MS = 150          # anticipatory responses (flag only)
RT_SLOW_MS = {"nback": 10_000, "stroop": 10_000, "flanker": 10_000, "visual": 120_000}


def _read(key: str) -> pd.DataFrame:
    d = pd.read_csv(paths.RAW / SOURCES[key], dtype=str, keep_default_na=False)
    d.columns = [c.strip() for c in d.columns]
    d = d.apply(lambda s: s.str.strip())
    d.insert(0, "src_file", "raw/" + SOURCES[key])
    d.insert(1, "src_row", np.arange(2, len(d) + 2))      # line number in the CSV incl. header
    d["submitted_local"] = pd.to_datetime(d["Timestamp"], format="%m/%d/%Y %H:%M:%S").dt.tz_localize(TZ)
    d["submitted_unix"] = local_to_unix(d["submitted_local"])
    d["participant_raw"] = d["ParticipantID"]
    d["session_raw"] = d["SessionID"]
    d["participant"] = d["ParticipantID"].str.upper()
    d["session"] = d["SessionID"].str.upper()
    return d


def _attach_session(d: pd.DataFrame) -> pd.DataFrame:
    s = sessions()
    d["date"] = d["submitted_local"].dt.strftime("%Y-%m-%d")
    d["exclusion"] = ""
    d.loc[~d["date"].isin(STUDY_DATES), "exclusion"] = "pilot_or_test_day"
    d.loc[(d["exclusion"] == "") & d["participant"].eq("TEST"), "exclusion"] = "test_identifier"
    d.loc[(d["exclusion"] == "") & (d["submitted_local"] < FIRST_SESSION), "exclusion"] = "before_first_session"
    d = d.merge(s[["participant", "session", "date", "light", "sound_round1", "sound_round2",
                   "log_start_unix", "log_end_unix"]], on=["participant", "session", "date"], how="left")
    d.loc[(d["exclusion"] == "") & d["light"].isna(), "exclusion"] = "no_matching_session"
    # plausibility: submissions within the (corrected) logged session window +- 15 min
    ok = d["exclusion"] == ""
    d["outside_logged_window"] = ok & ((d["submitted_unix"] < d["log_start_unix"] - 900) |
                                       (d["submitted_unix"] > d["log_end_unix"] + 900))
    return d.drop(columns=["log_start_unix", "log_end_unix"])


def _round_and_sound(d: pd.DataFrame, raw_variant: str) -> pd.DataFrame:
    vr = d[raw_variant].map(VARIANT)
    d["task"] = vr.map(lambda x: x[0] if isinstance(x, tuple) else np.nan)
    d["round"] = vr.map(lambda x: x[1] if isinstance(x, tuple) else np.nan)
    d["soundscape"] = np.where(d["round"] == 1, d["sound_round1"],
                               np.where(d["round"] == 2, d["sound_round2"], np.nan))
    return d


def _runs_by_counter(g: pd.DataFrame, counter: str) -> list[int]:
    """New run whenever a trial number repeats within the current run (restart/reload).
    A decreasing counter alone is not a restart: Google-Forms submissions occasionally
    arrived out of order (adjacent trials swapped within 1-2 s)."""
    run, seen, out = 1, set(), []
    for c in g[counter].astype(int):
        if c in seen:
            run, seen = run + 1, set()
        seen.add(c)
        out.append(run)
    return out


def _runs_by_gap(g: pd.DataFrame, gap_s: float) -> np.ndarray:
    t = g["submitted_unix"].to_numpy()
    return np.cumsum(np.r_[True, np.diff(t) > gap_s])


def _finalise_runs(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    d["n_runs_in_block"] = d.groupby(keys)["run"].transform("max")
    sup = (d["run"] > 0) & (d["run"] < d["n_runs_in_block"])
    d.loc[(d["exclusion"] == "") & sup, "exclusion"] = "superseded_run"
    return d


def _rt_flags(d: pd.DataFrame, family: str) -> pd.DataFrame:
    d["rt_flag"] = np.select([d["rt_ms"] < RT_FAST_MS, d["rt_ms"] > RT_SLOW_MS[family]],
                             ["fast_lt150ms", f"slow_gt{RT_SLOW_MS[family] // 1000}s"], default="ok")
    return d


# ----------------------------------------------------------------------------------- n-back
def load_nback() -> pd.DataFrame:
    d = _attach_session(_read("nback"))
    d = d.rename(columns={"Task_type": "variant_raw", "Level": "level", "Trial": "trial",
                          "Stimulus": "stimulus", "Response": "response",
                          "Correct": "correct_recorded", "ReactionTime_ms": "rt_ms"})
    d = _round_and_sound(d, "variant_raw")
    d["level"] = d["level"].astype(int)
    d["trial"] = d["trial"].astype(int)
    d["rt_ms"] = d["rt_ms"].astype(float)
    d["correct_recorded"] = d["correct_recorded"].astype(int)
    keys = ["participant", "session", "task", "level"]
    d = d.sort_values("src_row")
    d["run"] = 0
    for _, g in d[d["exclusion"] == ""].groupby(keys):
        d.loc[g.index, "run"] = _runs_by_counter(g, "trial")
    d = _finalise_runs(d, keys)
    # Re-score from the stimulus history within the run (target = stimulus equals the one n back).
    d["is_target"] = pd.array([pd.NA] * len(d), dtype="boolean")
    d["rescore_status"] = ""
    for _, g in d[d["run"] > 0].groupby(keys + ["run"]):
        n = int(g["level"].iloc[0])
        stim = dict(zip(g["trial"], g["stimulus"]))
        for idx, tr in zip(g.index, g["trial"]):
            if tr <= n:
                d.at[idx, "is_target"], d.at[idx, "rescore_status"] = False, "first_n_nontarget"
            elif (tr - n) in stim:
                d.at[idx, "is_target"], d.at[idx, "rescore_status"] = stim[tr] == stim[tr - n], "checked"
            else:
                d.at[idx, "rescore_status"] = "uncheckable_missing_history"
    ok = d["is_target"].notna()
    d["correct"] = pd.array([pd.NA] * len(d), dtype="Int64")
    d.loc[ok, "correct"] = ((d.loc[ok, "response"] == "TARGET") == d.loc[ok, "is_target"].astype(bool)).astype(int)
    d["score_discrepancy"] = ok & (d["correct"] != d["correct_recorded"])
    return _rt_flags(d, "nback")


# ----------------------------------------------------------------------------------- Stroop
def load_stroop() -> pd.DataFrame:
    d = _attach_session(_read("stroop"))
    d = d.rename(columns={"Block Type": "variant_raw", "Word": "word", "ink Color": "ink_colour",
                          "Response": "response", "Correct": "correct_recorded",
                          "ReactionTime_ms": "rt_ms"})
    d = _round_and_sound(d, "variant_raw")
    d["emotional_valence"] = d["variant_raw"].where(d["variant_raw"].isin(["Positive", "Negative", "Neutral"]))
    d["rt_ms"] = d["rt_ms"].astype(float)
    d["correct_recorded"] = d["correct_recorded"].astype(int)
    d["correct"] = (d["response"].str.lower() == d["ink_colour"].str.lower()).astype(int)
    d["score_discrepancy"] = d["correct"] != d["correct_recorded"]
    d["word_is_colour_name"] = d["word"].str.upper().isin({"RED", "GREEN", "BLUE", "YELLOW"})
    d["congruent"] = d["word"].str.upper() == d["ink_colour"].str.upper()
    keys = ["participant", "session", "task"]
    d = d.sort_values("src_row")
    d["run"] = 0
    for _, g in d[d["exclusion"] == ""].groupby(keys):
        d.loc[g.index, "run"] = _runs_by_gap(g, 60)
    d = _finalise_runs(d, keys)
    return _rt_flags(d, "stroop")


# ----------------------------------------------------------------------------------- visual
def load_visual(items: pd.DataFrame | None = None) -> pd.DataFrame:
    d = _attach_session(_read("visual"))
    d = d.rename(columns={"Image_Details": "image", "Question": "question", "Response": "response",
                          "Correct": "correct_recorded", "ReactionTime_ms": "rt_ms"})
    d["variant_raw"] = np.where(d["image"].str.contains(r"_(?:same|dif)\.", regex=True), "visualShape", "visualCount")
    d = _round_and_sound(d, "variant_raw")
    d["rt_ms"] = d["rt_ms"].astype(float)
    d["correct_recorded"] = d["correct_recorded"].astype(int)
    key = d["image"].str.extract(r"_(\d+|same|dif)\.", expand=False)
    d["key_nominal"] = key.replace({"dif": "DIFFERENT", "same": "SAME"})
    count = d["task"] == "visualcount"
    d["count_response"] = pd.to_numeric(d["response"], errors="coerce").where(count)
    d["count_key_nominal"] = pd.to_numeric(d["key_nominal"], errors="coerce").where(count)
    d["correct"] = np.where(count, (d["count_response"] == d["count_key_nominal"]).astype(int),
                            (d["response"].str.upper() == d["key_nominal"]).astype(int))
    d["score_discrepancy"] = d["correct"] != d["correct_recorded"]
    keys = ["participant", "session", "task"]
    d = d.sort_values("src_row")
    d["run"] = 0
    for _, g in d[d["exclusion"] == ""].groupby(keys):
        seen, run, runs = set(), 1, []          # a restarted block repeats an image
        for im in g["image"]:
            if im in seen:
                run, seen = run + 1, set()
            seen.add(im)
            runs.append(run)
        d.loc[g.index, "run"] = runs
    d = _finalise_runs(d, keys)
    d = _visual_consensus(d)
    if items is not None:
        d = d.merge(items[["image", "key_validity"]], on="image", how="left")
    return _rt_flags(d, "visual")


def _visual_consensus(d: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-participant-out consensus scores for the counting items.

    For each counting trial, the consensus is the median response of all *other* participants
    (analysable rows, both sessions) to the same image; ``count_log_ratio_consensus`` =
    ln(response / consensus).  This avoids reliance on the nominal (generation-target) key and
    is not circular for the scored participant."""
    d["count_consensus_loo"] = np.nan
    ok = (d["exclusion"] == "") & (d["task"] == "visualcount") & d["count_response"].gt(0)
    for img, g in d[ok].groupby("image"):
        for pid in g["participant"].unique():
            others = g.loc[g["participant"] != pid, "count_response"]
            d.loc[g.index[g["participant"] == pid], "count_consensus_loo"] = others.median()
    d["count_log_ratio_nominal"] = np.log(d["count_response"].where(d["count_response"] > 0) / d["count_key_nominal"])
    d["count_log_ratio_consensus"] = np.log(d["count_response"].where(d["count_response"] > 0) / d["count_consensus_loo"])
    return d


# ----------------------------------------------------------------------------------- flanker
def load_flanker() -> pd.DataFrame:
    d = _attach_session(_read("flanker"))
    d = d.rename(columns={"Task_Type": "variant_raw", "Condition": "condition_recorded",
                          "Stimulus_Displayed": "stimulus", "Target": "target",
                          "Response": "response", "Correct": "correct_recorded",
                          "ResponseTime_ms": "rt_ms"})
    d = _round_and_sound(d, "variant_raw")
    d["rt_ms"] = d["rt_ms"].astype(float)
    d["correct_recorded"] = d["correct_recorded"].astype(int)
    elems = d["stimulus"].str.split()
    centre, flank = elems.str[2], elems.str[0]
    arrow = d["task"] == "flankerarrow"
    d["expected_response"] = np.where(arrow, centre.map({"<": "Left", ">": "Right"}),
                                      np.where(pd.to_numeric(centre, errors="coerce") % 2 == 0, "Even", "Odd"))

    def _cond(f, c, is_arrow):
        if f in {"+", "#"}:
            return "Neutral"
        if is_arrow:
            return "Congruent" if f == c else "Incongruent"
        try:
            return "Congruent" if int(f) % 2 == int(c) % 2 else "Incongruent"
        except (TypeError, ValueError):
            return "Unknown"
    d["condition"] = [_cond(f, c, a) for f, c, a in zip(flank, centre, arrow)]
    d["correct"] = (d["response"] == d["expected_response"]).astype(int)
    d["score_discrepancy"] = d["correct"] != d["correct_recorded"]
    keys = ["participant", "session", "task"]
    d = d.sort_values("src_row")
    d["run"] = 0
    for _, g in d[d["exclusion"] == ""].groupby(keys):
        d.loc[g.index, "run"] = _runs_by_gap(g, 60)
    d = _finalise_runs(d, keys)
    return _rt_flags(d, "flanker")


# ----------------------------------------------------------------------------------- events
def load_events() -> pd.DataFrame:
    d = _attach_session(_read("events"))
    d = d.rename(columns={"Event_Type": "event_type", "Event_Detail": "event_detail",
                          "Start_Time": "start_utc", "End_Time": "end_utc", "Duration_ms": "duration_ms"})
    d["start_unix"] = local_to_unix(pd.to_datetime(d["start_utc"], utc=True))
    d["end_unix"] = local_to_unix(pd.to_datetime(d["end_utc"], utc=True))
    d["duration_ms"] = d["duration_ms"].astype(float)
    return d


def load_all(items: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    return {"nback": load_nback(), "stroop": load_stroop(), "visual": load_visual(items),
            "flanker": load_flanker()}


# ----------------------------------------------------------------------------------- configuration
def task_configuration(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Reconstruct the task configuration that ran in each session from the analysed runs
    (decision D17): trials per n-back level, levels reached, Stroop paradigm, flanker trials."""
    rows = []
    s = sessions()
    for r in s.itertuples():
        sel = {k: v[(v.participant == r.participant) & (v.session == r.session) & (v.exclusion == "")]
               for k, v in tables.items()}
        nb, st, vi, fl = sel["nback"], sel["stroop"], sel["visual"], sel["flanker"]
        per_level = nb.groupby(["task", "level"]).size()
        max_trial = nb.groupby(["task", "level"])["trial"].max()
        rows.append({
            "participant_id": r.participant_id, "session_id": r.session_id,
            "cfg_nback_trials_per_level": int(max_trial.max()) if len(max_trial) else np.nan,
            "nback_levels_reached": ";".join(f"{t}:{','.join(map(str, sorted(g.index.get_level_values(1))))}"
                                             for t, g in per_level.groupby(level=0)),
            "cfg_nback_lists": "fixed (pilot lists)" if r.session == "A" and r.participant in ("P01", "P02") else "generated per session",
            "cfg_stroop_paradigm": ";".join(sorted(st["task"].unique())),
            "stroop_trials": ";".join(f"{t}:{n}" for t, n in st.groupby("task").size().items()),
            "visual_trials": ";".join(f"{t}:{n}" for t, n in vi.groupby("task").size().items()),
            "cfg_flanker_trials_per_variant": int(fl.groupby("task").size().max()) if len(fl) else np.nan,
            "flanker_trials": ";".join(f"{t}:{n}" for t, n in fl.groupby("task").size().items()),
            "n_superseded_rows": int(sum(((v.participant == r.participant) & (v.session == r.session) &
                                          (v.exclusion == "superseded_run")).sum() for v in tables.values())),
        })
    return pd.DataFrame(rows)
