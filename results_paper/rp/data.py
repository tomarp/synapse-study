"""Analysis tables built from the published dataset: sessions, trials, ratings and physiology per phase.

Condition labels always come from the session table (``sessions()``), so that the blinding in ``common.scramble``
reaches every outcome. Coding (ANALYSIS_PLAN.md, section 2): L cool +1/2, S traffic +1/2, O second session +1/2,
R round 2 +1/2.
"""
from __future__ import annotations

import gzip
import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from .common import BIDS, EARLY, PAIR, STRATEGY_BLOCKS, SUBS, TASKS, UNBLIND, scramble, tsv

PHASES = ("baseline", "round1", "round2")


@lru_cache(maxsize=None)
def timeline() -> pd.DataFrame:
    return tsv("derivatives/session_timeline/timeline.tsv")


@lru_cache(maxsize=None)
def room() -> pd.DataFrame:
    return tsv("environment/room_monitor_1min.tsv")


@lru_cache(maxsize=None)
def sessions_true() -> pd.DataFrame:
    s = pd.concat([tsv(f"{p}/{p}_sessions.tsv").assign(participant_id=p) for p in SUBS], ignore_index=True)
    r = room()
    lux = []
    for _, x in s.iterrows():
        w = r[(r.time_unix >= pd.Timestamp(x.session_start).timestamp()) & (r.time_unix <= pd.Timestamp(x.session_end).timestamp())]
        lux.append(w.illuminance.median())
    s["illuminance_median"] = lux
    s["slot"] = [f"{PAIR[p]}-{ses}" for p, ses in zip(s.participant_id, s.session_id)]
    return s


@lru_cache(maxsize=None)
def sessions() -> pd.DataFrame:
    """Session table with condition codes; condition labels scrambled unless SYNAPSE_UNBLIND=1."""
    s = sessions_true() if UNBLIND else scramble(sessions_true())
    s = s.copy()
    s["L"] = np.where(s.light == "cool", 0.5, -0.5)
    s["O"] = np.where(s.session_order == 2, 0.5, -0.5)
    s["lux_z"] = (s.illuminance_median - s.illuminance_median.mean()) / s.illuminance_median.std()
    return s


def _session_codes() -> dict[tuple[str, str], pd.Series]:
    return {(x.participant_id, x.session_id): x for _, x in sessions().iterrows()}


def soundscape_code(name: str) -> float:
    return {"traffic": 0.5, "people": -0.5}[name]


# ------------------------------------------------------------------------------------------------- trials
@lru_cache(maxsize=None)
def trials() -> pd.DataFrame:
    """One row per trial of the analysed tasks (and shape comparison, descriptive only), superseded runs removed."""
    rows = []
    for f in sorted(BIDS.glob("sub-*/ses-*/beh/*_beh.tsv")):
        task = re.search(r"task-([a-z]+)_", f.name)[1]
        if task not in TASKS + ["visualshape"]:
            continue
        d = tsv(f)
        d = d[d.superseded_run.astype(str).str.lower() != "true"].copy()
        d["participant_id"], d["session_id"], d["task"] = f.parts[-4], f.parts[-3], task
        d["pos"] = np.arange(len(d))                               # trial position within the block
        rows.append(d)
    t = pd.concat(rows, ignore_index=True)
    # the soundscape of each trial in the dataset must equal the session table's round mapping (true labels)
    st = {(x.participant_id, x.session_id): x for _, x in sessions_true().iterrows()}
    expect = [st[(p, s)][f"soundscape_round{int(r)}"] for p, s, r in zip(t.participant_id, t.session_id, t["round"])]
    assert (t.soundscape.values == np.array(expect)).all(), "trial soundscape differs from the session table"
    sc = _session_codes()
    t["soundscape"] = [sc[(p, s)][f"soundscape_round{int(r)}"] for p, s, r in zip(t.participant_id, t.session_id, t["round"])]
    t["L"] = [sc[(p, s)].L for p, s in zip(t.participant_id, t.session_id)]
    t["O"] = [sc[(p, s)].O for p, s in zip(t.participant_id, t.session_id)]
    t["slot"] = [sc[(p, s)].slot for p, s in zip(t.participant_id, t.session_id)]
    t["lux_z"] = [sc[(p, s)].lux_z for p, s in zip(t.participant_id, t.session_id)]
    t["S"] = t.soundscape.map(soundscape_code)
    t["R"] = np.where(t["round"] == 2, 0.5, -0.5)
    t["cell"] = t.task + ":" + np.select(
        [t.task.str.startswith("nback"), t.task.str.startswith("flanker")],
        ["L" + t.level.fillna(0).astype(int).astype(str) + np.where(t.is_target.astype(str).str.lower() == "true", "T", "N"),
         t.condition.astype(str)], "all")
    t["log_pos"] = np.log1p(t.pos)
    t["log_pos"] -= t.groupby("task").log_pos.transform("mean")
    t["rt_ok"] = t.rt_flag == "ok"
    t["strategy"] = [(p, s, k) in STRATEGY_BLOCKS for p, s, k in zip(t.participant_id, t.session_id, t.task)]
    t["early"] = [(p, s) in EARLY for p, s in zip(t.participant_id, t.session_id)]
    t["log_rt"] = np.log(t.rt_ms.where(t.rt_ms > 0) / 1000)
    return t


# ------------------------------------------------------------------------------------------------ ratings
@lru_cache(maxsize=None)
def ratings() -> pd.DataFrame:
    """Long table: one row per participant, session, questionnaire (T1-T3) and item code."""
    sc = _session_codes()
    rows = []
    for q, tp in (("T1_ieq", "T1"), ("T2_task_affect_ieq", "T2"), ("T3_task_affect_ieq", "T3")):
        for ses in "AB":
            d = tsv(f"phenotype/{q}_session{ses}.tsv")
            for _, x in d.iterrows():
                key = (x.participant_id, x.session_id)
                c = sc[key]
                snd = None if tp == "T1" else c[f"soundscape_round{1 if tp == 'T2' else 2}"]
                for col in d.columns:
                    if col.endswith("_code") and pd.notna(x[col]):
                        rows.append(dict(participant_id=key[0], session_id=key[1], time=tp, item=col[:-5],
                                         value=float(x[col]), link_quality=x.get("link_quality"),
                                         L=c.L, O=c.O, slot=c.slot, lux_z=c.lux_z,
                                         snd=0.0 if snd is None else 1.0,
                                         S=0.0 if snd is None else soundscape_code(snd),
                                         R={"T1": 0.0, "T2": -0.5, "T3": 0.5}[tp]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- physiology
def _phase_windows() -> pd.DataFrame:
    t = timeline()
    w = t[t.phase.isin(PHASES)][["participant_id", "session_id", "phase", "onset_unix", "offset_unix"]]
    return w.reset_index(drop=True)


def _clean_ibi(ibi: pd.DataFrame) -> pd.DataFrame:
    """Plausible range 0.33-1.5 s and within 20 % of the local median of the 11 surrounding intervals.

    Deviation from ANALYSIS_PLAN.md (D1), decided before unblinding: the plan compared each interval with the previous
    accepted one, which locks onto an early artefact and then rejects all following beats."""
    v = ibi.ibi_s.where(ibi.ibi_s.between(0.33, 1.5))
    med = v.rolling(11, center=True, min_periods=3).median()
    return ibi[v.notna() & ((v - med).abs() <= 0.2 * med)]


def _wrist_start(p: str, ses: str, rec: str) -> tuple[float, float]:
    j = json.loads((BIDS / f"{p}/{ses}/beh/{p}_{ses}_task-cogbattery_recording-{rec}_physio.json").read_text())
    return j["SYNAPSE_StartTimeUnix"], j["SamplingFrequency"]


@lru_cache(maxsize=None)
def physiology() -> pd.DataFrame:
    """One row per session and phase: heart rate, baseline ln RMSSD, wrist skin temperature, facial temperature,
    room conditions; with the condition codes of the phase."""
    sc = _session_codes()
    r = room()
    out = []
    for _, w in _phase_windows().iterrows():
        p, ses, ph, t0, t1 = w.participant_id, w.session_id, w.phase, w.onset_unix, w.offset_unix
        c = sc[(p, ses)]
        rec = dict(participant_id=p, session_id=ses, phase=ph, duration_s=t1 - t0, L=c.L, O=c.O, slot=c.slot,
                   lux_z=c.lux_z, task=0.0 if ph == "baseline" else 1.0,
                   S=0.0 if ph == "baseline" else soundscape_code(c[f"soundscape_round{ph[-1]}"]),
                   R={"baseline": 0.0, "round1": -0.5, "round2": 0.5}[ph])
        # heart rate and HRV from the vendor inter-beat intervals
        f = BIDS / f"derivatives/wristband_vendor_metrics/{p}/{ses}/{p}_{ses}_task-cogbattery_desc-ibi_timeseries.tsv"
        if f.exists():
            ibi = tsv(f)
            ibi = ibi[(ibi.time_unix >= t0) & (ibi.time_unix < t1)]
            acc = _clean_ibi(ibi)
            rec["ibi_coverage"] = acc.ibi_s.sum() / (t1 - t0)
            if rec["ibi_coverage"] >= 0.5:
                rec["hr"] = float((60 / acc.ibi_s).mean())
                tt, v = acc.time_unix.values, acc.ibi_s.values
                consecutive = np.abs(np.diff(tt) - v[1:]) < 0.25           # adjacent accepted beats
                d = np.diff(v)[consecutive]
                if ph == "baseline" and len(d) >= 30:
                    rec["ln_rmssd"] = float(np.log(1000 * np.sqrt(np.mean(d ** 2))))
        # wrist skin temperature (1/12 Hz)
        f = BIDS / f"{p}/{ses}/beh/{p}_{ses}_task-cogbattery_recording-skintemp_physio.tsv.gz"
        if f.exists():
            start, fs = _wrist_start(p, ses, "skintemp")
            with gzip.open(f, "rt") as h:
                v = np.loadtxt(h)
            tt = start + np.arange(len(v)) / fs
            m = (tt >= t0) & (tt < t1)
            if m.sum() >= 5:
                rec["skin_temp"] = float(v[m].mean())
        # facial temperature (relative; valid snapshots only)
        f = BIDS / f"derivatives/facial_temperature/{p}/{ses}/{p}_{ses}_task-cogbattery_desc-facetemp_timeseries.tsv"
        if not f.exists():
            f = next((BIDS / f"derivatives/facial_temperature/{p}/{ses}").glob("*desc-facetemp_timeseries.tsv"), None)
        if f is not None and f.exists():
            ft = tsv(f)
            ft = ft[(ft.time_unix >= t0) & (ft.time_unix < t1) & (ft.valid.astype(str).str.lower() == "true")]
            if len(ft) >= 5:
                rec["face_temp_bg"] = float(ft.T_median_minus_bg.mean())
                rec["face_temp"] = float(ft.T_median.mean())
        # room conditions
        rw = r[(r.time_unix >= t0) & (r.time_unix < t1)]
        for k in ("air_temperature", "relative_humidity", "co2", "sound_level", "illuminance"):
            rec[f"room_{k}"] = float(rw[k].mean()) if len(rw) else np.nan
        out.append(rec)
    return pd.DataFrame(out)
