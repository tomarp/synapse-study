"""Per participant-session phase timeline (Unix seconds; reconstructed from platform data).

  baseline        [T0 submission, +300 s]  5-min eyes-open rest (D23); truncated at T1 if earlier
  T1              [baseline end, T1 submission]
  block           one per task: [first response - its RT, last response submission]
  round1 / round2 [first block start, last block end] of the round
  T2 / T3         [end of round 1 / round 2, form submission]
  soundscape_switch  [end of round 1, start of round 2]: interval in which the soundscape was
                  changed (not time-stamped; decision D19)

Response times are Google-server receipt times (1-s resolution, +network latency); onsets derived
from them are accurate to about +-2 s.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import sessions

BASELINE_S = 300.0


def build(beh: dict[str, pd.DataFrame], q: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for d in beh.values():
        d = d[d["exclusion"] == ""].copy()
        d["onset"] = d["submitted_unix"] - d["rt_ms"] / 1000.0
        parts.append(d.groupby(["participant", "session", "task", "round"]).agg(
            onset=("onset", "min"), offset=("submitted_unix", "max"), n_responses=("onset", "size"),
            soundscape=("soundscape", "first")).reset_index())
    blocks = pd.concat(parts, ignore_index=True)
    blocks["phase"] = "block"
    forms = {k: q[k][q[k]["exclusion"] == ""] for k in ("T0", "T1", "T2", "T3")}
    rows = []
    for r in sessions().itertuples():
        b = blocks[(blocks.participant == r.participant) & (blocks.session == r.session)]
        f = {k: v[(v.participant == r.participant) & (v.session == r.session)]["submitted_unix"]
             for k, v in forms.items()}
        f = {k: float(v.iloc[0]) for k, v in f.items() if len(v)}
        base = dict(participant=r.participant, session=r.session)
        r1, r2 = b[b["round"] == 1], b[b["round"] == 2]
        if "T0" in f:
            be = f["T0"] + BASELINE_S
            if "T1" in f:
                be = min(be, f["T1"])
            rows.append({**base, "phase": "T0", "onset": np.nan, "offset": f["T0"]})
            rows.append({**base, "phase": "baseline", "onset": f["T0"], "offset": be, "soundscape": "none"})
            if "T1" in f:
                rows.append({**base, "phase": "T1", "onset": be, "offset": f["T1"], "soundscape": "none"})
        rows.append({**base, "phase": "round1", "round": 1, "onset": r1.onset.min(), "offset": r1.offset.max(),
                     "soundscape": r.sound_round1})
        rows.append({**base, "phase": "round2", "round": 2, "onset": r2.onset.min(), "offset": r2.offset.max(),
                     "soundscape": r.sound_round2})
        rows.append({**base, "phase": "soundscape_switch", "onset": r1.offset.max(), "offset": r2.onset.min(),
                     "soundscape": "undetermined"})
        if "T2" in f:
            rows.append({**base, "phase": "T2", "onset": r1.offset.max(), "offset": f["T2"], "soundscape": "undetermined"})
        if "T3" in f:
            rows.append({**base, "phase": "T3", "onset": r2.offset.max(), "offset": f["T3"], "soundscape": r.sound_round2})
    tl = pd.concat([pd.DataFrame(rows), blocks], ignore_index=True)
    tl["duration_s"] = tl["offset"] - tl["onset"]
    order = {"T0": 0, "baseline": 1, "T1": 2, "round1": 3, "block": 4, "soundscape_switch": 5, "T2": 6, "round2": 7, "T3": 8}
    tl["_o"] = tl["phase"].map(order)
    tl = tl.sort_values(["participant", "session", "onset", "_o"], na_position="first").drop(columns="_o")
    tl.insert(0, "participant_id", "sub-" + tl.pop("participant"))
    tl.insert(1, "session_id", "ses-" + tl.pop("session"))
    return tl.reset_index(drop=True)


def session_span(tl: pd.DataFrame) -> pd.DataFrame:
    """Data-derived session start (T0 submission) and end (T3 submission or last response)."""
    g = tl.groupby(["participant_id", "session_id"])
    start = g.apply(lambda x: x.loc[x.phase == "T0", "offset"].min(), include_groups=False)
    end = g.apply(lambda x: np.nanmax([x.loc[x.phase == "T3", "offset"].max(), x["offset"].max()]), include_groups=False)
    return pd.DataFrame({"session_start_unix": start, "session_end_unix": end}).reset_index()
