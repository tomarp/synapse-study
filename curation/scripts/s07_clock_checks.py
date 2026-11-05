"""Step 07: post-hoc clock consistency checks between devices (decision D11).  Clocks were not
synchronised during the study; offsets cannot be measured directly, so they are bounded with
independent timing anchors.  Nothing is corrected in the distributed timestamps.

  A  task laptop vs Google server: server receipt time - client page-end time of the same
     platform event (= clock offset + network latency, 1-s server resolution)
  B  EEG acquisition laptop vs Google server: last EEG sample - T3 form submission (the operator
     stopped the recording after T3; a negative value would indicate a clock offset or early stop)
  C  EEG laptop vs operator log: end of each recording gap vs logged 'EEG restarted' minute
  D  wrist device vs ActiLife PC: download time (UTC header) - last sample
  E  movement cross-correlation EEG vs wrist acceleration (attempted; reported only if identifiable)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synapse_curation import behavior, paths, wearable  # noqa: E402
from synapse_curation.common import local_to_unix, write_tsv  # noqa: E402

QC = paths.BIDS / "derivatives" / "quality_control"


def main() -> None:
    rows = []
    ev = behavior.load_events()
    ev = ev[ev.exclusion == ""]
    for (p, s), g in ev.groupby(["participant", "session"]):
        lag = g["submitted_unix"] - g["end_unix"]
        rows.append({"check": "A_taskPC_vs_server", "participant_id": f"sub-{p}", "session_id": f"ses-{s}", "n": len(g),
                     "median_s": round(lag.median(), 2), "min_s": round(lag.min(), 2), "max_s": round(lag.max(), 2),
                     "interpretation": "server - client (offset + latency); |offset| <= max(|min|, |max|) + 1 s"})
    eq = pd.read_csv(QC / "eeg_quality.tsv", sep="\t")
    t3 = pd.concat([pd.read_csv(f, sep="\t") for f in sorted((paths.BIDS / "phenotype").glob("T3_task_affect_ieq_session*.tsv"))])
    m = eq.merge(t3[["participant_id", "session_id", "submitted_time"]], on=["participant_id", "session_id"], how="left")
    gaps = pd.read_csv(QC / "eeg_gaps.tsv", sep="\t")
    for r in m.itertuples():
        d = local_to_unix([pd.Timestamp(r.recording_end)])[0] - local_to_unix([pd.Timestamp(r.submitted_time)])[0]
        dropout = gaps[(gaps.participant_id == r.participant_id) & (gaps.session_id == r.session_id) & gaps.gap_end.eq("n/a")]
        rows.append({"check": "B_EEGlaptop_end_minus_T3", "participant_id": r.participant_id, "session_id": r.session_id, "n": 1,
                     "median_s": round(d, 1), "min_s": np.nan, "max_s": np.nan,
                     "interpretation": "EEG ended by dropout before T3 (not informative)" if len(dropout) else
                                       "positive = recording stopped after T3 (consistent clocks)"})
    for pid, ses, logged in [("sub-P02", "ses-B", "2025-06-24 14:22"), ("sub-P02", "ses-B", "2025-06-24 14:30"),
                             ("sub-P03", "ses-B", "2025-06-24 17:06"), ("sub-P04", "ses-B", "2025-06-24 17:06")]:
        g = gaps[(gaps.participant_id == pid) & (gaps.session_id == ses) & gaps.gap_end.ne("n/a")].copy()
        g["end_unix"] = local_to_unix(pd.to_datetime(g.gap_end, format="ISO8601"))
        t_log = local_to_unix([pd.Timestamp(logged)])[0]
        k = (g.end_unix - (t_log + 30)).abs().idxmin()
        rows.append({"check": "C_EEG_restart_vs_log", "participant_id": pid, "session_id": ses, "n": 1,
                     "median_s": round(g.loc[k, "end_unix"] - t_log, 1), "min_s": 0, "max_s": 60,
                     "interpretation": f"recording resumed (EEG clock) minus logged restart minute {logged[-5:]}; consistent if within [0, 60] s"})
    for p, mem in wearable.members().items():
        h = wearable.header(mem["acc"])
        dl = pd.Timestamp(f"{h['Download Date']} {h['Download Time']}", tz="UTC").timestamp()
        last = wearable.load_streams(mem)["skintemp"]["t"].max()
        rows.append({"check": "D_wrist_download_minus_last_sample", "participant_id": f"sub-{p}", "session_id": "n/a", "n": 1,
                     "median_s": round(dl - last, 1), "min_s": np.nan, "max_s": np.nan,
                     "interpretation": "ActiLife PC (UTC) download - last device sample; positive and small = device clock consistent with the PC (device stopped on download)"})
    rows.append({"check": "E_movement_xcorr_EEG_vs_wrist", "participant_id": "n/a", "session_id": "n/a", "n": 3,
                 "median_s": np.nan, "min_s": np.nan, "max_s": np.nan,
                 "interpretation": "Attempted on sub-P07/ses-A, sub-P10/ses-B, sub-P03/ses-A (0.25-s envelopes, lags +-240 s): no identifiable peak (robust z <= 2.3); offset not estimable by this method"})
    out = pd.DataFrame(rows)
    write_tsv(out, QC / "clock_checks.tsv")
    print(out.groupby("check")["median_s"].describe().round(1).to_string())


if __name__ == "__main__":
    main()
