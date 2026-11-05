"""Step 05: ActiGraph LEAP -> BIDS physio (ppg, acc, skin temperature) per session, vendor-derived
series (IBI, HR, HRV) as derivatives, device table and coverage QC."""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synapse_curation import paths, wearable  # noqa: E402
from synapse_curation.common import NA, sessions, unix_to_iso, write_json, write_tsv  # noqa: E402

B = paths.BIDS
TASK = "cogbattery"
STREAMS = {  # name -> (fs, columns, units, descriptions)
    "ppg": (25.0, ["ppg_green_ch0", "ppg_ambient_ch0", "ppg_green_ch1", "ppg_ambient_ch1"], "ADC counts",
            {"ppg_green_ch0": "Green-LED photoplethysmogram, optical channel 0 (raw ADC; +-2096920 = saturation/invalid sentinel)",
             "ppg_ambient_ch0": "Ambient-light photodiode signal, optical channel 0 (raw ADC)",
             "ppg_green_ch1": "Green-LED photoplethysmogram, optical channel 1 (raw ADC; +-2096920 = sentinel)",
             "ppg_ambient_ch1": "Ambient-light photodiode signal, optical channel 1 (raw ADC)"}),
    "acc": (32.0, ["acc_x", "acc_y", "acc_z"], "g",
            {"acc_x": "Acceleration, device x axis", "acc_y": "Acceleration, device y axis", "acc_z": "Acceleration, device z axis"}),
    "skintemp": (1 / 12, ["skin_temperature"], "degC", {"skin_temperature": "Skin temperature at the wrist (device sensor)"}),
}


def write_physio(df: pd.DataFrame, name: str, folder: Path, base: str, extra: dict) -> None:
    fs, cols, unit, desc = STREAMS[name]
    folder.mkdir(parents=True, exist_ok=True)
    tsv = folder / f"{base}_recording-{name}_physio.tsv.gz"
    with open(tsv, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
        gz.write(df[cols].to_csv(sep="\t", header=False, index=False, na_rep=NA, lineterminator="\n").encode())
    write_json({"SamplingFrequency": fs, "StartTime": 0, "Columns": cols, "TaskName": "SYNAPSE cognitive battery",
                "PhysioType": "generic",
                **{c: {"Description": desc[c], "Units": unit} for c in cols},
                "Manufacturer": "ActiGraph", "ManufacturersModelName": "LEAP v2", **extra},
               folder / f"{base}_recording-{name}_physio.json")


def main() -> None:
    mem = wearable.members()
    s = sessions()
    dev_rows, cov_rows, avail = [], [], []
    for p in sorted(s.participant.unique()):
        pid = f"sub-{p}"
        if p not in mem:
            for ses in ("A", "B"):
                cov_rows.append({"participant_id": pid, "session_id": f"ses-{ses}", "status": "missing (device worn, no data retrieved; D14)"})
            continue
        m = mem[p]
        h = wearable.header(m["acc"])
        st = wearable.load_streams(m)
        dev_rows.append({"participant_id": pid, "device": h.get("device"), "serial_number": h.get("serial"),
                         "firmware": h.get("firmware"), "actilife_version": h.get("actilife"),
                         "programmed_start_local": f"{h['Start Date']} {h['Start Time']}",
                         "download_utc": f"{h['Download Date']} {h['Download Time']}",
                         "wrist": "non-dominant (D13)", "archive": f"raw/leap_csv/{m['acc'][0]}"})
        wear = wearable.wear_flag(st["skintemp"])
        t_worn = float(st["skintemp"]["t"].to_numpy()[wear.argmax()]) if wear.any() else np.nan
        for ses, (w0, w1) in wearable.session_windows(p).items():
            ses_id = f"ses-{ses}"
            folder = B / pid / ses_id / "beh"
            base = f"{pid}_{ses_id}_task-{TASK}"
            row = {"participant_id": pid, "session_id": ses_id, "status": "available",
                   "window_start": unix_to_iso([w0])[0], "window_end": unix_to_iso([w1])[0]}
            for name in STREAMS:
                d = st[name]
                d = d[(d.t >= w0) & (d.t < w1)].reset_index(drop=True)
                reg = wearable.regularity(d["t"].to_numpy(), STREAMS[name][0])
                if reg["n_irregular"]:
                    raise SystemExit(f"{pid} {ses_id} {name}: irregular sampling")
                extra = {"SYNAPSE_StartTimeLocal": unix_to_iso([d.t.iloc[0]])[0],
                         "SYNAPSE_StartTimeUnix": float(d.t.iloc[0]),
                         "SYNAPSE_TimeBase": "Device clock (programmed in ActiLife from the host PC; not synchronised with other devices, D11). Sample k is at StartTime + k/SamplingFrequency.",
                         "SYNAPSE_Placement": "non-dominant wrist (D13)", "SYNAPSE_DeviceSerial": h.get("serial"),
                         "SYNAPSE_Window": "Device recording trimmed to [session start - 15 min, midpoint of break] (session A) or [midpoint of break, session end + 15 min] (session B)"}
                if name == "ppg":
                    sent = (d[STREAMS["ppg"][1]].abs() == wearable.PPG_SENTINEL).any(axis=1).mean()
                    extra["SYNAPSE_SentinelFraction"] = round(float(sent), 5)
                write_physio(d, name, folder, base, extra)
                row[f"{name}_start"] = unix_to_iso([d.t.iloc[0]])[0]
                row[f"{name}_samples"] = len(d)
            st_w = st["skintemp"][(st["skintemp"].t >= w0) & (st["skintemp"].t < w1)]
            row["worn_from"] = unix_to_iso([max(t_worn, w0)])[0] if np.isfinite(t_worn) and t_worn < w1 else NA
            row["skin_temperature_median"] = round(float(st_w.loc[st_w.skin_temperature >= 30, "skin_temperature"].median()), 2)
            # vendor derivatives
            der = B / "derivatives" / "wristband_vendor_metrics" / pid / ses_id
            ibi = st["ibi"][(st["ibi"].t >= w0) & (st["ibi"].t < w1)]
            write_tsv(pd.DataFrame({"time": unix_to_iso(ibi.t), "time_unix": ibi.t.round(3), "ibi_s": ibi.ibi_s}),
                      der / f"{base}_desc-ibi_timeseries.tsv")
            hr = st["hr"][(st["hr"].t >= w0) & (st["hr"].t < w1)]
            write_tsv(pd.DataFrame({"time": unix_to_iso(hr.t, ms=False), "time_unix": hr.t, "heart_rate_bpm": hr.heart_rate_bpm}),
                      der / f"{base}_desc-heartrate_timeseries.tsv")
            hv = st["hrv"][(st["hrv"].t >= w0) & (st["hrv"].t < w1)].copy()
            hv.insert(0, "window_end", unix_to_iso(hv.pop("t_end")))
            hv.insert(0, "window_start", unix_to_iso(hv.pop("t")))
            write_tsv(hv, der / f"{base}_desc-hrv_timeseries.tsv")
            row["ibi_n"] = len(ibi)
            row["hrv_reliability_median"] = float(hv["hrv_reliability"].median())
            cov_rows.append(row)
            avail.append({"participant_id": pid, "session_id": ses_id, "has_wristphysio": True})
    der = B / "derivatives" / "wristband_vendor_metrics"
    write_json({"Name": "ActiGraph vendor-derived cardiac series", "BIDSVersion": "1.10.0", "DatasetType": "derivative",
                "GeneratedBy": [{"Name": "ActiLife", "Version": "7.1.0", "Description": "Vendor algorithms (proprietary); exported as CSV and re-packaged per session without modification"}],
                "SourceDatasets": [{"URL": "bids::"}]}, der / "dataset_description.json")
    write_tsv(pd.DataFrame(dev_rows), B / "derivatives" / "quality_control" / "wearable_devices.tsv")
    write_tsv(pd.DataFrame(cov_rows), B / "derivatives" / "quality_control" / "wearable_coverage.tsv")
    pd.DataFrame(avail).to_pickle(paths.WORK / "availability_physio.pkl")
    print(pd.DataFrame(cov_rows)[["participant_id", "session_id", "status", "worn_from", "skin_temperature_median", "hrv_reliability_median"]].to_string())


if __name__ == "__main__":
    main()
