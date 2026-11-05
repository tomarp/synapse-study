"""ActiGraph LEAP v2 wrist device (firmware 2.1.0, ActiLife 7.1.0 exports).

Level-0 inputs: ``raw/leap_csv/<day>.zip`` (ActiLife CSV exports; the vendor binary .agsd/.agd
files in the same archives and in ``raw/leap_raw`` are byte-identical and are inventoried only).

Streams distributed as BIDS physio (uniformly sampled, values exactly as exported):
  ppg   25 Hz   green and ambient photodiode, two optical channels (ADC counts);
                +-2096920 is the device saturation/invalid sentinel and is kept as exported
  acc   32 Hz   tri-axial acceleration (g), ActiLife 'Normal' filter
  skintemp 1/12 Hz  skin temperature (degC)
Vendor-derived series (irregular or windowed) are distributed as derivatives:
  inter-beat intervals, 1-min heart rate, 300-s HRV windows (30-s step) with reliability.
Vendor rhythm classifications are not distributed (decision D25).

Time base: data rows are naive local time (CEST); vendor-derived files carry explicit offsets.
Header 'Download Time' is UTC.  The 23 June archive (pilot day) is excluded (D02).
Each device recording spans both sessions of a participant; it is split at the midpoint of the
break and trimmed to [session start - 15 min, session end + 15 min] (decision in README).
"""
from __future__ import annotations

import io
import re
import zipfile

import numpy as np
import pandas as pd

from . import paths
from .common import local_to_unix, sessions

ZIPS = ["Tuesday-Afternoon.zip", "Wednesday.zip", "Thursday.zip"]
MARGIN_S = 900.0
PPG_SENTINEL = 2096920


def members() -> dict[str, dict[str, tuple[str, str]]]:
    out: dict[str, dict[str, tuple[str, str]]] = {}
    for z in ZIPS:
        with zipfile.ZipFile(paths.RAW / "leap_csv" / z) as zf:
            for m in zf.namelist():
                parts = m.split("/")
                pid = re.match(r"(P\d\d)", parts[1]) if len(parts) >= 3 else None
                if not pid:
                    continue
                name = parts[-1]
                kind = ("ibi" if name == "InterBeatInterval.csv" else "hr" if name == "HeartRate.csv" else
                        "hrv" if name == "HeartRateVar.csv" else "temp" if name.endswith("Temperature.csv") else
                        "acc" if name.endswith("RAW.csv") else "ppg" if name.endswith("ppg25Hz.csv") else None)
                if kind:
                    out.setdefault(pid.group(1), {})[kind] = (z, m)
    return out


def _bytes(zm: tuple[str, str]) -> bytes:
    with zipfile.ZipFile(paths.RAW / "leap_csv" / zm[0]) as zf:
        return zf.read(zm[1])


def header(zm: tuple[str, str]) -> dict:
    head = _bytes(zm)[:2048].decode("latin-1").replace("\r", "").split("\n")
    meta = {"title": head[0].strip("- ")}
    m = re.search(r"ActiGraph (.+?) ActiLife v([\d.]+) Firmware v([\d.]+)", head[0])
    if m:
        meta.update(device=m.group(1), actilife=m.group(2), firmware=m.group(3))
    for line in head[1:12]:
        if line.startswith("Serial Number:"):
            meta["serial"] = line.split(":", 1)[1].strip()
        for k in ("Start Time", "Start Date", "Download Time", "Download Date"):
            if line.startswith(k):
                meta[k] = line[len(k):].strip()
    return meta


def _actilife(zm: tuple[str, str]) -> pd.DataFrame:
    raw = _bytes(zm)
    head = raw[:4096].decode("latin-1").replace("\r", "").split("\n")
    skip = next(i for i, line in enumerate(head) if line.startswith("Timestamp,"))
    d = pd.read_csv(io.BytesIO(raw), skiprows=skip)
    d["t"] = local_to_unix(pd.to_datetime(d["Timestamp"], format="%m/%d/%Y %H:%M:%S.%f"))
    return d


def _vendor(zm: tuple[str, str]) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_bytes(zm)))


def _utc(s: pd.Series) -> np.ndarray:
    return local_to_unix(pd.to_datetime(s, format="ISO8601", utc=True))


def load_streams(m: dict[str, tuple[str, str]]) -> dict[str, pd.DataFrame]:
    acc = _actilife(m["acc"]).rename(columns={"Accelerometer X": "acc_x", "Accelerometer Y": "acc_y",
                                              "Accelerometer Z": "acc_z"})
    temp = _actilife(m["temp"]).rename(columns={"Temperature": "skin_temperature"})
    ppg = _actilife(m["ppg"])
    w = ppg.pivot_table(index="t", columns="Channel", values=["Green", "Ambient"], aggfunc="first")
    w.columns = [f"ppg_{a.lower()}_ch{c}" for a, c in w.columns]
    ppg = w.reset_index()[["t", "ppg_green_ch0", "ppg_ambient_ch0", "ppg_green_ch1", "ppg_ambient_ch1"]]
    ibi = _vendor(m["ibi"])
    ibi = pd.DataFrame({"t": _utc(ibi["timestamputc_ibi"]), "ibi_s": ibi["interbeat_interval"]})
    hr = _vendor(m["hr"])
    hr = pd.DataFrame({"t": _utc(hr["timestamputc"]), "heart_rate_bpm": hr["HeartRate"]})
    hrv = _vendor(m["hrv"])
    hrv.insert(0, "t_end", _utc(hrv["hrv_epoch_end"]))
    hrv.insert(0, "t", _utc(hrv["hrv_epoch_start"]))
    hrv = hrv.drop(columns=["start", "end", "hrv_epoch_start", "hrv_epoch_end"])
    return {"acc": acc[["t", "acc_x", "acc_y", "acc_z"]], "skintemp": temp[["t", "skin_temperature"]],
            "ppg": ppg, "ibi": ibi, "hr": hr, "hrv": hrv}


def regularity(t: np.ndarray, fs: float) -> dict:
    dt = np.diff(t)
    nominal = 1.0 / fs
    return {"n": len(t), "dt_median_s": float(np.median(dt)), "dt_max_s": float(dt.max()),
            "n_irregular": int(np.sum(np.abs(dt - nominal) > 0.25 * nominal)),
            "effective_fs": float((len(t) - 1) / (t[-1] - t[0]))}


def session_windows(participant: str) -> dict[str, tuple[float, float]]:
    s = sessions()
    s = s[s.participant == participant].set_index("session")
    mid = (s.loc["A", "log_end_unix"] + s.loc["B", "log_start_unix"]) / 2
    return {"A": (s.loc["A", "log_start_unix"] - MARGIN_S, mid),
            "B": (mid, s.loc["B", "log_end_unix"] + MARGIN_S)}


def wear_flag(temp: pd.DataFrame, threshold_c: float = 30.0) -> np.ndarray:
    """Heuristic wear indicator from skin temperature: worn once the 1-min rolling median exceeds
    the threshold (the device sits at room temperature ~24-27 degC before donning)."""
    med = temp["skin_temperature"].rolling(5, center=True, min_periods=1).median()
    return (med >= threshold_c).to_numpy()
