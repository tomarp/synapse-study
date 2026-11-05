"""Room monitor (ATMOCUBE, 1-min records; decision D15).

Source: ``raw/atmo/full_experiment.xlsx`` sheet ``OG_After-June24`` (9235 records, 20-27 June 2025).
Sheet ``Useful`` is a derived copy that omits one valid record (24 June 10:23:37) and is not used.
``ts`` (Unix s) is authoritative.  ``ts_cest`` is an Excel formula that references the next row
(labels shifted by one record, the last row reads 1970-01-01) and is not used.  ``ch2o`` is 0 in
every record (no formaldehyde reading) and is not distributed.
"""
from __future__ import annotations

import pandas as pd

from . import paths
from .common import unix_to_iso

SHEET = "OG_After-June24"
VARS = {  # raw -> (output column, unit, description)
    "t": ("air_temperature", "degC", "Air temperature (Sensirion STS35)"),
    "h": ("relative_humidity", "%", "Relative humidity (Sensirion SHTC3)"),
    "abs_h": ("absolute_humidity", "g/m3", "Absolute humidity, device-computed (integer)"),
    "p": ("air_pressure", "hPa", "Barometric pressure (Infineon DPS310)"),
    "co2": ("co2", "ppm", "CO2 concentration (Sensirion SCD41, automatic baseline correction)"),
    "pm1": ("pm1", "ug/m3", "PM1 mass concentration (Sensirion SPS30)"),
    "pm25": ("pm2_5", "ug/m3", "PM2.5 mass concentration (Sensirion SPS30)"),
    "pm4": ("pm4", "ug/m3", "PM4 mass concentration (Sensirion SPS30)"),
    "pm10": ("pm10", "ug/m3", "PM10 mass concentration (Sensirion SPS30)"),
    "voc": ("tvoc", "ppm", "Total VOC, manufacturer conversion of the SGP41 signal"),
    "voc_index": ("voc_index", "index points (1-500)", "Sensirion VOC index (relative to the sensor's 24-h baseline; 100 = average)"),
    "nox_index": ("nox_index", "index points (1-500)", "Sensirion NOx index (1 = baseline)"),
    "noise": ("sound_level", "dB", "Background sound pressure level, device microphone (Knowles SPH0641), 1-min value; frequency weighting not documented; does not capture headphone playback"),
    "light": ("illuminance", "lx", "Illuminance from the device colour/light sensor (AMS TCS34725) at the device position; uncalibrated, orientation not documented; relative indicator only"),
}


def load() -> pd.DataFrame:
    d = pd.read_excel(paths.RAW / "atmo" / "full_experiment.xlsx", sheet_name=SHEET)
    n_raw = len(d)
    d = d.dropna(subset=["ts"]).copy()
    d.attrs["dropped_blank_rows"] = n_raw - len(d)
    d["time_unix"] = d["ts"].astype(float)
    d = d.sort_values("time_unix").reset_index(drop=True)
    out = pd.DataFrame({"time": unix_to_iso(d["time_unix"], ms=False), "time_unix": d["time_unix"].astype(int)})
    for raw, (col, _, _) in VARS.items():
        out[col] = d[raw].to_numpy()
    out.attrs = d.attrs
    return out


def sidecar() -> dict:
    meta = {"time": {"Description": "Record time, ISO 8601 local (CEST, UTC+02:00)"},
            "time_unix": {"Description": "Record time, Unix seconds (UTC); authoritative device timestamp", "Units": "s"}}
    for col, unit, desc in VARS.values():
        meta[col] = {"Description": desc, "Units": unit}
    return meta
