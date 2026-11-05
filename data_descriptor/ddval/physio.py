"""Wrist-device validation: heart rate derived independently from the raw PPG versus the device's own heart rate."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

from .common import GEN, SESSIONS, physio, session_window, vendor

SENTINEL = 2096920          # raw ADC value marking saturated/invalid PPG samples (see the PPG sidecars)
CHANNEL = "ppg_green_ch0"
BAND_HZ = (0.7, 3.5)        # 42-210 bpm
MIN_BEATS = 20              # minimum valid beats in a 1-min window


def ppg_beats(p: pd.DataFrame, channel: str = CHANNEL) -> tuple[np.ndarray, np.ndarray]:
    """Beat times (Unix s) and inter-beat intervals from one green PPG channel.

    Sentinel samples are interpolated, the signal is band-pass filtered (3rd-order Butterworth, zero phase) and
    pulse peaks are detected with a 0.33-s refractory distance and a prominence of 0.3 SD.
    """
    fs = p.attrs["fs"]
    x = p[channel].to_numpy(float).copy()
    x[np.abs(x) >= SENTINEL] = np.nan
    x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    y = sosfiltfilt(butter(3, BAND_HZ, "bandpass", fs=fs, output="sos"), x)
    peaks, _ = find_peaks(y, distance=int(0.33 * fs), prominence=0.3 * np.nanstd(y))
    t = p["time_unix"].to_numpy()[peaks]
    return t[1:], np.diff(t)


def heart_rate_agreement() -> pd.DataFrame:
    """Per 1-min window inside each session: device heart rate and the median-IBI heart rate from the raw PPG."""
    rows = []
    for sub, ses in SESSIONS:
        p = physio(sub, ses, "ppg")
        if p is None:
            continue
        t0, t1 = session_window(sub, ses)
        tb, ibi = ppg_beats(p)
        ok = (ibi > 0.33) & (ibi < 1.5)
        hr = vendor(sub, ses, "heartrate")
        n_minutes = int(((hr.time_unix >= t0) & (hr.time_unix + 60 <= t1)).sum())
        hr = hr.dropna(subset=["heart_rate_bpm"])
        hr = hr[(hr.time_unix >= t0) & (hr.time_unix + 60 <= t1)]
        for r in hr.itertuples():
            m = (tb >= r.time_unix) & (tb < r.time_unix + 60) & ok
            rows.append({"participant_id": sub, "session_id": ses, "time_unix": r.time_unix, "n_session_minutes": n_minutes,
                         "hr_device": r.heart_rate_bpm, "n_beats": int(m.sum()),
                         "hr_ppg": 60 / np.median(ibi[m]) if m.sum() >= MIN_BEATS else np.nan})
    d = pd.DataFrame(rows)
    GEN.mkdir(parents=True, exist_ok=True)
    d.to_csv(GEN / "ppg_heart_rate_agreement.csv", index=False)
    return d


def wrist_coverage() -> pd.DataFrame:
    """Fraction of each session (baseline onset to end of T3) covered by valid PPG samples."""
    rows = []
    for sub, ses in SESSIONS:
        t0, t1 = session_window(sub, ses)
        p = physio(sub, ses, "ppg")
        if p is None:
            rows.append({"participant_id": sub, "session_id": ses, "ppg_coverage": 0.0})
            continue
        w = p[(p.time_unix >= t0) & (p.time_unix <= t1)]
        valid = (w[CHANNEL].abs() < SENTINEL).sum()
        rows.append({"participant_id": sub, "session_id": ses, "ppg_coverage": min(1.0, valid / ((t1 - t0) * p.attrs["fs"]))})
    return pd.DataFrame(rows)
