"""EEG technical validation: readability with MNE-BIDS, usable-epoch fractions and baseline power spectra."""
from __future__ import annotations

import warnings

import mne
import mne_bids
import numpy as np
import pandas as pd

from .common import BIDS, CHANNELS, GEN, SESSIONS, tsv

EPOCH_S = 2.0
RAIL_UV = 999.0          # the Muse S saturates at +-1000 uV (values are stored at 1000/2048 uV resolution)
PTP_UV = 200.0           # peak-to-peak criterion for a usable 2-s epoch after 1-Hz high-pass
FMIN, FMAX = 1.0, 45.0
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


def read(sub: str, ses: str) -> mne.io.BaseRaw:
    path = mne_bids.BIDSPath(root=BIDS, subject=sub[4:], session=ses[4:], task="cogbattery", datatype="eeg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mne_bids.read_raw_bids(path, verbose="ERROR")


def _epochs(x: np.ndarray, fs: float) -> np.ndarray:
    n = int(EPOCH_S * fs)
    k = x.shape[1] // n
    return x[:, : k * n].reshape(x.shape[0], k, n)


def session_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per session and channel: fraction of usable 2-s epochs (whole recording and baseline) and the baseline PSD.

    Epochs that contain a recording gap (BIDS 'boundary' event) are excluded. An epoch is usable when no sample is at
    the device rail and its peak-to-peak amplitude after a 1-Hz high-pass filter is below 200 uV.
    """
    rows, spectra, hist = [], [], []
    edges = np.logspace(0, np.log10(5000), 61)
    for sub, ses in SESSIONS:
        raw = read(sub, ses).load_data()
        fs = raw.info["sfreq"]
        data_raw = raw.get_data() * 1e6
        hp = raw.copy().filter(l_freq=1.0, h_freq=None, verbose="ERROR").get_data() * 1e6
        gap = np.zeros(raw.n_times, bool)
        for a in raw.annotations:
            if a["description"].lower().startswith(("bad", "edge", "boundary")):
                i0 = int(a["onset"] * fs)
                gap[max(0, i0 - int(fs)): i0 + int(fs)] = True       # +-1 s around each discontinuity
        ev = tsv(f"{sub}/{ses}/eeg/{sub}_{ses}_task-cogbattery_events.tsv")
        base = ev[ev.trial_type == "baseline"].iloc[0]
        i0, i1 = int(base.onset * fs), int(min(base.onset + base.duration, raw.times[-1]) * fs)
        for scope, (a, b) in {"recording": (0, raw.n_times), "baseline": (i0, i1)}.items():
            e_raw, e_hp, e_gap = _epochs(data_raw[:, a:b], fs), _epochs(hp[:, a:b], fs), _epochs(gap[None, a:b], fs)[0]
            railed = (np.abs(e_raw) >= RAIL_UV).any(axis=2)
            ptp = np.ptp(e_hp, axis=2)
            ok = (~railed) & (ptp < PTP_UV) & (~e_gap.any(axis=1))[None, :]
            if scope == "recording":
                valid = ~e_gap.any(axis=1)
                for ci, ch in enumerate(CHANNELS):
                    v = np.where(railed[ci], edges[-1] * 0.999, ptp[ci])[valid]
                    hist.append(pd.DataFrame({"participant_id": sub, "session_id": ses, "channel": ch, "bin_lo": edges[:-1],
                                              "bin_hi": edges[1:], "count": np.histogram(v, edges)[0]}))
            for ci, ch in enumerate(CHANNELS):
                rows.append({"participant_id": sub, "session_id": ses, "scope": scope, "channel": ch,
                             "n_epochs": int((~e_gap.any(axis=1)).sum()), "usable_fraction": float(ok[ci].sum() / max(1, (~e_gap.any(axis=1)).sum())),
                             "railed_fraction": float(railed[ci].mean())})
            if scope == "baseline":
                for ci, ch in enumerate(CHANNELS):
                    good = e_hp[ci][ok[ci]]
                    if len(good) < 10:
                        continue
                    f, p = _welch(good, fs)
                    spectra.append(pd.DataFrame({"participant_id": sub, "session_id": ses, "channel": ch,
                                                 "freq": f, "psd_uv2_hz": p, "n_epochs": len(good)}))
    m, s = pd.DataFrame(rows), pd.concat(spectra, ignore_index=True)
    GEN.mkdir(parents=True, exist_ok=True)
    m.to_csv(GEN / "eeg_epoch_quality.csv", index=False)
    s.to_csv(GEN / "eeg_baseline_psd.csv", index=False)
    pd.concat(hist, ignore_index=True).to_csv(GEN / "eeg_ptp_histogram.csv", index=False)
    return m, s


def _welch(epochs: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean Hann-windowed periodogram of usable 2-s epochs (0.5-Hz resolution)."""
    n = epochs.shape[1]
    w = np.hanning(n)
    x = (epochs - epochs.mean(axis=1, keepdims=True)) * w
    p = (np.abs(np.fft.rfft(x, axis=1)) ** 2).mean(axis=0) * 2 / (fs * (w ** 2).sum())
    f = np.fft.rfftfreq(n, 1 / fs)
    keep = (f >= FMIN) & (f <= FMAX)
    return f[keep], p[keep]


def alpha_peaks(spectra: pd.DataFrame) -> pd.DataFrame:
    """Individual alpha peak: maximum of the 1/f-detrended log spectrum within 7-13 Hz, if it exceeds the fit by >= 1 dB."""
    rows = []
    for (sub, ses, ch), d in spectra.groupby(["participant_id", "session_id", "channel"]):
        f, logp = d.freq.to_numpy(), 10 * np.log10(d.psd_uv2_hz.to_numpy())
        fit_band = ((f >= 2) & (f < 7)) | ((f > 14) & (f <= 30))
        coef = np.polyfit(np.log10(f[fit_band]), logp[fit_band], 1)
        resid = logp - np.polyval(coef, np.log10(f))
        band = (f >= 7) & (f <= 13)
        k = np.argmax(resid[band])
        rows.append({"participant_id": sub, "session_id": ses, "channel": ch, "peak_hz": f[band][k],
                     "peak_db": resid[band][k], "has_peak": resid[band][k] >= 1.0})
    return pd.DataFrame(rows)
