"""Muse S (BrainFlow) EEG: inventory, lossless de-duplication, assembly per participant-session,
dropout ("stall") detection and BrainVision export.

Acquisition facts (raw code/muse/muse-eeg.py, June 2025): every ~100 ms the script called
``board.get_current_board_data(500)`` and appended the whole 500-sample window to
``<ID>_eeg_data.csv``; on a clean exit it also wrote ``<ID>_eeg_data_full_session.csv``.  Every
sample is therefore stored ~18 times, restarts append to the same file, and the participant ID was
edited by hand (stale IDs occur).  When the headset stopped streaming, the script kept re-writing
the same frozen 500-sample buffer ("stall"), so an abnormally high copy count at the end of a
segment marks the start of a dropout.

Curation rules
  1. A sample is identified by its BrainFlow host timestamp (float64 Unix s, laptop clock).
     Exact duplicates are collapsed; duplicates with different values are conflicts (none allowed).
  2. Values are microvolts as delivered (quantised in steps of 1000/2048 uV, range -1000..999.51);
     they are written as int16 BrainVision with that resolution, i.e. bit-exact.
  3. Continuous segments are split at host-time gaps > GAP_S; within a segment consecutive samples
     are consecutive device samples (no polling loss, verified), so the sample index is exact.
     Segments are concatenated in the BrainVision file and every discontinuity is marked by a
     'boundary' event carrying the true gap duration.
  4. No filtering, re-referencing, resampling or interpolation.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from . import paths
from .common import local_to_unix, sha256, unix_to_iso

CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
FS = 256.0
RESOLUTION_UV = 1000.0 / 2048.0
GAP_S = 0.5
RAIL_UV = 999.5
CACHE = paths.WORK / "eeg_unique"


def raw_files() -> list:
    return sorted(p for sub in ("eeg", "muse") for p in (paths.RAW / sub).rglob("*.csv"))


def read_unique(path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (N x 5 array [t, TP9, AF7, AF8, TP10] sorted by t, copies per sample, stats)."""
    tbl = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(
        column_types={c: pa.float64() for c in ["Timestamp", *CHANNELS]}))
    d = np.column_stack([tbl[c].to_numpy(zero_copy_only=False) for c in ["Timestamp", *CHANNELS]])
    del tbl
    n_rows = len(d)
    finite = np.isfinite(d).all(axis=1)
    d = d[finite]
    d = d[np.lexsort((np.arange(len(d)), d[:, 0]))]
    same_t = np.r_[False, np.diff(d[:, 0]) == 0]
    diff_v = np.r_[False, np.any(d[1:, 1:] != d[:-1, 1:], axis=1)]
    conflicts = int(np.sum(same_t & diff_v))
    first = np.flatnonzero(~same_t)
    copies = np.diff(np.r_[first, len(d)])
    u = d[first]
    q = np.abs(u[:, 1:] / RESOLUTION_UV - np.round(u[:, 1:] / RESOLUTION_UV)).max() if len(u) else 0.0
    return u, copies.astype(np.int32), {
        "rows": n_rows, "nonfinite_rows": int((~finite).sum()), "unique_samples": len(u),
        "duplicate_factor": n_rows / max(len(u), 1), "conflicting_duplicates": conflicts,
        "max_quantisation_error_steps": float(q),
        "t_first": float(u[0, 0]) if len(u) else np.nan, "t_last": float(u[-1, 0]) if len(u) else np.nan,
    }


@dataclass
class Segment:
    i0: int          # first sample index
    i1: int          # last sample index (inclusive)
    start: float
    end: float

    @property
    def n(self) -> int:
        return self.i1 - self.i0 + 1


def segments(t: np.ndarray, gap: float = GAP_S) -> list[Segment]:
    if len(t) == 0:
        return []
    br = np.flatnonzero(np.diff(t) > gap)
    s, e = np.r_[0, br + 1], np.r_[br, len(t) - 1]
    return [Segment(int(a), int(b), float(t[a]), float(t[b])) for a, b in zip(s, e)]


# ----------------------------------------------------------------------------------- inventory
def inventory() -> pd.DataFrame:
    """Hash every raw EEG file, cache unique samples per distinct content, cluster into streams."""
    paths.ensure(CACHE)
    rows, stats_by_hash = [], {}
    for p in raw_files():
        h = sha256(p)
        rel = "raw/" + p.relative_to(paths.RAW).as_posix()
        if h not in stats_by_hash:
            npz = CACHE / f"{h[:16]}.npz"
            if npz.exists():
                z = np.load(npz, allow_pickle=True)
                stats = z["stats"].item()
            else:
                u, copies, stats = read_unique(p)
                np.savez(npz, u=u, copies=copies, stats=np.array(stats, dtype=object))
            stats_by_hash[h] = stats
            print(f"  {rel}: {stats['unique_samples']} unique samples, x{stats['duplicate_factor']:.1f}", flush=True)
        rows.append({"source": rel, "bytes": p.stat().st_size, "sha256": h, **stats_by_hash[h]})
    man = pd.DataFrame(rows)
    man["duplicate_of"] = man.groupby("sha256")["source"].transform("first")
    man.loc[man["duplicate_of"] == man["source"], "duplicate_of"] = ""
    # streams: distinct contents sharing >= 1000 samples with >= 99 % identical values.  Two
    # concurrent headsets on one host share a few exact timestamps by chance, never values.
    hashes = list(stats_by_hash)
    arr = {h: np.load(CACHE / f"{h[:16]}.npz")["u"] for h in hashes}
    parent = {h: h for h in hashes}

    def find(h):
        while parent[h] != h:
            parent[h] = parent[parent[h]]
            h = parent[h]
        return h
    for i, a in enumerate(hashes):
        for b in hashes[i + 1:]:
            ta, tb = arr[a][:, 0], arr[b][:, 0]
            if ta[-1] < tb[0] or tb[-1] < ta[0]:
                continue
            common, ia, ib = np.intersect1d(ta, tb, assume_unique=True, return_indices=True)
            if common.size >= 1000 and np.mean(np.all(arr[a][ia, 1:] == arr[b][ib, 1:], axis=1)) >= 0.99:
                parent[find(a)] = find(b)
    man["stream"] = man["sha256"].map(lambda h: find(h)[:8])
    return man


def load_unique(sha: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(CACHE / f"{sha[:16]}.npz")
    return z["u"], z["copies"]


# ----------------------------------------------------------------------------------- assembly
def assemble(man: pd.DataFrame, assign: pd.DataFrame) -> dict[tuple[str, str], dict]:
    """One de-duplicated recording per participant-session from the assignment table."""
    canon = man.drop_duplicates("sha256").set_index("source")
    by_src = man.set_index("source")
    assigned = assign[assign["participant_id"] != "n/a"]
    # every distinct content must be assigned, excluded, or wholly contained in an assigned one
    listed = set(assign["source"].map(lambda s: "raw/" + s))
    for src, r in canon.iterrows():
        if src in listed:
            continue
        t = load_unique(r.sha256)[0][:, 0]
        if not any(np.isin(t, load_unique(by_src.loc[a, "sha256"])[0][:, 0], assume_unique=True).all()
                   for a in listed if a in by_src.index):
            raise SystemExit(f"EEG content not accounted for by the assignment table: {src}")
    out = {}
    for (pid, ses), g in assigned.groupby(["participant_id", "session_id"]):
        parts, cps = [], []
        for r in g.itertuples():
            u, c = load_unique(by_src.loc["raw/" + r.source, "sha256"])
            lo, hi = local_to_unix([r.window_start_local, r.window_end_local])
            m = (u[:, 0] >= lo) & (u[:, 0] <= hi)
            parts.append(u[m])
            cps.append(c[m])
        d, c = np.concatenate(parts), np.concatenate(cps)
        o = np.lexsort((np.arange(len(d)), d[:, 0]))
        d, c = d[o], c[o]
        dup = np.r_[False, np.diff(d[:, 0]) == 0]
        if (dup & np.r_[False, np.any(d[1:, 1:] != d[:-1, 1:], axis=1)]).any():
            raise SystemExit(f"{pid} {ses}: conflicting samples across assigned files")
        keep = ~dup
        # copies of samples present in several assigned files are summed per file; keep the max
        d, c = d[keep], c[keep]
        out[(pid, ses)] = {"data": d, "copies": c, "segments": segments(d[:, 0]),
                           "sources": sorted("raw/" + s for s in g["source"]),
                           "evidence": sorted(set(g["evidence_level"])),
                           "evidence_text": " | ".join(g["evidence"])}
    return out


def gap_table(rec: dict, pid: str, ses: str) -> pd.DataFrame:
    """Every discontinuity with its duration and whether it was preceded by a buffer stall."""
    segs, c = rec["segments"], rec["copies"]
    typical = float(np.median(c))
    rows = []
    for a, b in zip(segs, segs[1:]):
        tail = c[max(a.i0, a.i1 - 499): a.i1 + 1]
        excess_polls = max(0.0, float(tail.max()) - typical)
        rows.append({"participant_id": pid, "session_id": ses,
                     "gap_start": unix_to_iso([a.end])[0], "gap_end": unix_to_iso([b.start])[0],
                     "gap_duration_s": round(b.start - a.end, 3),
                     "stall_before_gap": excess_polls > 5 * max(1.0, 0.1 * typical),
                     "stall_extra_buffer_writes": int(excess_polls)})
    # a stall at the very end of the recording = dropout that ended the recording
    tail = c[max(0, segs[-1].i1 - 499):]
    if tail.max() - typical > 5 * max(1.0, 0.1 * typical):
        rows.append({"participant_id": pid, "session_id": ses, "gap_start": unix_to_iso([segs[-1].end])[0],
                     "gap_end": "n/a", "gap_duration_s": np.nan, "stall_before_gap": True,
                     "stall_extra_buffer_writes": int(tail.max() - typical)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------- export
def host_to_file_time(rec: dict, t_unix: np.ndarray) -> np.ndarray:
    """Map host timestamps to seconds from the start of the concatenated BrainVision file.
    Times inside a gap map to the boundary; times outside the recording give NaN."""
    t = rec["data"][:, 0]
    idx = np.searchsorted(t, t_unix).astype(float)
    idx[(t_unix < t[0]) | (t_unix > t[-1])] = np.nan
    return idx / FS


def write_brainvision(rec: dict, folder, base: str, extra_events: list[dict] | None = None) -> None:
    from pybv import write_brainvision as _wbv
    d = rec["data"]
    ev = [{"onset": s.i0, "duration": 1, "description": "boundary", "type": "Comment"}
          for s in rec["segments"][1:]]
    ev += extra_events or []
    _wbv(data=(d[:, 1:].T * 1e-6), sfreq=FS, ch_names=CHANNELS, ref_ch_names="Fpz",
         fname_base=base, folder_out=folder, overwrite=True, events=sorted(ev, key=lambda e: e["onset"]),
         resolution=RESOLUTION_UV, unit="µV", fmt="binary_int16",
         meas_date=pd.Timestamp(d[0, 0], unit="s", tz="UTC").floor("us").to_pydatetime())
    # pybv scales in float32 and truncates; the samples lie exactly on the 1000/2048 uV grid, so the
    # payload is rewritten with explicit rounding (multiplexed little-endian int16, as in the header)
    q = np.round(d[:, 1:] / RESOLUTION_UV)
    if np.abs(q * RESOLUTION_UV - d[:, 1:]).max() > 0 or q.min() < -32768 or q.max() > 32767:
        raise ValueError("EEG samples are not representable exactly as int16 at the stated resolution")
    q.astype("<i2").tofile(folder / f"{base}.eeg")


def write_hosttime(rec: dict, path) -> float:
    """Host timestamp of every EEG sample (BIDS-physio style .tsv.gz, no header), relative to the first sample.

    Row i belongs to sample i of the BrainVision file. Returns the Unix time of the first sample."""
    t = rec["data"][:, 0]
    rel = np.round(t - t[0], 6)
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
        gz.write(("\n".join(f"{v:.6f}" for v in rel) + "\n").encode())
    return float(t[0])


def qc_row(rec: dict, pid: str, ses: str, session_start: float, session_end: float) -> dict:
    d, segs = rec["data"], rec["segments"]
    t = d[:, 0]
    rail = np.abs(d[:, 1:]) >= RAIL_UV
    in_sess = (t >= session_start) & (t <= session_end)
    covered = sum(max(0.0, min(s.end, session_end) - max(s.start, session_start)) for s in segs)
    seg_rates = [(s.n - 1) / (s.end - s.start) for s in segs if s.end - s.start > 60]
    return {
        "participant_id": pid, "session_id": ses,
        "recording_start": unix_to_iso([t[0]])[0], "recording_end": unix_to_iso([t[-1]])[0],
        "n_samples": len(d), "n_segments": len(segs),
        "gap_total_s": round(sum(b.start - a.end for a, b in zip(segs, segs[1:])), 2),
        "session_coverage_fraction": round(covered / (session_end - session_start), 4),
        "within_segment_rate_hz_median": round(float(np.median(seg_rates)), 4) if seg_rates else np.nan,
        **{f"rail_fraction_{c}": round(float(rail[in_sess, i].mean()), 4) for i, c in enumerate(CHANNELS)},
        "assignment_evidence": ";".join(rec["evidence"]),
        "source_files": ";".join(rec["sources"]),
    }
