"""Step 05 - FLIR AX8 snapshot curation: scale OCR, palette inversion, seat ROIs,
occupancy, vacancy events, validation and QC figures.

Inputs : RAW/thermal/<folder>/*_snapshot_NNNN.png, *_flir_ax8_recording.avi (read-only)
Outputs: CURATED/thermal/thermal_frames.csv         one row per snapshot
         CURATED/thermal/thermal_seats.csv          one row per snapshot x seat
         CURATED/thermal/seat_vacancy_events.csv    vacancy intervals >= 30 s per folder x seat
         CURATED/thermal/thermal_streams.csv        per stream timing + AVI evidence
         CURATED/thermal/ocr_manual_check.csv       OCR vs visual transcription
         CURATED/thermal/occupancy_visual_census.csv occupancy vs complete visual census
         CURATED/thermal/other_person_episodes.csv  experimenter visits etc. (nearest seat)
         CURATED/thermal/thermal_validation.json
         FIGURES/thermal_qc/*.png
Temp   : $SYNAPSE_THERMAL_TMP (palette LUT, per-frame cache)

Usage  : SYNAPSE_THERMAL_TMP=/path/to/tmp python scripts/s05_thermal.py [--reuse] [--workers N]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synapse_curation import paths  # noqa: E402
from synapse_curation import thermal as th  # noqa: E402

RAW = paths.RAW / "thermal"
OUT = paths.WORK / "thermal"
FIG = paths.WORK / "thermal" / "qc_figures"
CACHE = th.env_cache_dir()

DE_MASK = 12.0          # Delta E76 above which a pixel is not a palette colour (MSX lines, text, artefacts)
BAR_DE_MAX = 20.0       # frame rejected if its colour bar deviates from the canonical palette by more
SHIFT_MAX_PX = 10.0     # frame flagged as camera moved if registered shift vs its folder/pose reference exceeds this
                        # (seat boxes carry >= 25 px margin; quarter-resolution registration noise is ~1-3 px)
SHIFT_MIN_RESP = 0.15   # phase-correlation response needed to trust a shift estimate
SCENE_MAX_SHARE = 0.5   # face ROI holding >= this share of the frame's top-plateau pixels -> MAX label is on the face
SEED = 20250624
WARM_C = 4.0            # "human-warm" = >= 4 C above the frame median (cf. Otsu occupancy threshold 3.9 C)
ZONE_ROWS = (32, 96)    # standing-head zone: above all seated-head search boxes (y0 >= 100), below the labels
UPPER_WARM_MAX = 0.004  # frame flagged "other person present" if warm fraction in the zone exceeds this ...
SCENE_WARM_MAX = 0.30   # ... or if more than this fraction of the scene is human-warm (person close to camera)
SPOT_CHECK_C = 0.5      # per-frame inversion check: |spot residual - stream rolling median| (spot on background)
STEP_C = 0.5            # concurrent spot & background step (C) flagged as possible camera offset step
CORRIDOR_WARM_MAX = 0.02  # ... or warm fraction in the corridor regions beside the seats (seated-head height)
LOW_CONTRAST_FRAC = 0.35  # frame with scene p90-p10 below this fraction of the stream's rolling median

# ---------------------------------------------------------------------------------------------
# Visual ground truth (transcribed by eye from 3x-enlarged label crops, blind to OCR output).
# Random sample: 72 frames drawn with numpy default_rng(20250624) from all 3,522 snapshots.
# Stratified sample: 16 frames with unusual digits (spot >= 27, max < 33 or >= 36, ...).
# ---------------------------------------------------------------------------------------------
OCR_MANUAL_RANDOM = {
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0045.png": ("25.1", "35.1", "24.0"),
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0052.png": ("25.1", "34.9", "23.7"),
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0151.png": ("25.2", "35.2", "24.2"),
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0283.png": ("25.1", "34.8", "23.7"),
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0325.png": ("25.2", "35.3", "24.2"),
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0340.png": ("25.3", "35.0", "24.2"),
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0393.png": ("25.1", "35.1", "24.0"),
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0395.png": ("24.9", "34.8", "23.7"),
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0489.png": ("24.1", "34.3", "23.4"),
    "P01B-P02B/2025-06-24_13-51-16_snapshot_0051.png": ("24.9", "35.1", "24.2"),
    "P01B-P02B/2025-06-24_13-51-16_snapshot_0127.png": ("25.1", "34.7", "24.1"),
    "P01B-P02B/2025-06-24_13-51-16_snapshot_0244.png": ("25.2", "34.8", "24.3"),
    "P01B-P02B/2025-06-24_13-51-16_snapshot_0293.png": ("24.9", "34.3", "23.8"),
    "P03A-P04A/2025-06-24_15-31-09_snapshot_0041.png": ("25.1", "34.4", "24.3"),
    "P03A-P04A/2025-06-24_15-31-09_snapshot_0062.png": ("29.5", "34.7", "24.8"),
    "P03A-P04A/2025-06-24_15-31-09_snapshot_0153.png": ("24.9", "34.2", "24.1"),
    "P03A-P04A/2025-06-24_15-31-09_snapshot_0172.png": ("25.1", "34.4", "24.2"),
    "P03A-P04A/2025-06-24_15-31-09_snapshot_0185.png": ("25.1", "34.0", "24.1"),
    "P03A-P04A/2025-06-24_15-31-09_snapshot_0209.png": ("25.4", "34.6", "24.2"),
    "P03A-P04A/2025-06-24_15-31-09_snapshot_0233.png": ("25.6", "34.8", "24.7"),
    "P03A-P04A/2025-06-24_15-31-09_snapshot_0257.png": ("25.3", "34.8", "24.5"),
    "P03A-P04A/2025-06-24_15-31-09_snapshot_0274.png": ("25.1", "34.3", "24.5"),
    "P03B-P04B/2025-06-24_16-38-31_snapshot_0015.png": ("25.1", "34.3", "24.3"),
    "P03B-P04B/2025-06-24_16-38-31_snapshot_0020.png": ("24.9", "34.1", "24.1"),
    "P03B-P04B/2025-06-24_16-38-31_snapshot_0038.png": ("25.7", "34.4", "24.7"),
    "P03B-P04B/2025-06-24_16-38-31_snapshot_0042.png": ("25.3", "34.3", "24.6"),
    "P03B-P04B/2025-06-24_16-38-31_snapshot_0105.png": ("25.2", "34.3", "24.2"),
    "P05A-P06A/2025-06-25_10-15-25_snapshot_0002.png": ("24.7", "35.0", "23.8"),
    "P05A-P06A/2025-06-25_10-35-37_snapshot_0010.png": ("23.9", "34.1", "23.2"),
    "P05A-P06A/2025-06-25_10-35-37_snapshot_0152.png": ("25.2", "35.6", "24.8"),
    "P05A-P06A/2025-06-25_10-35-37_snapshot_0207.png": ("24.8", "35.1", "24.1"),
    "P05A-P06A/2025-06-25_10-35-37_snapshot_0236.png": ("24.6", "34.7", "24.1"),
    "P05B-P06B/2025-06-25_11-30-42_snapshot_0110.png": ("25.3", "35.1", "24.7"),
    "P05B-P06B/2025-06-25_11-30-42_snapshot_0139.png": ("25.0", "34.9", "24.4"),
    "P05B-P06B/2025-06-25_11-30-42_snapshot_0239.png": ("25.0", "34.6", "24.2"),
    "P07A/2025-06-25_12-34-40_snapshot_0013.png": ("24.6", "33.5", "23.7"),
    "P07A/2025-06-25_12-34-40_snapshot_0039.png": ("25.1", "34.0", "24.3"),
    "P07A/2025-06-25_12-34-40_snapshot_0069.png": ("24.8", "33.5", "24.1"),
    "P07A/2025-06-25_12-34-40_snapshot_0116.png": ("25.5", "34.2", "24.8"),
    "P07A/2025-06-25_12-34-40_snapshot_0136.png": ("25.3", "33.3", "24.7"),
    "P07A/2025-06-25_12-34-40_snapshot_0254.png": ("25.0", "33.7", "24.2"),
    "P07A/2025-06-25_12-34-40_snapshot_0259.png": ("25.3", "33.3", "24.7"),
    "P07A/2025-06-25_12-34-40_snapshot_0272.png": ("25.5", "34.0", "24.7"),
    "P07B/2025-06-25_13-31-33_snapshot_0009.png": ("25.0", "32.5", "24.2"),
    "P07B/2025-06-25_13-31-33_snapshot_0024.png": ("25.3", "33.4", "24.4"),
    "P07B/2025-06-25_13-31-33_snapshot_0071.png": ("25.8", "33.6", "25.0"),
    "P07B/2025-06-25_13-31-33_snapshot_0077.png": ("25.5", "33.8", "24.9"),
    "P07B/2025-06-25_13-31-33_snapshot_0089.png": ("25.5", "33.3", "24.5"),
    "P07B/2025-06-25_13-31-33_snapshot_0195.png": ("25.2", "32.8", "24.2"),
    "P08A-P09A/2025-06-25_15-05-40_snapshot_0032.png": ("25.3", "33.9", "24.4"),
    "P08A-P09A/2025-06-25_15-05-40_snapshot_0133.png": ("25.5", "34.6", "24.7"),
    "P08A-P09A/2025-06-25_15-05-40_snapshot_0176.png": ("25.9", "34.9", "25.1"),
    "P08A-P09A/2025-06-25_15-05-40_snapshot_0203.png": ("25.2", "34.6", "24.5"),
    "P08A-P09A/2025-06-25_15-05-40_snapshot_0235.png": ("25.7", "35.0", "25.1"),
    "P08B-P09B/2025-06-25_16-15-26_snapshot_0009.png": ("25.4", "34.2", "24.5"),
    "P08B-P09B/2025-06-25_16-15-26_snapshot_0035.png": ("26.0", "35.0", "25.1"),
    "P08B-P09B/2025-06-25_16-15-26_snapshot_0076.png": ("25.1", "34.3", "24.0"),
    "P08B-P09B/2025-06-25_16-15-26_snapshot_0090.png": ("26.0", "35.2", "25.0"),
    "P08B-P09B/2025-06-25_16-15-26_snapshot_0111.png": ("25.8", "34.6", "24.7"),
    "P08B-P09B/2025-06-25_16-15-26_snapshot_0120.png": ("26.2", "34.8", "25.2"),
    "P08B-P09B/2025-06-25_16-15-26_snapshot_0206.png": ("26.0", "35.1", "25.3"),
    "P08B-P09B/2025-06-25_16-15-26_snapshot_0219.png": ("25.6", "34.8", "24.8"),
    "P10A/2025-06-26_13-01-58_snapshot_0004.png": ("24.9", "33.7", "23.8"),
    "P10A/2025-06-26_13-01-58_snapshot_0142.png": ("25.0", "34.3", "24.0"),
    "P10A/2025-06-26_13-01-58_snapshot_0144.png": ("25.1", "34.4", "24.1"),
    "P10A/2025-06-26_13-01-58_snapshot_0155.png": ("24.8", "34.1", "24.0"),
    "P10B/2025-06-26_14-11-42_snapshot_0007.png": ("25.3", "34.2", "24.2"),
    "P10B/2025-06-26_14-11-42_snapshot_0009.png": ("25.1", "34.1", "24.1"),
    "P10B/2025-06-26_14-11-42_snapshot_0083.png": ("25.3", "33.1", "24.2"),
    "P10B/2025-06-26_14-11-42_snapshot_0090.png": ("25.8", "34.0", "24.8"),
    "P10B/2025-06-26_14-11-42_snapshot_0092.png": ("25.3", "33.7", "24.4"),
    "P10B/2025-06-26_14-11-42_snapshot_0213.png": ("25.4", "34.1", "24.4"),
}
OCR_MANUAL_STRATIFIED = {
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0018.png": ("29.7", "35.0", "23.7"),   # also a template reference
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0174.png": ("24.8", "34.4", "23.3"),
    "P01B-P02B/2025-06-24_13-51-16_snapshot_0005.png": ("24.5", "34.6", "23.4"),
    "P01B-P02B/2025-06-24_13-51-16_snapshot_0060.png": ("32.9", "35.2", "24.1"),
    "P01B-P02B/2025-06-24_13-51-16_snapshot_0121.png": ("24.3", "34.0", "23.2"),
    "P07B/2025-06-25_13-31-33_snapshot_0055.png": ("25.1", "32.8", "24.3"),
    "P07B/2025-06-25_13-31-33_snapshot_0122.png": ("24.9", "32.4", "24.2"),
    "P07B/2025-06-25_13-31-33_snapshot_0133.png": ("25.4", "32.9", "24.9"),
    "P07B/2025-06-25_13-31-33_snapshot_0214.png": ("25.3", "32.8", "24.4"),
    "P08A-P09A/2025-06-25_15-05-40_snapshot_0340.png": ("29.0", "33.7", "24.4"),
    "P10A/2025-06-26_13-01-58_snapshot_0002.png": ("34.3", "34.8", "24.2"),
    "P10A/2025-06-26_13-01-58_snapshot_0107.png": ("29.3", "34.9", "23.9"),
    "P10B/2025-06-26_14-11-42_snapshot_0186.png": ("30.1", "34.3", "24.6"),
    "P10B/2025-06-26_14-11-42_snapshot_0187.png": ("29.9", "34.1", "24.2"),
    "P10B/2025-06-26_14-11-42_snapshot_0193.png": ("29.3", "34.1", "23.9"),
    "P10B/2025-06-26_14-11-42_snapshot_0197.png": ("29.8", "34.5", "24.6"),
}
# Visual census of occupancy: every snapshot of every folder was reviewed on 120x90 px contact sheets
# (scratch only, not distributed).  Participant seats were occupied in ALL snapshots except the ranges
# below (snapshot indices of the main stream, inclusive).  Unused seats (P07A/B left, P10A/B left) never
# held a participant; the experimenter was transiently present in them (episodes listed below).
VISUAL_VACANT = {("P08A-P09A", "2025-06-25_15-05-40", "L"): [(341, 345)],
                 ("P08A-P09A", "2025-06-25_15-05-40", "R"): [(340, 345)]}
# Snapshots in which a person other than the seated participants (experimenter; or a participant
# standing up) was visible in the scene, noted during the same review (approximate, +-1 snapshot).
VISUAL_OTHER_PERSON = {
    "2025-06-24_12-46-10": [(53, 53), (135, 138), (186, 190), (376, 379), (477, 478)],
    "2025-06-24_13-51-16": [(39, 39), (46, 47), (52, 62), (64, 72), (122, 122), (218, 218)],
    "2025-06-24_15-31-09": [(16, 17), (61, 63), (101, 101), (190, 191)],
    "2025-06-24_16-38-31": [(50, 52), (165, 166)],
    "2025-06-25_10-35-37": [(74, 75), (151, 152)],
    "2025-06-25_11-30-42": [(61, 63), (71, 75)],
    "2025-06-25_12-34-40": [(16, 17), (28, 28), (71, 71), (194, 195)],
    "2025-06-25_13-31-33": [(155, 156)],
    "2025-06-25_15-05-40": [(74, 82), (213, 214), (340, 340)],
    "2025-06-25_16-15-26": [(159, 162)],
    "2025-06-26_13-01-58": [(2, 3), (6, 9), (97, 98), (206, 207)],
    "2025-06-26_14-11-42": [(175, 176)],
}


def in_episode(stream_id: str, idx: int, pad: int = 0) -> bool:
    return any(lo - pad <= idx <= hi + pad for lo, hi in VISUAL_OTHER_PERSON.get(stream_id, []))


# ---------------------------------------------------------------------------------------------
# worker state (process pool)
# ---------------------------------------------------------------------------------------------
_W: dict = {}


def _init_pass1() -> None:
    _W["reader"] = th.DigitReader.from_references(RAW)
    _W["text"], _W["cross"] = th.overlay_masks()


def _pass1(path: str) -> dict:
    img = cv2.imread(path)
    r = _W["reader"].read(img)
    r["bar"] = th.bar_column(img).astype(np.float32)
    r["struct"] = th.small_structure(img, _W["text"]).astype(np.float16)
    return r


def _init_pass2(palette: th.Palette) -> None:
    _W["inv"] = th.Inverter(palette, CACHE)
    _W["text"], _W["cross"] = th.overlay_masks()
    _W["disc"] = th.spot_disc()


def _pass2(job: dict) -> tuple[dict, list[dict]]:
    img = cv2.imread(job["path"])
    T, de, phw = _W["inv"].temperature(img, job["ocr_min"], job["ocr_max"])
    text, cross, disc = _W["text"], _W["cross"], _W["disc"]
    valid = (de <= DE_MASK) & ~text & ~cross
    scene = ~text & ~cross
    tb = float(np.median(T[valid]))
    fr = dict(offpalette_frac=float((de[scene] > DE_MASK).mean()), T_background=tb,
              T_scene_p10=float(np.percentile(T[valid], 10)), T_scene_p90=float(np.percentile(T[valid], 90)))
    # other-person features: warm pixels at standing-head height (above every seated head box) and
    # warm fraction of the whole scene (a person close to the camera)
    zy0, zy1 = ZONE_ROWS
    zone = th.filtered_temperature(T[zy0:zy1], valid[zy0:zy1])
    zwarm = (zone >= tb + WARM_C) & valid[zy0:zy1]
    fr["upper_warm_frac"] = float(zwarm.mean())
    fr["upper_warm_cx"] = float(np.nonzero(zwarm)[1].mean()) if zwarm.any() else np.nan
    fr["scene_warm_frac"] = float(((T >= tb + WARM_C) & valid).sum() / max(valid.sum(), 1))
    cw, cn = 0, 0
    for x0, cy0, x1, cy1 in th.POSE_CORRIDORS[job["pose"]]:
        t_ = th.filtered_temperature(T[cy0:cy1, x0:x1], valid[cy0:cy1, x0:x1])
        v_ = valid[cy0:cy1, x0:x1]
        cw += int(((t_ >= tb + WARM_C) & v_).sum()); cn += int(v_.sum())
    fr["corridor_warm_frac"] = cw / max(cn, 1)
    sv = disc & (de <= DE_MASK)
    fr["inverted_spot"] = float(np.median(T[sv])) if sv.sum() >= 10 else np.nan
    fr["spot_valid_px"] = int(sv.sum())
    fr["spot_rel_position"] = (job["ocr_spot"] - job["ocr_min"]) / (job["ocr_max"] - job["ocr_min"]) \
        if job["ocr_max"] > job["ocr_min"] else np.nan
    Tf = th.filtered_temperature(T, valid)
    # saturated top plateau of the palette (filtered map, so thin white MSX lines do not count)
    pal = _W["inv"].palette
    top_lo = job["ocr_min"] + (job["ocr_max"] - job["ocr_min"]) * (pal.level_pos[0] - pal.level_halfwidth[0])
    top = (Tf >= top_lo - 1e-4) & valid
    fr["n_top_frame"] = int(top.sum())
    seats = []
    for s in job["seats"]:
        m = th.seat_metrics(Tf, valid, phw, s.box, text, cross, fr["T_background"], top_mask=top)
        m.update(seat=s.seat, seat_expected=s.expected, box_x0=s.box[0], box_y0=s.box[1],
                 box_x1=s.box[2], box_y1=s.box[3])
        seats.append(m)
    return fr, seats


# ---------------------------------------------------------------------------------------------
def timing_tables(snaps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, corr = [], pd.Series(index=snaps.index, dtype=float)
    for (folder, sid), g in snaps.groupby(["session_folder", "stream_id"]):
        st, c = th.fit_stream_timing(g)
        corr.loc[g.index] = c
        avi = RAW / folder / f"{sid}{th.AVI_SUFFIX}"
        a = th.probe_avi(avi) if avi.exists() else dict(avi=None)
        start = th.prefix_epoch(sid)
        last_k = int(g.snapshot_index.max())
        rec = dict(session_folder=folder, **st.__dict__, **a,
                   nominal_last_snapshot_s=(last_k - 1) * th.NOMINAL_INTERVAL_S,
                   observed_last_snapshot_s=float(g.mtime_epoch.max() - start))
        rec["drift_last_snapshot_s"] = rec["observed_last_snapshot_s"] - rec["nominal_last_snapshot_s"]
        if a.get("avi_readable"):
            rec["avi_wall_elapsed_s"] = a["avi_mtime_epoch"] - start
            rec["avi_effective_fps"] = a["avi_frames"] / rec["avi_wall_elapsed_s"]
            rec["avi_nominal_minus_wall_s"] = a["avi_duration_nominal_s"] - rec["avi_wall_elapsed_s"]
            rec["avi_frames_per_snapshot"] = a["avi_frames"] / max(rec["observed_last_snapshot_s"], 1) * \
                (st.interval_s if np.isfinite(st.interval_s) else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows), corr


def epoch_to_local(x) -> pd.Series:
    return pd.to_datetime(pd.Series(x), unit="s", utc=True).dt.tz_convert(th.TZ).dt.tz_localize(None)


def run_frames(snaps: pd.DataFrame, workers: int, reuse: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    f1 = CACHE / "pass1.pkl"
    if reuse and f1.exists():
        res = pickle.loads(f1.read_bytes())
    else:
        with ProcessPoolExecutor(workers, initializer=_init_pass1) as ex:
            res = list(ex.map(_pass1, snaps.path, chunksize=16))
        f1.write_bytes(pickle.dumps(res))
    bars = np.stack([r.pop("bar") for r in res])
    struct = np.stack([r.pop("struct") for r in res]).astype(np.float32)
    return pd.DataFrame(res, index=snaps.index), bars, struct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="reuse cached per-frame passes")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    paths.ensure(OUT, FIG, CACHE)
    t0 = time.time()

    # ---------------- inventory & timing
    snaps = th.list_snapshots(RAW)
    streams, corrected = timing_tables(snaps)
    snaps["time_nominal"] = epoch_to_local(
        [th.prefix_epoch(s) + (k - 1) * th.NOMINAL_INTERVAL_S for s, k in zip(snaps.stream_id, snaps.snapshot_index)]).values
    snaps["time_local"] = epoch_to_local(corrected.values).values
    snaps["time_file_mtime"] = epoch_to_local(snaps.mtime_epoch.values).values
    snaps["time_method"] = snaps.stream_id.map(streams.set_index("stream_id").method)
    snaps["pose"] = [th.pose_of(f, s, k) for f, s, k in zip(snaps.session_folder, snaps.stream_id, snaps.snapshot_index)]
    print(f"[{time.time()-t0:5.0f}s] inventory: {len(snaps)} snapshots, {len(streams)} streams")

    # ---------------- pass 1: OCR, colour bar, structure
    ocr, bars, struct = run_frames(snaps, args.workers, args.reuse)
    fr = pd.concat([snaps, ocr], axis=1)
    palette = th.Palette.from_bars(bars[fr.ocr_max_ok & fr.ocr_min_ok])
    fr["bar_deltaE"] = [palette.frame_bar_de(b) for b in bars]
    print(f"[{time.time()-t0:5.0f}s] pass 1 done; palette K={palette.K}")

    # camera stability: shift vs per-pose reference (median structure image)
    for _, g in fr.groupby(["session_folder", "pose"]):
        ref = np.median(struct[g.index], axis=0)
        sh = np.array([th.pose_reference_shift(struct[i], ref) for i in g.index])
        fr.loc[g.index, "cam_dx"], fr.loc[g.index, "cam_dy"], fr.loc[g.index, "cam_resp"] = sh.T
    fr["cam_shift_px"] = np.hypot(fr.cam_dx, fr.cam_dy)
    fr["cam_moved"] = (fr.cam_shift_px > SHIFT_MAX_PX) & (fr.cam_resp >= SHIFT_MIN_RESP)

    # consistency checks of the OCR (flags only - never altered)
    fr["ocr_ok"] = fr.ocr_max_ok & fr.ocr_min_ok
    lo, hi = th.PLAUSIBLE_C
    fr["chk_order"] = fr.ocr_min < fr.ocr_max
    fr["chk_range"] = fr[["ocr_min", "ocr_max"]].apply(lambda c: c.between(lo, hi)).all(axis=1)
    fr["chk_spot_in_scale"] = (fr.ocr_spot >= fr.ocr_min - 0.05) & (fr.ocr_spot <= fr.ocr_max + 0.05)
    for c in ("ocr_max", "ocr_min"):
        med = fr[fr.ocr_ok].groupby("stream_id")[c].transform(lambda s: s.rolling(5, center=True, min_periods=1).median())
        fr[f"chk_{c}_jump"] = (fr[c] - med).abs().reindex(fr.index) > 1.5
    fr["frame_ok"] = fr.ocr_ok & fr.chk_order & fr.chk_range & (fr.bar_deltaE <= BAR_DE_MAX) & ~fr.cam_moved

    # ---------------- pass 2: inversion, spot, seats
    f2 = CACHE / "pass2.pkl"
    jobs = []
    for i, r in fr.iterrows():
        if r.frame_ok:
            jobs.append(dict(i=i, path=r.path, ocr_min=r.ocr_min, ocr_max=r.ocr_max, ocr_spot=r.ocr_spot, pose=r.pose,
                             seats=th.seats_for(r.session_folder, r.pose)))
    if args.reuse and f2.exists():
        res2 = pickle.loads(f2.read_bytes())
    else:
        th.Inverter(palette, CACHE)  # build LUT once before forking workers
        with ProcessPoolExecutor(args.workers, initializer=_init_pass2, initargs=(palette,)) as ex:
            res2 = list(ex.map(_pass2, jobs, chunksize=8))
        f2.write_bytes(pickle.dumps(res2))
    print(f"[{time.time()-t0:5.0f}s] pass 2 done ({len(jobs)} frames)")
    frame_extra = pd.DataFrame([r[0] for r in res2], index=[j["i"] for j in jobs])
    fr = fr.join(frame_extra)
    fr["palette_K"] = palette.K
    fr["resolution_C"] = (fr.ocr_max - fr.ocr_min) / (palette.K - 1)
    fr["top_plateau_C"] = (fr.ocr_max - fr.ocr_min) * 2 * palette.level_halfwidth[0]
    fr.loc[fr.spot_valid_px.fillna(0) < 10, "inverted_spot"] = np.nan

    # per-frame inversion check against the spot meter (only when the spot is on background, where the
    # inversion was shown to agree to ~0.15 C): residual vs the stream's rolling median residual
    fr["spot_residual"] = fr.inverted_spot - fr.ocr_spot.where(fr.ocr_spot_ok)
    roll = fr.groupby("stream_id").spot_residual.transform(lambda x: x.rolling(9, center=True, min_periods=3).median())
    fr["spot_check"] = np.where(fr.spot_residual.isna() | roll.isna() | (fr.spot_rel_position >= 0.3), "unchecked",
                                np.where((fr.spot_residual - roll).abs() <= SPOT_CHECK_C, "pass", "fail"))
    contrast = fr.T_scene_p90 - fr.T_scene_p10
    fr["low_contrast"] = contrast < LOW_CONTRAST_FRAC * contrast.groupby(fr.stream_id).transform(
        lambda x: x.rolling(9, center=True, min_periods=3).median())
    fr["inversion_ok"] = fr.frame_ok & (fr.spot_check != "fail") & ~fr.low_contrast.fillna(False)
    fr["other_person"] = ((fr.upper_warm_frac > UPPER_WARM_MAX) | (fr.scene_warm_frac > SCENE_WARM_MAX)
                          | (fr.corridor_warm_frac > CORRIDOR_WARM_MAX))
    # possible camera offset steps (e.g. after a shutter/NUC event): the spot (fixed background point) and
    # the scene median jump together by >= STEP_C between consecutive snapshots
    g_ = fr.groupby("stream_id")
    dspot = g_.ocr_spot.diff().where(fr.ocr_spot_ok & (fr.spot_rel_position < 0.3))
    dbg = g_.T_background.diff()
    fr["spot_step_C"], fr["bg_step_C"] = dspot, dbg
    fr["offset_step"] = (dspot.abs() >= STEP_C) & (dbg.abs() >= STEP_C) & (np.sign(dspot) == np.sign(dbg))

    seat_rows = []
    for j, r in zip(jobs, res2):
        for s in r[1]:
            seat_rows.append(dict(i=j["i"], **s))
    seats = pd.DataFrame(seat_rows)
    seats = seats.join(fr[["session_folder", "stream_id", "snapshot_index", "time_local", "pose", "file",
                           "inversion_ok", "other_person", "T_background"]], on="i")

    # ---------------- occupancy
    thr = th.otsu_threshold(seats.occ_score)
    seats["occupied_raw"] = (seats.occ_score >= thr).astype(float)
    seats.loc[seats.occ_score.isna(), "occupied_raw"] = np.nan
    seats = seats.sort_values(["session_folder", "seat", "stream_id", "snapshot_index"])
    seats["occupied"] = np.nan
    for _, g in seats.groupby(["session_folder", "seat", "stream_id"]):
        seats.loc[g.index, "occupied"] = th.smooth_occupancy(g.occupied_raw.to_numpy())
    # a warm head in an unused seat's box can only be another person (experimenter)
    ctrl_occ = seats[(seats.seat_expected == "empty_control") & (seats.occupied_raw == 1)].i.unique()
    fr.loc[ctrl_occ, "other_person"] = True
    seats.loc[seats.i.isin(ctrl_occ), "other_person"] = True
    seats["valid"] = ((seats.occupied == 1) & ~seats.roi_touches_text & seats.T_max.notna()
                      & seats.inversion_ok & ~seats.other_person)
    seats["invalid_reason"] = ""
    for cond, tag in ((seats.occupied != 1, "not_occupied"), (seats.roi_touches_text, "roi_touches_text"),
                      (~seats.inversion_ok, "inversion_check_failed"), (seats.other_person, "other_person_present")):
        seats.loc[cond, "invalid_reason"] += tag + ";"
    seats["invalid_reason"] = seats.invalid_reason.str.rstrip(";")
    for c in ("T_max", "T_p95", "T_top5mean", "T_median", "T_mean"):
        seats.loc[~seats.valid, c] = np.nan
        seats[f"{c}_minus_bg"] = seats[c] - seats.T_background
    # when the face holds the scene maximum, the camera's MAX label is the face maximum (radiometric,
    # 0.1 C display resolution) - better than the inverted value, which saturates in the top plateau
    seats = seats.join(fr[["n_top_frame", "ocr_max"]], on="i")
    seats["scene_max_on_face"] = seats.valid & (seats.n_top_frame > 0) & \
        (seats.n_top_in_roi >= SCENE_MAX_SHARE * seats.n_top_frame)
    seats["T_max_scale"] = seats.ocr_max.where(seats.scene_max_on_face)
    seats["T_max_scale_minus_bg"] = seats.T_max_scale - seats.T_background
    print(f"[{time.time()-t0:5.0f}s] occupancy threshold (Otsu) = {thr:.2f} C above background")

    # ---------------- vacancy events (per folder x used seat x stream)
    ev = []
    for (folder, seat, sid), g in seats[seats.seat_expected == "used"].groupby(["session_folder", "seat", "stream_id"]):
        g = g.sort_values("snapshot_index")
        # all snapshots of the stream (frames rejected in pass 1 count as unknown)
        allk = fr[(fr.session_folder == folder) & (fr.stream_id == sid)][["snapshot_index", "time_local"]]
        m = allk.merge(g[["snapshot_index", "occupied"]], on="snapshot_index", how="left")
        interval = float(streams.set_index("stream_id").loc[sid, "interval_s"])
        for e in th.vacancy_intervals(m.time_local, m.occupied.to_numpy()):
            ev.append(dict(session_folder=folder, seat=seat, stream_id=sid, sampling_interval_s=round(interval, 3)
                           if np.isfinite(interval) else np.nan, **e))
    events = pd.DataFrame(ev)

    # other-person episodes (experimenter visits etc.), attributed to the nearest seat by x position
    op = []
    for (folder, sid), g in fr.groupby(["session_folder", "stream_id"]):
        g = g.sort_values("snapshot_index")
        flag = g.other_person.fillna(False).to_numpy()
        i = 0
        while i < len(g):
            if not flag[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(g) and flag[j + 1]:
                j += 1
            seg = g.iloc[i:j + 1]
            cx = np.nanmedian(seg.upper_warm_cx) if seg.upper_warm_cx.notna().any() else np.nan
            seats_here = th.seats_for(folder, seg.pose.iloc[0])
            near = min(seats_here, key=lambda s_: abs((s_.box[0] + s_.box[2]) / 2 - cx)).seat if np.isfinite(cx) else ""
            op.append(dict(session_folder=folder, stream_id=sid, first_snapshot=int(seg.snapshot_index.iloc[0]),
                           last_snapshot=int(seg.snapshot_index.iloc[-1]), start_local=seg.time_local.iloc[0],
                           end_local=seg.time_local.iloc[-1], n_snapshots=len(seg),
                           standing_zone_x_px=round(float(cx), 0) if np.isfinite(cx) else np.nan,
                           nearest_seat=near, max_scene_warm_frac=round(float(seg.scene_warm_frac.max()), 3)))
            i = j + 1
    other_ep = pd.DataFrame(op)

    # ---------------- validation
    val = validate(fr, seats, events, streams, palette, thr)

    # ---------------- outputs
    write_outputs(fr, seats, events, streams, val)
    fmt_times(other_ep).to_csv(OUT / "other_person_episodes.csv", index=False)
    figures(fr, seats, events, palette, val)
    print(f"[{time.time()-t0:5.0f}s] done")


# ---------------------------------------------------------------------------------------------
def validate(fr, seats, events, streams, palette, thr) -> dict:
    refs = set(th.OCR_REFERENCES)
    man = []
    for sample, d in (("random", OCR_MANUAL_RANDOM), ("stratified", OCR_MANUAL_STRATIFIED)):
        for f, vals in d.items():
            r = fr.loc[fr.file == f].iloc[0]
            for name, v in zip(("spot", "max", "min"), vals):
                man.append(dict(sample=sample, file=f, field=name, visual=float(v), ocr=r[f"ocr_{name}"],
                                ocr_field_ok=bool(r[f"ocr_{name}_ok"]), is_template_reference=f in refs,
                                match=bool(np.isclose(float(v), r[f"ocr_{name}"]))))
    man = pd.DataFrame(man)
    man.to_csv(OUT / "ocr_manual_check.csv", index=False)
    ocr_val = {}
    for sample, g in man[~man.is_template_reference].groupby("sample"):
        per_frame = g.groupby("file").match.all()
        ocr_val[sample] = dict(n_frames=int(per_frame.size), n_fields=int(len(g)),
                               frame_exact_match_rate=float(per_frame.mean()),
                               field_exact_match_rate=float(g.match.mean()),
                               fields_flagged_by_reader=int((~g.ocr_field_ok).sum()))
    ocr_val["acceptance"] = {f: dict(n_accepted=int(fr[f"ocr_{f}_ok"].sum()), n_rejected=int((~fr[f"ocr_{f}_ok"]).sum()),
                                     rejected_files=fr.loc[~fr[f"ocr_{f}_ok"], "file"].tolist(),
                                     reasons=fr.loc[~fr[f"ocr_{f}_ok"], f"ocr_{f}_reason"].tolist())
                             for f in ("spot", "max", "min")}
    ok = fr.ocr_ok
    ocr_val["consistency"] = dict(
        min_lt_max=int((ok & ~fr.chk_order).sum()), outside_15_40=int((ok & ~fr.chk_range).sum()),
        spot_outside_scale=int((ok & fr.ocr_spot_ok & ~fr.chk_spot_in_scale).sum()),
        max_jump_gt_1p5_vs_rolling_median=int((ok & fr.chk_ocr_max_jump.fillna(False)).sum()),
        min_jump_gt_1p5_vs_rolling_median=int((ok & fr.chk_ocr_min_jump.fillna(False)).sum()),
        max_range=[float(fr.ocr_max[ok].min()), float(fr.ocr_max[ok].max())],
        min_range=[float(fr.ocr_min[ok].min()), float(fr.ocr_min[ok].max())],
        span_range=[float((fr.ocr_max - fr.ocr_min)[ok].min()), float((fr.ocr_max - fr.ocr_min)[ok].max())])

    # spot-meter agreement
    s = fr[fr.frame_ok & fr.ocr_spot_ok & fr.inverted_spot.notna()]
    ba = {"all": th.bland_altman(s.inverted_spot, s.ocr_spot)}
    for name, m in (("background_rel_lt_0.3", s.spot_rel_position < 0.3), ("warm_rel_ge_0.3", s.spot_rel_position >= 0.3)):
        b = th.bland_altman(s.inverted_spot[m], s.ocr_spot[m])
        d = (s.inverted_spot - s.ocr_spot)[m]
        b.update(median=float(d.median()), mad_sd=float(1.4826 * (d - d.median()).abs().median()))
        ba[name] = b
    ba["per_folder"] = {f: dict(n=int(len(g)), bias=float((g.inverted_spot - g.ocr_spot).mean()),
                                sd=float((g.inverted_spot - g.ocr_spot).std()))
                        for f, g in s.groupby("session_folder")}
    ba["note"] = ("difference = inverted - camera spot reading; spot disc r<=3.5 px inside the crosshair ring, "
                  "pixels with DeltaE>12 excluded; camera reading displayed to 0.1 C (quantisation SD 0.03 C)")

    # occupancy validation
    occ = {"threshold_C_above_background": thr, "method": "Otsu on 99th-percentile filtered temperature in the head "
           "search box minus frame median temperature; isolated single-snapshot flips smoothed"}
    ctrl = seats[seats.seat_expected == "empty_control"]
    occ["empty_control_seats"] = {f"{f}/{s_}": dict(n=int(g.occupied_raw.notna().sum()),
                                                     specificity_raw=float((g.occupied_raw == 0).mean()),
                                                     specificity=float((g.occupied == 0).mean()))
                                  for (f, s_), g in ctrl.groupby(["session_folder", "seat"])}
    used = seats[seats.seat_expected == "used"]
    occ["score_distribution"] = dict(
        used_seats_pct=[float(x) for x in np.nanpercentile(used.occ_score, [1, 5, 50, 95])],
        control_seats_pct=[float(x) for x in np.nanpercentile(ctrl.occ_score, [50, 95, 99, 100])])
    # census validation (all snapshot x seat rows with a classification)
    def visual_label(r):
        if r.seat_expected == "empty_control":
            return 0
        for lo_, hi_ in VISUAL_VACANT.get((r.session_folder, r.stream_id, r.seat), []):
            if lo_ <= r.snapshot_index <= hi_:
                return 0
        return 1
    k = seats.dropna(subset=["occupied"]).copy()
    k["visual"] = k.apply(visual_label, axis=1)
    k["other_person_visual"] = [in_episode(sid, i) for sid, i in zip(k.stream_id, k.snapshot_index)]

    def confusion(df, col):
        tp = int(((df.visual == 1) & (df[col] == 1)).sum()); tn = int(((df.visual == 0) & (df[col] == 0)).sum())
        fp = int(((df.visual == 0) & (df[col] == 1)).sum()); fn = int(((df.visual == 1) & (df[col] == 0)).sum())
        return dict(n=len(df), tp=tp, tn=tn, fp=fp, fn=fn, accuracy=(tp + tn) / max(len(df), 1),
                    sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1))
    occ["visual_census"] = {
        "all_rows": confusion(k, "occupied"),
        "all_rows_raw": confusion(k, "occupied_raw"),
        "excluding_other_person_snapshots": confusion(k[~k.other_person_visual], "occupied"),
        "note": ("labels from a complete visual review of all snapshots; 'visual=0' on unused seats means no "
                 "participant - false positives there are the experimenter standing in the seat region")}
    fpk = k[(k.visual == 0) & (k.occupied == 1)]
    occ["visual_census"]["false_positive_rows"] = (fpk.session_folder + "/" + fpk.seat + "#" +
                                                  fpk.snapshot_index.astype(str)).tolist()
    fnk = k[(k.visual == 1) & (k.occupied == 0)]
    occ["visual_census"]["false_negative_rows"] = (fnk.session_folder + "/" + fnk.seat + "#" +
                                                  fnk.snapshot_index.astype(str)).tolist()
    k[["session_folder", "stream_id", "snapshot_index", "seat", "seat_expected", "visual", "other_person_visual",
       "occupied", "occupied_raw", "occ_score"]].to_csv(OUT / "occupancy_visual_census.csv", index=False)

    # other-person flag vs visual episodes
    f2 = fr[fr.inversion_ok.notna() & fr.upper_warm_frac.notna()].copy()
    f2["visual"] = [in_episode(sid, i) for sid, i in zip(f2.stream_id, f2.snapshot_index)]
    f2["visual_pm1"] = [in_episode(sid, i, pad=1) for sid, i in zip(f2.stream_id, f2.snapshot_index)]
    eps = [(sid, lo_, hi_) for sid, L in VISUAL_OTHER_PERSON.items() for lo_, hi_ in L]
    hit = []
    for sid, lo_, hi_ in eps:
        g = f2[(f2.stream_id == sid) & f2.snapshot_index.between(lo_, hi_)]
        hit.append(bool(g.other_person.any()))
    occ["other_person_flag"] = dict(
        n_visual_episodes=len(eps), episodes_detected=int(sum(hit)),
        episode_sensitivity=float(np.mean(hit)) if hit else None,
        n_flagged_snapshots=int(f2.other_person.sum()),
        flagged_within_visual_episode_pm1=float(f2[f2.other_person].visual_pm1.mean()) if f2.other_person.any() else None,
        snapshot_sensitivity=float(f2[f2.visual].other_person.mean()) if f2.visual.any() else None,
        missed_episodes=[f"{sid}#{lo_}-{hi_}" for (sid, lo_, hi_), h in zip(eps, hit) if not h],
        flagged_outside_episodes=(f2[f2.other_person & ~f2.visual_pm1].file).tolist())

    # timing
    tim = streams[["session_folder", "stream_id", "n", "method", "interval_s", "offset_s", "resid_sd_s",
                   "resid_maxabs_s", "first_snapshot_after_start_s", "drift_last_snapshot_s"]].copy()
    tim_summary = dict(
        note=("PNG mtimes (2-s resolution) are the timing evidence; time_local = linear fit of mtime on index "
              "per regular stream; time_nominal = prefix + (index-1)*10 s as specified for comparison."),
        regular_streams_interval_s=[float(tim[tim.method == 'linear_fit_mtime'].interval_s.min()),
                                    float(tim[tim.method == 'linear_fit_mtime'].interval_s.max())],
        streams=tim.to_dict(orient="records"))
    if "avi_effective_fps" in streams:
        tim_summary["avi"] = streams[["stream_id", "avi_readable", "avi_frames", "avi_fps_nominal",
                                      "avi_duration_nominal_s", "avi_wall_elapsed_s", "avi_effective_fps",
                                      "avi_nominal_minus_wall_s"]].to_dict(orient="records")

    cam = {p: dict(n=int(len(g)), median_shift_px=float(g.cam_shift_px.median()),
                   p95_shift_px=float(g.cam_shift_px.quantile(.95)), n_flagged_moved=int(g.cam_moved.sum()))
           for p, g in fr.groupby(["session_folder", "pose"])}
    cam = {f"{k[0]}/{k[1]}": v for k, v in cam.items()}

    usable = seats[seats.seat_expected == "used"].groupby(["session_folder", "seat"]).agg(
        n_snapshots=("i", "size"), n_occupied=("occupied", lambda x: int((x == 1).sum())),
        n_valid=("valid", "sum")).reset_index()
    fold_total = fr.groupby("session_folder").size()
    usable["n_snapshots_total"] = usable.session_folder.map(fold_total)

    # facial metric diagnostics
    v = seats[seats.valid]
    fm = dict(n_valid_rows=int(len(v)),
              frac_Tmax_in_top_plateau=float((v.n_top_in_roi > 0).mean()),
              frac_scene_max_on_face=float(v.scene_max_on_face.mean()),
              Tmax_scale_minus_Tmax_inverted=dict(mean=float((v.T_max_scale - v.T_max).mean()),
                                                  sd=float((v.T_max_scale - v.T_max).std())),
              n_frames_offset_step=int(fr.offset_step.fillna(False).sum()),
              n_frames_spot_check_fail=int((fr.spot_check == "fail").sum()),
              n_frames_other_person=int(fr.other_person.fillna(False).sum()))
    sd_rows = []
    for _, g in v.sort_values("snapshot_index").groupby(["session_folder", "seat", "stream_id"]):
        if len(g) < 50:
            continue
        consec = g.snapshot_index.diff() == 1
        sd_rows.append(dict(raw=g.T_max.diff()[consec].std(), ref=g.T_max_minus_bg.diff()[consec].std(),
                            r=g.T_max.diff()[consec].corr(g.T_background.diff()[consec])))
    sd_rows = pd.DataFrame(sd_rows)
    fm["successive_diff_sd_Tmax"] = float(sd_rows.raw.mean())
    fm["successive_diff_sd_Tmax_minus_bg"] = float(sd_rows.ref.mean())
    fm["corr_successive_diff_Tmax_vs_background"] = float(sd_rows.r.mean())
    cam_spot = fr[fr.spot_check == "pass"].sort_values("snapshot_index")
    dist = {}
    for fo, g in v.groupby("session_folder"):
        w = g.pivot_table(index=["stream_id", "snapshot_index"], columns="seat", values=["T_max", "T_median"])
        if ("T_max", "L") in w and ("T_max", "R") in w:
            dist[fo] = {f"{m}_mean_L_minus_R": float((w[m]["L"] - w[m]["R"]).mean()) for m in ("T_max", "T_median")}
            dist[fo].update({f"{m}_corr_successive_diff_L_R": float(w[m]["L"].diff().corr(w[m]["R"].diff()))
                             for m in ("T_max", "T_median")})
    fm["seat_distinctness"] = dist
    fm["sat_frac_median"] = float(v.sat_frac.median())
    fm["camera_spot_successive_diff_sd"] = float(cam_spot.groupby("stream_id").ocr_spot.diff().std())

    return dict(
        generated=pd.Timestamp.now().isoformat(timespec="seconds"),
        facial_metrics=fm,
        n_snapshots=int(len(fr)), n_frames_ok=int(fr.frame_ok.sum()),
        frame_rejections=dict(ocr=int((~fr.ocr_ok).sum()), bar_deltaE=int((fr.bar_deltaE > BAR_DE_MAX).sum()),
                              camera_moved=int(fr.cam_moved.sum())),
        ocr=ocr_val, spot_bland_altman=ba, occupancy=occ, timing=tim_summary, camera_stability=cam,
        palette=dict(K_levels=palette.K, bar_rows=int(len(palette.rows_bgr)),
                     top_plateau_fraction_of_span=float(2 * palette.level_halfwidth[0]),
                     median_level_step_fraction_of_span=float(np.median(-np.diff(palette.level_pos))),
                     de_mask_threshold=DE_MASK,
                     frame_bar_deltaE_pct=[float(x) for x in np.percentile(fr.bar_deltaE, [50, 95, 99, 100])],
                     offpalette_frac_pct=[float(x) for x in np.nanpercentile(fr.offpalette_frac, [50, 95, 99])]),
        usable_frames=usable.to_dict(orient="records"),
        n_vacancy_events=int(len(events)),
    )


def write_outputs(fr, seats, events, streams, val) -> None:
    def note(r):
        n = []
        for f in ("spot", "max", "min"):
            if not r[f"ocr_{f}_ok"]:
                n.append(f"ocr_{f}_rejected({r[f'ocr_{f}_reason']})")
        if r.bar_deltaE > BAR_DE_MAX:
            n.append("colour_bar_mismatch")
        if r.cam_moved:
            n.append(f"camera_shift_{r.cam_shift_px:.1f}px")
        if r.ocr_ok and not r.chk_spot_in_scale:
            n.append("spot_outside_scale")
        if r.scale_jump:
            n.append("scale_jump")
        if r.time_method != "linear_fit_mtime":
            n.append("time_from_raw_mtime")
        if r.pose == "C0":
            n.append("pre_session_stream")
        if r.spot_check == "fail":
            n.append("spot_check_failed(scale/image mismatch)")
        if r.low_contrast is True:
            n.append("low_contrast_frame")
        if r.other_person is True:
            n.append("other_person_present")
        if r.offset_step is True:
            n.append(f"offset_step_spot{r.spot_step_C:+.1f}C_bg{r.bg_step_C:+.1f}C(possible_camera_offset_step)")
        return ";".join(n)
    fr = fr.copy()
    fr["scale_jump"] = fr.chk_ocr_max_jump.fillna(False).astype(bool) | fr.chk_ocr_min_jump.fillna(False).astype(bool)
    fr["notes"] = fr.apply(note, axis=1)
    for c in ("ocr_max", "ocr_min", "ocr_spot"):
        fr.loc[~fr[f"{c}_ok"], c] = np.nan
    cols = ["session_folder", "stream_id", "snapshot_index", "file", "time_local", "time_nominal", "time_file_mtime",
            "time_method", "pose", "ocr_max", "ocr_min", "ocr_spot", "ocr_ok", "ocr_spot_ok", "frame_ok",
            "inversion_ok", "inverted_spot", "spot_residual", "spot_check", "spot_rel_position", "palette_K",
            "resolution_C", "top_plateau_C", "T_background", "T_scene_p10", "T_scene_p90", "offpalette_frac",
            "bar_deltaE", "cam_dx", "cam_dy", "cam_resp", "other_person", "upper_warm_frac", "scene_warm_frac",
            "corridor_warm_frac", "low_contrast", "spot_step_C", "bg_step_C", "offset_step", "notes"]
    out = fr[cols].copy()
    out["frame_ok"] = out.frame_ok.astype(bool)
    for c in ("inverted_spot", "spot_residual", "spot_rel_position", "resolution_C", "top_plateau_C", "T_background",
              "T_scene_p10", "T_scene_p90", "bar_deltaE", "cam_dx", "cam_dy", "cam_resp", "spot_step_C", "bg_step_C"):
        out[c] = out[c].round(3)
    for c in ("offpalette_frac", "upper_warm_frac", "scene_warm_frac", "corridor_warm_frac"):
        out[c] = out[c].round(4)
    fmt_times(out).to_csv(OUT / "thermal_frames.csv", index=False)

    s = seats.rename(columns={"occ_score": "occupancy_score"})
    s["roi_bbox"] = s.apply(lambda r: "" if pd.isna(r.roi_x0) else
                            f"{int(r.roi_x0)},{int(r.roi_y0)},{int(r.roi_x1)},{int(r.roi_y1)}", axis=1)
    s["search_box"] = s.apply(lambda r: f"{r.box_x0},{r.box_y0},{r.box_x1},{r.box_y1}", axis=1)
    cols = ["session_folder", "stream_id", "snapshot_index", "time_local", "pose", "seat", "seat_expected",
            "occupied", "occupied_raw", "occupancy_score", "valid", "invalid_reason", "T_max", "T_p95", "T_top5mean",
            "T_max_scale", "scene_max_on_face", "T_background", "T_max_minus_bg", "T_max_scale_minus_bg",
            "T_p95_minus_bg", "T_top5mean_minus_bg", "T_median", "T_mean", "T_median_minus_bg", "sat_frac",
            "roi_area_px",
            "roi_bbox", "search_box", "tmax_plateau_hw", "roi_touches_text", "roi_touches_crosshair",
            "roi_touches_box_edge", "valid_frac"]
    s = s[cols].sort_values(["session_folder", "stream_id", "snapshot_index", "seat"])
    for c in ("occupancy_score", "T_max", "T_p95", "T_top5mean", "T_background", "T_max_minus_bg",
              "T_max_scale_minus_bg", "T_p95_minus_bg", "T_median", "T_mean", "T_median_minus_bg", "sat_frac",
              "T_top5mean_minus_bg", "tmax_plateau_hw", "valid_frac"):
        s[c] = s[c].round(3)
    fmt_times(s).to_csv(OUT / "thermal_seats.csv", index=False)

    if len(events):
        fmt_times(events).to_csv(OUT / "seat_vacancy_events.csv", index=False)
    else:
        pd.DataFrame(columns=["session_folder", "seat", "stream_id", "start_local", "end_local", "duration_s"]).to_csv(
            OUT / "seat_vacancy_events.csv", index=False)
    streams.to_csv(OUT / "thermal_streams.csv", index=False)
    (OUT / "thermal_validation.json").write_text(json.dumps(val, indent=2, default=_json_default))


def fmt_times(df: pd.DataFrame) -> pd.DataFrame:
    """Datetime columns -> 'YYYY-MM-DD HH:MM:SS.mmm' local-time strings (empty for NaT)."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3].fillna("")
    return df


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if o is pd.NaT:
        return None
    return str(o)


# ---------------------------------------------------------------------------------------------
# Figures (static QC; palette slots from the validated reference categorical palette)
# ---------------------------------------------------------------------------------------------
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(plt):
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                         "legend.frameon": False})


def figures(fr, seats, events, palette, val) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _style(plt)

    # 1. Bland-Altman of inverted vs camera spot
    s = fr[fr.frame_ok & fr.ocr_spot_ok & fr.inverted_spot.notna()]
    d = s.inverted_spot - s.ocr_spot
    m = (s.inverted_spot + s.ocr_spot) / 2
    warm = s.spot_rel_position >= 0.3
    ba = val["spot_bland_altman"]["all"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].scatter(m[~warm], d[~warm], s=8, color=C1, alpha=.35, lw=0, label=f"spot on background (n={int((~warm).sum())})")
    ax[0].scatter(m[warm], d[warm], s=14, color=C2, alpha=.8, lw=0, label=f"spot on warm target, rel. scale >= 0.3 (n={int(warm.sum())})")
    for y, ls in ((ba["bias"], "-"), (ba["loa_low"], "--"), (ba["loa_high"], "--")):
        ax[0].axhline(y, color=INK2, lw=1, ls=ls)
    ax[0].text(ax[0].get_xlim()[1], ba["bias"], f" bias {ba['bias']:+.2f}", va="bottom", ha="right", color=INK2)
    ax[0].text(ax[0].get_xlim()[1], ba["loa_high"], f" +1.96 SD {ba['loa_high']:+.2f}", va="bottom", ha="right", color=INK2)
    ax[0].text(ax[0].get_xlim()[1], ba["loa_low"], f" -1.96 SD {ba['loa_low']:+.2f}", va="top", ha="right", color=INK2)
    ax[0].set_xlabel("mean of inverted and camera spot (°C)")
    ax[0].set_ylabel("inverted − camera spot (°C)")
    ax[0].set_title("Bland–Altman: palette inversion vs camera spot meter", loc="left")
    ax[0].legend(loc="lower left")
    ax[1].scatter(s.spot_rel_position, d, s=8, color=C1, alpha=.35, lw=0)
    ax[1].axhline(0, color=INK2, lw=1)
    ax[1].set_xlabel("spot position within displayed scale (0 = MIN, 1 = MAX)")
    ax[1].set_ylabel("inverted − camera spot (°C)")
    ax[1].set_title("Difference vs position in the auto-ranged scale", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "spot_bland_altman.png", dpi=150)
    plt.close(fig)

    # 2. per-folder seat time series
    seat_col = {"L": C1, "R": C2, "C": C3}
    for folder, g in seats.groupby("session_folder"):
        used = [x for x in ["L", "R", "C"] if x in set(g[g.seat_expected == "used"].seat)]
        fig, axes = plt.subplots(len(used) + 2, 1, figsize=(11, 2.4 * (len(used) + 2)), sharex=True)
        f0 = fr[fr.session_folder == folder]
        for sid, ff in f0.groupby("stream_id"):
            first = sid == f0.stream_id.iloc[0]
            axes[0].plot(ff.time_local, ff.ocr_max, color=C2, lw=1, label="scale MAX" if first else None)
            axes[0].plot(ff.time_local, ff.ocr_spot.where(ff.ocr_spot_ok), color=C3, lw=1,
                         label="spot (camera, fixed point)" if first else None)
            axes[0].plot(ff.time_local, ff.T_background, color=INK2, lw=1, label="scene median (inverted)" if first else None)
            axes[0].plot(ff.time_local, ff.ocr_min, color=C1, lw=1, label="scale MIN" if first else None)
        axes[0].set_ylabel("°C")
        axes[0].set_title(f"{folder} — on-screen scale (OCR) and scene median", loc="left", pad=18)
        axes[0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, borderaxespad=0.1)
        for ax, seat in zip(axes[1:-1], used):
            h = g[g.seat == seat].sort_values("time_local")
            for sid, hh in h.groupby("stream_id"):
                first = sid == h.stream_id.iloc[0]
                ax.plot(hh.time_local, hh.T_max, color=INK2, lw=0.8,
                        label="T_max (inverted; saturates in palette top plateau)" if first else None)
                ax.plot(hh.time_local, hh.T_median, color=seat_col[seat], lw=1.2,
                        label="T_median (head blob, recommended)" if first else None)
                ax.scatter(hh.time_local, hh.T_max_scale, s=6, color=INK, lw=0,
                           label="T_max_scale (camera MAX, face holds scene max)" if first else None)
            ev = events[(events.session_folder == folder) & (events.seat == seat)] if len(events) else []
            for _, e in (ev.iterrows() if len(ev) else []):
                ax.axvspan(e.start_local, e.end_local, color=INK2, alpha=.15, lw=0)
            inval = h[~h.valid & (h.occupied == 1)]
            if len(inval):
                y = np.nanmin(h.T_max) - 0.3 if h.T_max.notna().any() else 30
                ax.scatter(inval.time_local, np.full(len(inval), y), marker="|", color=C2, s=25,
                           label="excluded snapshot (other person / check failed)")
            ax.set_ylabel("°C")
            ax.set_title(f"seat {seat}: head-ROI temperature (grey band = vacancy ≥ 30 s)", loc="left", pad=30)
            ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, borderaxespad=0.1)
        ax = axes[-1]
        for seat in used:
            h = g[g.seat == seat].sort_values("time_local")
            for sid, hh in h.groupby("stream_id"):
                ax.plot(hh.time_local, hh.T_median_minus_bg, color=seat_col[seat], lw=1,
                        label=f"seat {seat}" if sid == h.stream_id.iloc[0] else None)
        ax.set_ylabel("°C above scene median")
        ax.set_title("background-referenced T_median − T_background (cancels common-mode camera offsets)", loc="left", pad=18)
        ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, borderaxespad=0.1)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes[-1].set_xlabel("local time (Europe/Copenhagen)")
        fig.tight_layout()
        fig.savefig(FIG / f"seat_timeseries_{folder}.png", dpi=130)
        plt.close(fig)

    # 3. OCR examples: label crops only
    rng = np.random.default_rng(SEED)
    ex = fr[fr.ocr_ok].sample(8, random_state=SEED).index.tolist()
    ex += fr.index[~(fr.ocr_spot_ok & fr.ocr_max_ok & fr.ocr_min_ok)].tolist()[:4]
    fig, axes = plt.subplots(len(ex), 3, figsize=(6.5, 0.75 * len(ex) + 0.4))
    for row, i in zip(axes, ex):
        img = cv2.imread(fr.path[i])
        crops = th.label_crops(img)
        for a, k in zip(row, ("spot", "max", "min")):
            a.imshow(cv2.cvtColor(crops[k], cv2.COLOR_BGR2RGB), interpolation="nearest")
            okk = fr.loc[i, f"ocr_{k}_ok"]
            a.set_title(f"{k}: {fr.loc[i, f'ocr_{k}']:.1f}" + ("" if okk else " (rejected)"), fontsize=7,
                        color=INK if okk else C2, loc="left")
            a.axis("off")
        row[0].text(-0.05, 0.5, fr.loc[i, "file"].split("/")[0] + "\n#" + str(fr.loc[i, "snapshot_index"]),
                    transform=row[0].transAxes, ha="right", va="center", fontsize=6, color=INK2)
    fig.suptitle("OCR examples (label crops only; last rows = rejected fields)", fontsize=9, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "ocr_examples.png", dpi=150)
    plt.close(fig)
    del rng

    # 4. palette
    fig, ax = plt.subplots(figsize=(8, 2.6))
    rows = palette.rows_bgr[::-1, ::-1] / 255.0
    ax.imshow(rows[None, :, :], aspect="auto", extent=(0, 1, 0, 1))
    ax.vlines(palette.level_pos, 1.0, 1.12, color=INK2, lw=0.4, clip_on=False)
    ax.set_yticks([])
    ax.set_xlabel("palette position (0 = MIN, 1 = MAX)")
    ax.set_title(f"Canonical iron palette from the colour bar: {len(palette.rows_bgr)} rows → K = {palette.K} distinct "
                 f"levels (ticks); saturated top plateau = {2*palette.level_halfwidth[0]*100:.1f}% of span", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "palette.png", dpi=150)
    plt.close(fig)

    # 5. timing: observed mtime vs nominal
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for k, (sid, g) in enumerate(fr.groupby("stream_id")):
        if len(g) < 20:
            continue
        lag = (g.time_file_mtime - g.time_nominal).dt.total_seconds()
        ax.plot(g.snapshot_index, lag, lw=1, color=[C1, C2, C3][k % 3], alpha=.8)
    ax.set_xlabel("snapshot index")
    ax.set_ylabel("file mtime − nominal time (s)")
    ax.set_title("Snapshot timing drift vs nominal prefix + (k−1)·10 s (one line per stream; P01A stream 1 used 5 s)",
                 loc="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "timing_drift.png", dpi=150)
    plt.close(fig)

    # 6. occupancy score distribution
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bins = np.linspace(np.nanmin(seats.occ_score), np.nanmax(seats.occ_score), 80)
    ax.hist(seats[seats.seat_expected == "used"].occ_score.dropna(), bins=bins, color=C1, alpha=.7, label="used seats")
    ax.hist(seats[seats.seat_expected == "empty_control"].occ_score.dropna(), bins=bins, color=C2, alpha=.7,
            label="empty-control seats (P07A/B left, P10A/B left)")
    ax.axvline(val["occupancy"]["threshold_C_above_background"], color=INK, lw=1)
    ax.set_xlabel("occupancy score: p99 head-box temperature − frame median (°C)")
    ax.set_ylabel("snapshot × seat")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Occupancy score and Otsu threshold", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "occupancy_score.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
