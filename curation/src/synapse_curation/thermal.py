"""FLIR AX8 snapshot curation: OCR of the on-screen scale, palette inversion
to an approximate temperature map, seat ROIs/occupancy and timing.

Source material
---------------
The AX8 was read over RTSP (``rtsp://.../avc``, H.264, 640x480, ~8.9 fps) by
``code/flir/script_4.py``; every decoded frame went into an XVID AVI and one
frame every ``SNAPSHOT_INTERVAL_SECONDS`` (10 s; 5 s in the first P01A-P02A
stream) was written losslessly as PNG with ``cv2.imwrite``.  The frames are
*rendered* video (iron palette, MSX edge overlay, auto-ranged scale, text
overlays) - no radiometric data.  The only absolute information on screen is
the scale (MAX top-right, MIN bottom-right) and the spot-meter reading
(top-left, measured at the centre crosshair).

Method (see ``config/thermal_method_README.md``)
--------------------------------------------------------------
1. OCR: the three labels always have the layout ``DD.D`` at fixed pixel
   positions.  Each digit slot is matched (normalised cross-correlation, +-2 px
   search) against glyph templates cut from visually verified reference frames
   (``OCR_REFERENCES``).  A field is accepted only if every slot is confident,
   unambiguous, the decimal point is present and no ink lies outside the slots.
2. Palette: the colour bar interior (x 627-632, rows 30-447) is the palette
   from MAX (top) to MIN (bottom).  A canonical palette is the per-row median
   over all frames; consecutive rows that are identical (palette plateaus,
   e.g. the saturated white top) are merged into one level located at the
   mean row.  Every RGB triplet is mapped once (256^3 LUT) to the nearest level
   in CIELAB; distance = Delta E76.
3. Temperature: T = MIN + (MAX - MIN) * level_position, position 0 at the bottom
   row and 1 at the top row of the bar (linear scale assumed and tested
   against the spot meter).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

TZ = "Europe/Copenhagen"
FRAME_W, FRAME_H = 640, 480
SENSOR_W, SENSOR_H = 80, 60          # AX8 microbolometer; display is 8x upscaled
SNAP_RE = re.compile(r"(?P<prefix>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_snapshot_(?P<idx>\d{4})\.png$")
AVI_SUFFIX = "_flir_ax8_recording.avi"
NOMINAL_INTERVAL_S = 10.0            # script_4.py SNAPSHOT_INTERVAL_SECONDS

# --------------------------------------------------------------------------------------
# Frame inventory and timing
# --------------------------------------------------------------------------------------


def list_snapshots(thermal_root: Path) -> pd.DataFrame:
    """One row per snapshot PNG: folder, stream (= filename prefix), index, mtime."""
    rows = []
    for folder in sorted(p for p in thermal_root.iterdir() if p.is_dir()):
        for f in sorted(folder.iterdir()):
            m = SNAP_RE.match(f.name)
            if not m:
                continue
            rows.append(dict(session_folder=folder.name, stream_id=m["prefix"],
                             snapshot_index=int(m["idx"]), path=str(f),
                             file=f"{folder.name}/{f.name}", mtime_epoch=f.stat().st_mtime))
    df = pd.DataFrame(rows).sort_values(["session_folder", "stream_id", "snapshot_index"])
    return df.reset_index(drop=True)


def prefix_epoch(prefix: str) -> float:
    t = _dt.datetime.strptime(prefix, "%Y-%m-%d_%H-%M-%S")
    return pd.Timestamp(t, tz=TZ).timestamp()


def probe_avi(path: Path) -> dict:
    """Container metadata of a recording (ffprobe); frame count by packet count."""
    out = dict(avi=path.name, avi_bytes=path.stat().st_size, avi_mtime_epoch=path.stat().st_mtime,
               avi_readable=False, avi_frames=np.nan, avi_fps_nominal=np.nan, avi_duration_nominal_s=np.nan)
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
                            "-show_entries", "stream=nb_read_packets,r_frame_rate", "-of", "json", str(path)],
                           capture_output=True, text=True, timeout=600)
        s = json.loads(r.stdout or "{}").get("streams", [])
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        s = []
    if s and s[0].get("nb_read_packets"):
        num, den = s[0]["r_frame_rate"].split("/")
        fps = float(num) / float(den)
        n = int(s[0]["nb_read_packets"])
        out.update(avi_readable=True, avi_frames=n, avi_fps_nominal=fps, avi_duration_nominal_s=n / fps)
    return out


@dataclass
class StreamTiming:
    stream_id: str
    n: int
    method: str                 # 'linear_fit_mtime' | 'mtime'
    interval_s: float           # fitted cadence (s / snapshot)
    offset_s: float             # fitted time of index 0 relative to stream start
    resid_sd_s: float
    resid_maxabs_s: float
    first_snapshot_after_start_s: float


def fit_stream_timing(g: pd.DataFrame, min_n: int = 20, max_resid_s: float = 2.5) -> tuple[StreamTiming, np.ndarray]:
    """Per-stream time base from file modification times.

    The PNG mtimes survive in the archive at 2-s resolution (FAT-style even
    seconds).  For regular streams a straight line mtime ~ a + b*index is
    fitted (quantisation averages out); streams that are short or irregular
    (reconnections) fall back to the raw mtime.  Returns timing summary and the
    corrected epoch per row of ``g`` (same order).
    """
    start = prefix_epoch(g["stream_id"].iloc[0])
    k = g["snapshot_index"].to_numpy(float)
    t = g["mtime_epoch"].to_numpy(float) - start
    method, b, a = "mtime", np.nan, np.nan
    resid = np.zeros_like(t)
    if len(k) >= min_n:
        b, a = np.polyfit(k, t, 1)
        resid = t - (a + b * k)
        if np.abs(resid).max() <= max_resid_s:
            method = "linear_fit_mtime"
    corrected = start + ((a + b * k) if method == "linear_fit_mtime" else t)
    st = StreamTiming(g["stream_id"].iloc[0], len(k), method, float(b), float(a),
                      float(resid.std()), float(np.abs(resid).max()), float(t[0]))
    return st, corrected


# --------------------------------------------------------------------------------------
# OCR of the scale labels and spot meter
# --------------------------------------------------------------------------------------

# Field geometry in full-frame pixel coordinates (layout "DD.D", verified to be constant over
# all 3,522 snapshots except a handful of corrupted frames).
#   rows: text band; slots: x-windows searched for each digit; dot: x-window of the decimal point
#   box: x-extent of the dark label box that must contain no ink outside slots/dot.
FIELDS = {
    "spot": dict(font="big", rows=(4, 27), slots=[(7, 22), (21, 37), (42, 58)], dot=(36, 43), box=(5, 60)),
    "max": dict(font="small", rows=(5, 24), slots=[(597, 607), (607, 618), (622, 634)], dot=(618, 622), box=(592, 636)),
    "min": dict(font="small", rows=(453, 474), slots=[(597, 607), (607, 618), (622, 634)], dot=(618, 622), box=(592, 636)),
}
INK = 128                  # gray level above which a pixel counts as text ink
OCR_MIN_SCORE = 0.80       # NCC needed for a digit
OCR_MIN_MARGIN = 0.04      # best minus best other character
PLAUSIBLE_C = (15.0, 40.0)

# Reference frames (visually verified: label values transcribed by eye from 4x enlarged crops).
# Each entry: file -> (spot, max, min) as displayed.  Templates for all ten digits of both fonts
# are cut from these frames; several variants per digit capture sub-pixel rendering differences.
OCR_REFERENCES = {
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0001.png": ("25.0", "34.9", "23.9"),
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0002.png": ("24.9", "34.9", "23.7"),
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0003.png": ("24.7", "34.7", "23.7"),
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0004.png": ("24.8", "34.7", "23.8"),
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0006.png": ("24.8", "34.6", "23.7"),
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0011.png": ("25.0", "35.0", "24.1"),
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0017.png": ("30.8", "35.1", "23.8"),
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0018.png": ("29.7", "35.0", "23.7"),
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0137.png": ("26.2", "35.7", "24.9"),
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0241.png": ("27.1", "34.9", "24.1"),
    "P01A-P02A/2025-06-24_12-46-10_snapshot_0264.png": ("31.6", "35.1", "24.3"),
    "P05A-P06A/2025-06-25_10-35-37_snapshot_0074.png": ("28.4", "35.0", "24.0"),
    "P07A/2025-06-25_12-34-40_snapshot_0162.png": ("24.6", "32.6", "23.7"),
    "P08A-P09A/2025-06-25_15-05-40_snapshot_0341.png": ("25.3", "31.9", "24.5"),
}


def _gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _tight(patch: np.ndarray, thr: float = 60.0) -> np.ndarray | None:
    ys, xs = np.nonzero(patch > thr)
    if len(ys) < 4:
        return None
    return patch[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()


class DigitReader:
    """Deterministic template-matching reader for the three on-screen numbers."""

    def __init__(self, templates: dict[str, list[tuple[str, np.ndarray]]]):
        self.templates = templates     # font -> [(char, glyph)]

    @classmethod
    def from_references(cls, thermal_root: Path, refs: dict = OCR_REFERENCES) -> "DigitReader":
        templates: dict[str, list[tuple[str, np.ndarray]]] = {"big": [], "small": []}
        for rel, values in refs.items():
            g = _gray(cv2.imread(str(thermal_root / rel)))
            for fname, text in zip(("spot", "max", "min"), values):
                spec = FIELDS[fname]
                chars = text.replace(".", "")
                r0, r1 = spec["rows"]
                for (c0, c1), ch in zip(spec["slots"], chars):
                    t = _tight(g[r0:r1, c0:c1])
                    if t is None:
                        raise ValueError(f"empty reference glyph {rel} {fname}")
                    # keep distinct variants only
                    bank = templates[spec["font"]]
                    dup = any(c == ch and v.shape == t.shape and np.abs(v - t).mean() < 6 for c, v in bank)
                    if not dup:
                        bank.append((ch, t))
        missing = {f: sorted(set("0123456789") - {c for c, _ in b}) for f, b in templates.items()}
        if any(missing.values()):
            raise ValueError(f"reference frames do not cover all digits: {missing}")
        return cls(templates)

    def _read_slot(self, win: np.ndarray, font: str) -> tuple[str, float, float, float]:
        """Best char, its score, margin to best other char, ink outside the matched glyph."""
        pad = cv2.copyMakeBorder(win, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
        best: dict[str, tuple[float, tuple[int, int], tuple[int, int]]] = {}
        for ch, t in self.templates[font]:
            if t.shape[0] > pad.shape[0] or t.shape[1] > pad.shape[1]:
                continue
            res = cv2.matchTemplate(pad, t, cv2.TM_CCOEFF_NORMED)
            _, s, _, loc = cv2.minMaxLoc(res)
            if ch not in best or s > best[ch][0]:
                best[ch] = (s, loc, t.shape)
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
        ch, (s, (x, y), (h, w)) = ranked[0]
        margin = s - ranked[1][1][0] if len(ranked) > 1 else 1.0
        ink = pad > INK
        ink[max(0, y - 1):y + h + 1, max(0, x - 1):x + w + 1] = False
        return ch, float(s), float(margin), float(ink.sum())

    def read_field(self, g: np.ndarray, name: str) -> dict:
        spec = FIELDS[name]
        r0, r1 = spec["rows"]
        band = g[r0:r1]
        chars, scores, margins, reasons = [], [], [], []
        for c0, c1 in spec["slots"]:
            ch, s, m, stray = self._read_slot(band[:, c0:c1], spec["font"])
            chars.append(ch); scores.append(s); margins.append(m)
            if s < OCR_MIN_SCORE:
                reasons.append(f"low_score:{s:.2f}")
            if m < OCR_MIN_MARGIN:
                reasons.append(f"ambiguous:{m:.2f}")
            if stray > 3:
                reasons.append("stray_ink_in_slot")
        d0, d1 = spec["dot"]
        dot = band[:, d0:d1] > INK
        if dot.sum() < 2 or dot[: band.shape[0] // 2].any():
            reasons.append("no_decimal_point")
        # ink in the label box outside slots and dot (extra digit, minus sign, corruption)
        b0, b1 = spec["box"]
        allowed = np.zeros(FRAME_W, bool)
        for c0, c1 in spec["slots"] + [spec["dot"]]:
            allowed[c0:c1] = True
        outside = band[:, b0:b1][:, ~allowed[b0:b1]] > INK
        if outside.sum() > 2:
            reasons.append("ink_outside_slots")
        value = float(f"{chars[0]}{chars[1]}.{chars[2]}")
        return dict(value=value, ok=not reasons, min_score=min(scores), min_margin=min(margins),
                    reason=";".join(reasons))

    def read(self, img_bgr: np.ndarray) -> dict:
        g = _gray(img_bgr)
        out = {}
        for name in FIELDS:
            r = self.read_field(g, name)
            out[f"ocr_{name}"] = r["value"]
            out[f"ocr_{name}_ok"] = r["ok"]
            out[f"ocr_{name}_score"] = r["min_score"]
            out[f"ocr_{name}_margin"] = r["min_margin"]
            out[f"ocr_{name}_reason"] = r["reason"]
        return out


def label_crops(img_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Crops of the three numeric labels only (for QC figures; contain no scene content)."""
    return {"spot": img_bgr[2:28, 4:62], "max": img_bgr[3:26, 592:637], "min": img_bgr[451:476, 592:637]}


# --------------------------------------------------------------------------------------
# Colour bar / palette inversion
# --------------------------------------------------------------------------------------
BAR_X = (627, 633)          # bar interior columns used (border at 624-625 and 634-635)
BAR_ROWS = (30, 448)        # interior rows: 30 = MAX end (top), 447 = MIN end (bottom)


def bar_column(img_bgr: np.ndarray) -> np.ndarray:
    """Per-row median colour (BGR float) of the bar interior, top (MAX) to bottom (MIN)."""
    strip = img_bgr[BAR_ROWS[0]:BAR_ROWS[1], BAR_X[0]:BAR_X[1]].astype(np.float32)
    return np.median(strip, axis=1)


def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    """uint8/float BGR (any shape [...,3], 0-255) -> CIELAB (D65), float32."""
    a = (np.asarray(bgr, np.float32) / 255.0).reshape(-1, 1, 3)
    return cv2.cvtColor(a, cv2.COLOR_BGR2LAB).reshape(np.shape(bgr))


@dataclass
class Palette:
    rows_bgr: np.ndarray            # (n_rows, 3) canonical per-row colour, top -> bottom
    level_lab: np.ndarray           # (K, 3)
    level_pos: np.ndarray           # (K,) position 0 (MIN, bottom row) .. 1 (MAX, top row)
    level_halfwidth: np.ndarray     # (K,) half extent of the level's plateau, in position units
    merge_de: float

    @property
    def K(self) -> int:
        return len(self.level_pos)

    @classmethod
    def from_bars(cls, bars: np.ndarray, merge_de: float = 0.5) -> "Palette":
        """bars: (n_frames, n_rows, 3) per-frame bar columns.  Canonical = per-row median."""
        rows = np.median(bars, axis=0)
        lab = bgr_to_lab(rows)
        n = len(rows)
        pos_row = (n - 1 - np.arange(n)) / (n - 1)      # top row -> 1, bottom row -> 0
        # merge runs of consecutive rows with (near-)identical colour into one level
        groups, cur = [], [0]
        for i in range(1, n):
            if np.linalg.norm(lab[i] - lab[cur[0]]) < merge_de:
                cur.append(i)
            else:
                groups.append(cur); cur = [i]
        groups.append(cur)
        level_lab = np.array([lab[g].mean(0) for g in groups], np.float32)
        level_pos = np.array([pos_row[g].mean() for g in groups])
        half = np.array([(pos_row[g[0]] - pos_row[g[-1]]) / 2 + 0.5 / (n - 1) for g in groups])
        return cls(rows, level_lab, level_pos, half, merge_de)

    def frame_bar_de(self, bar: np.ndarray) -> float:
        """Median Delta E between a frame's bar and the canonical palette rows."""
        return float(np.median(np.linalg.norm(bgr_to_lab(bar) - bgr_to_lab(self.rows_bgr), axis=1)))


class Inverter:
    """RGB -> palette level LUT over all 256^3 colours (nearest level in CIELAB)."""

    def __init__(self, palette: Palette, cache: Path | None = None):
        self.palette = palette
        key = f"lut_K{palette.K}_{hashlib.sha1(palette.level_lab.round(3).tobytes()).hexdigest()[:12]}.npz"
        if cache is not None and (cache / key).exists():
            z = np.load(cache / key)
            self.lut_level, self.lut_de = z["level"], z["de"]
            return
        tree = cKDTree(palette.level_lab.astype(np.float64))
        level = np.empty(256 ** 3, np.uint16)
        de = np.empty(256 ** 3, np.float16)
        b, g = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
        for r in range(256):          # chunk by red channel (index = b*65536 + g*256 + r)
            bgr = np.stack([b.ravel(), g.ravel(), np.full(b.size, r)], 1).astype(np.uint8)
            dist, idx = tree.query(bgr_to_lab(bgr).astype(np.float64), workers=-1)
            sel = b.ravel() * 65536 + g.ravel() * 256 + r
            level[sel] = idx
            de[sel] = dist
        self.lut_level, self.lut_de = level, de
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
            np.savez(cache / key, level=level, de=de)

    def levels(self, img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Palette position (0..1) and Delta E per pixel."""
        i = (img_bgr[..., 0].astype(np.int64) << 16) | (img_bgr[..., 1].astype(np.int64) << 8) | img_bgr[..., 2]
        lv = self.lut_level[i]
        return self.palette.level_pos[lv], self.lut_de[i].astype(np.float32)

    def temperature(self, img_bgr: np.ndarray, tmin: float, tmax: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Temperature map (deg C), Delta E map and plateau half-width (deg C) per pixel."""
        i = (img_bgr[..., 0].astype(np.int64) << 16) | (img_bgr[..., 1].astype(np.int64) << 8) | img_bgr[..., 2]
        lv = self.lut_level[i]
        span = tmax - tmin
        return (tmin + span * self.palette.level_pos[lv]).astype(np.float32), \
            self.lut_de[i].astype(np.float32), (span * self.palette.level_halfwidth[lv]).astype(np.float32)


# --------------------------------------------------------------------------------------
# Static overlay mask, spot meter
# --------------------------------------------------------------------------------------
CROSS_C = (319.5, 239.5)        # (x, y) centre of the crosshair ring
SPOT_RADIUS = 3.5               # inner, overlay-free disc used for the spot comparison
OVERLAY_TEXT_BOXES = {          # x0, y0, x1, y1 (exclusive) - text / logo / colour bar
    "spot_label": (0, 0, 90, 30),
    "title": (212, 0, 428, 30),
    "max_label": (588, 0, 640, 28),
    "colour_bar": (616, 20, 640, 460),
    "min_label": (588, 448, 640, 480),
    "logo": (0, 438, 102, 480),
}


def overlay_masks() -> tuple[np.ndarray, np.ndarray]:
    """(text_mask, crosshair_mask) booleans (True = overlay pixel)."""
    text = np.zeros((FRAME_H, FRAME_W), bool)
    for x0, y0, x1, y1 in OVERLAY_TEXT_BOXES.values():
        text[y0:y1, x0:x1] = True
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
    r = np.hypot(xx - CROSS_C[0], yy - CROSS_C[1])
    cross = (r >= 4.5) & (r <= 10.0)
    cross |= (np.abs(yy - 239.5) <= 2.5) & (xx >= 299) & (xx <= 341) & (r > 4.5)
    cross |= (np.abs(xx - 318.5) <= 2.5) & (yy >= 219) & (yy <= 261) & (r > 4.5)
    return text, cross


def spot_disc() -> np.ndarray:
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
    return np.hypot(xx - CROSS_C[0], yy - CROSS_C[1]) <= SPOT_RADIUS


# --------------------------------------------------------------------------------------
# Camera stability
# --------------------------------------------------------------------------------------


def structure_image(img_bgr: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    """Scale-invariant structure image for registration: MSX edge strength (local
    lightness high-pass, independent of the auto-ranged colours), overlays zeroed."""
    L = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    hp = L - cv2.GaussianBlur(L, (0, 0), 3)
    hp[text_mask] = 0
    hp[200:280, 280:360] = 0      # crosshair
    return hp


def phase_shift(ref: np.ndarray, img: np.ndarray, region: tuple[int, int, int, int]) -> tuple[float, float, float]:
    """(dx, dy, response) of img relative to ref within region (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = region
    a, b = ref[y0:y1, x0:x1], img[y0:y1, x0:x1]
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), win)
    return float(dx), float(dy), float(resp)


# --------------------------------------------------------------------------------------
# Seat ROIs
# --------------------------------------------------------------------------------------


@dataclass
class SeatSpec:
    seat: str                        # 'L', 'R' (two-seat folders), 'C' (P10 single central seat)
    box: tuple[int, int, int, int]   # x0, y0, x1, y1 head search region
    expected: str = "used"           # 'used' | 'empty_control' (seat not used in this folder)


# Camera poses.  The camera was static within each pose; poses were identified by phase
# correlation of the MSX edge structure (shift vs P03A-P04A reference: B 0 px, C +37 px,
# C0 +47 px, D -56 px; A ~ -94 px) and the P01A-P02A move between snapshots 398 and 399
# of stream 2025-06-24_12-46-10 was confirmed visually.  Head search boxes (x0, y0, x1, y1)
# were placed on the per-pose warm-pixel probability maps (head + neck, above the laptop
# lids, with margin for posture changes).
POSE_SEATS: dict[str, list[SeatSpec]] = {
    "A": [SeatSpec("L", (0, 105, 150, 268)), SeatSpec("R", (328, 105, 475, 268))],
    "B": [SeatSpec("L", (45, 105, 205, 268)), SeatSpec("R", (355, 105, 515, 268))],
    "C": [SeatSpec("L", (70, 100, 235, 268)), SeatSpec("R", (372, 100, 540, 268))],
    "C0": [SeatSpec("L", (80, 100, 245, 268)), SeatSpec("R", (382, 100, 550, 268))],
    "D": [SeatSpec("C", (285, 115, 450, 280)), SeatSpec("L", (0, 105, 160, 268), "empty_control")],
}
SINGLE_PARTICIPANT_EMPTY = {"P07A": "L", "P07B": "L"}   # seat present in pose but not used
# "Corridor" regions at seated-head height (y 105-240) beside/between the seat search boxes, where
# no seated participant's head should be: warm pixels here indicate another person (experimenter).
POSE_CORRIDORS: dict[str, list[tuple[int, int, int, int]]] = {
    "A": [(150, 105, 328, 240), (475, 105, 610, 240)],
    "B": [(205, 105, 355, 240), (515, 105, 610, 240)],
    "C": [(235, 100, 372, 240), (540, 100, 610, 240)],
    "C0": [(245, 100, 382, 240), (550, 100, 610, 240)],
    "D": [(160, 115, 285, 240), (450, 115, 610, 240)],
}


def pose_of(folder: str, stream_id: str, index: int) -> str:
    if folder == "P01A-P02A":
        return "A" if (stream_id == "2025-06-24_12-46-10" and index <= 398) else "B"
    if folder in ("P01B-P02B", "P03A-P04A", "P03B-P04B"):
        return "B"
    if folder == "P05A-P06A":
        return "C" if stream_id == "2025-06-25_10-35-37" else "C0"
    if folder in ("P05B-P06B", "P07A", "P07B", "P08A-P09A", "P08B-P09B"):
        return "C"
    if folder in ("P10A", "P10B"):
        return "D"
    raise KeyError(folder)


def seats_for(folder: str, pose: str) -> list[SeatSpec]:
    out = []
    for s in POSE_SEATS[pose]:
        exp = "empty_control" if SINGLE_PARTICIPANT_EMPTY.get(folder) == s.seat else s.expected
        out.append(SeatSpec(s.seat, s.box, exp))
    return out


def filtered_temperature(T: np.ndarray, valid: np.ndarray, size: int = 5) -> np.ndarray:
    """Median-filtered temperature (suppresses 1-2 px MSX edge lines).  Invalid pixels
    are first replaced by the value of the nearest valid pixel."""
    Tf = T
    if (~valid).any() and valid.any():
        idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        Tf = T[idx[0], idx[1]]
    return cv2.medianBlur(np.ascontiguousarray(Tf, np.float32), size)


def seat_metrics(Tf: np.ndarray, valid: np.ndarray, plateau_hw: np.ndarray, box: tuple[int, int, int, int],
                 text_mask: np.ndarray, cross_mask: np.ndarray, t_bg: float,
                 blob_drop_c: float = 3.0, top_frac: float = 0.05, top_mask: np.ndarray | None = None) -> dict:
    """Occupancy score and facial (head) blob metrics for one seat in one frame.

    Blob = 8-connected component, inside the search box, of valid pixels with
    Tf >= (box maximum - blob_drop_c) that contains the box maximum.
    ``top_mask`` (optional): pixels in the saturated top palette plateau; their count inside the blob is
    returned so the caller can tell whether the scene maximum (MAX label) lies on this face.
    """
    x0, y0, x1, y1 = box
    sub = Tf[y0:y1, x0:x1]
    v = valid[y0:y1, x0:x1]
    out = dict(occ_score=np.nan, T_max=np.nan, T_p95=np.nan, T_top5mean=np.nan, T_median=np.nan, T_mean=np.nan,
               sat_frac=np.nan, roi_area_px=0,
               roi_x0=np.nan, roi_y0=np.nan, roi_x1=np.nan, roi_y1=np.nan, roi_touches_text=False,
               roi_touches_crosshair=False, roi_touches_box_edge=False, tmax_plateau_hw=np.nan,
               valid_frac=float(v.mean()), n_top_in_roi=0)
    if v.mean() < 0.5:
        return out
    vals = sub[v]
    out["occ_score"] = float(np.percentile(vals, 99) - t_bg)
    smax = np.where(v, sub, -np.inf)
    iy, ix = np.unravel_index(np.argmax(smax), smax.shape)
    tmax = sub[iy, ix]
    lab, _ = ndimage.label(v & (sub >= tmax - blob_drop_c), structure=np.ones((3, 3)))
    blob = lab == lab[iy, ix]
    bv = np.sort(sub[blob])
    ys, xs = np.nonzero(blob)
    ntop = max(1, int(np.ceil(top_frac * len(bv))))
    full = np.zeros_like(valid)
    full[y0:y1, x0:x1] = blob
    out.update(T_max=float(tmax), T_p95=float(np.percentile(bv, 95)), T_top5mean=float(bv[-ntop:].mean()),
               T_median=float(np.median(bv)), T_mean=float(bv.mean()),
               sat_frac=float(top_mask[y0:y1, x0:x1][blob].mean()) if top_mask is not None else np.nan,
               roi_area_px=int(blob.sum()), roi_x0=int(x0 + xs.min()), roi_y0=int(y0 + ys.min()),
               roi_x1=int(x0 + xs.max() + 1), roi_y1=int(y0 + ys.max() + 1),
               roi_touches_text=bool(ndimage.binary_dilation(full, iterations=2)[text_mask].any()),
               roi_touches_crosshair=bool(ndimage.binary_dilation(full, iterations=2)[cross_mask].any()),
               roi_touches_box_edge=bool(ys.min() == 0 or xs.min() == 0 or ys.max() == sub.shape[0] - 1
                                         or xs.max() == sub.shape[1] - 1),
               tmax_plateau_hw=float(plateau_hw[y0 + iy, x0 + ix]),
               n_top_in_roi=int(top_mask[y0:y1, x0:x1][blob].sum()) if top_mask is not None else 0)
    return out


# --------------------------------------------------------------------------------------
# Occupancy threshold, vacancy intervals, agreement statistics
# --------------------------------------------------------------------------------------


def otsu_threshold(x: np.ndarray, bins: int = 256) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    h, e = np.histogram(x, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(c[np.argmax(between)])


def smooth_occupancy(occ: np.ndarray, min_run: int = 2) -> np.ndarray:
    """Remove isolated flips shorter than ``min_run`` samples (NaN = unknown kept)."""
    o = occ.astype(float).copy()
    known = np.isfinite(o)
    vals = o[known]
    if len(vals) == 0:
        return o
    # run-length on known samples
    starts = np.r_[0, np.nonzero(np.diff(vals))[0] + 1]
    ends = np.r_[starts[1:], len(vals)]
    for s, e in zip(starts, ends):
        if e - s < min_run and s > 0 and e < len(vals) and vals[s - 1] == vals[e]:
            vals[s:e] = vals[s - 1]
    o[known] = vals
    return o


def vacancy_intervals(times: pd.Series, occupied: np.ndarray, min_dur_s: float = 30.0) -> list[dict]:
    """Runs of consecutive vacant snapshots (occupied == 0) within one stream.

    A single unknown sample (NaN) between vacant samples does not break a run.
    The true transition times are only known to lie between snapshots, so each
    interval is reported with bounds: the vacancy started after the last occupied
    snapshot (``start_earliest``) and no later than the first vacant one
    (``start_latest``); likewise for the end.  The point estimates are the
    midpoints and ``duration_s`` = end - start.  Runs that touch the start or end
    of the stream are censored (bound = NaT).  Kept if ``duration_s >= min_dur_s``.
    """
    t = pd.to_datetime(pd.Series(times)).reset_index(drop=True)
    o = np.asarray(occupied, float)
    out, i, n = [], 0, len(o)
    while i < n:
        if o[i] != 0:
            i += 1
            continue
        j, n_unknown = i, 0
        while j + 1 < n and (o[j + 1] == 0 or (np.isnan(o[j + 1]) and j + 2 < n and o[j + 2] == 0)):
            j += 1
            n_unknown += int(np.isnan(o[j]))
        prev_t = t[i - 1] if i > 0 else pd.NaT
        next_t = t[j + 1] if j + 1 < n else pd.NaT
        start = t[i] if pd.isna(prev_t) else prev_t + (t[i] - prev_t) / 2
        end = t[j] if pd.isna(next_t) else t[j] + (next_t - t[j]) / 2
        dur = (end - start).total_seconds()
        if dur >= min_dur_s:
            out.append(dict(start_local=start, end_local=end, duration_s=round(dur, 1),
                            start_earliest=prev_t, start_latest=t[i], end_earliest=t[j], end_latest=next_t,
                            duration_min_s=round((t[j] - t[i]).total_seconds(), 1),
                            duration_max_s=round((next_t - prev_t).total_seconds(), 1)
                            if not (pd.isna(prev_t) or pd.isna(next_t)) else np.nan,
                            n_vacant_snapshots=j - i + 1 - n_unknown, n_unknown_snapshots=n_unknown,
                            censored_start=i == 0, censored_end=j == n - 1))
        i = j + 1
    return out


def bland_altman(a: np.ndarray, b: np.ndarray) -> dict:
    """Agreement of method a against reference b (difference = a - b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    m = np.isfinite(d)
    d = d[m]
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    n = len(d)
    se_bias = sd / np.sqrt(n)
    se_loa = sd * np.sqrt(3.0 / n)
    return dict(n=n, bias=bias, sd=sd, loa_low=bias - 1.96 * sd, loa_high=bias + 1.96 * sd,
                bias_ci95=[bias - 1.96 * se_bias, bias + 1.96 * se_bias],
                loa_low_ci95=[bias - 1.96 * sd - 1.96 * se_loa, bias - 1.96 * sd + 1.96 * se_loa],
                loa_high_ci95=[bias + 1.96 * sd - 1.96 * se_loa, bias + 1.96 * sd + 1.96 * se_loa],
                mae=float(np.abs(d).mean()), within_0_1=float((np.abs(d) <= 0.1 + 1e-9).mean()),
                within_0_3=float((np.abs(d) <= 0.3 + 1e-9).mean()), max_abs=float(np.abs(d).max()))


STRUCT_SCALE = 4                     # structure images are stored at 1/4 resolution (160x120)


def small_structure(img_bgr: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    s = structure_image(img_bgr, text_mask)
    return cv2.resize(s, (FRAME_W // STRUCT_SCALE, FRAME_H // STRUCT_SCALE), interpolation=cv2.INTER_AREA)


def pose_reference_shift(struct_small: np.ndarray, ref_small: np.ndarray) -> tuple[float, float, float]:
    """Shift (full-res px) of a quarter-resolution structure image vs its pose reference
    (region above the table: full-res x 20-600, y 32-450)."""
    dx, dy, resp = phase_shift(ref_small, struct_small, (5, 8, 150, 112))
    return STRUCT_SCALE * dx, STRUCT_SCALE * dy, resp


def env_cache_dir() -> Path:
    return Path(os.environ.get("SYNAPSE_THERMAL_TMP", Path(os.environ.get("TMPDIR", "/tmp")) / "synapse_thermal"))
