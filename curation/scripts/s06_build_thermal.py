"""Step 06: thermal derivatives.  The extraction (s04_thermal.py, method unchanged from the validated
first release; outputs reproduced byte-identically) yields per-snapshot x seat facial-ROI metrics.
Here seats are linked to participants (decision D07), per-session baseline coverage is assessed,
and the derivative tables are written.  Raw frames/video (faces) are restricted (D26)."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synapse_curation import __version__, paths  # noqa: E402
from synapse_curation.common import decisions, local_to_unix, sessions, unix_to_iso, write_json, write_tsv  # noqa: E402

SRC = paths.WORK / "thermal"
OUT = paths.BIDS / "derivatives" / "facial_temperature"
KEEP = ["seat", "valid", "invalid_reason", "occupied", "T_median", "T_median_minus_bg", "T_mean", "T_p95",
        "T_max", "T_background", "sat_frac", "roi_area_px", "valid_frac"]


def main() -> None:
    seats = pd.read_csv(SRC / "thermal_seats.csv")
    seats["time_unix"] = local_to_unix(pd.to_datetime(seats["time_local"]))
    s = sessions()
    tl = pd.read_pickle(paths.WORK / "timeline.pkl")
    rows, cov = [], []
    for r in s.itertuples():
        folder = next(f for f in seats.session_folder.unique()
                      if f"{r.participant}{r.session}" in f.split("-"))
        g = seats[(seats.session_folder == folder) & (seats.seat == r.seat)].copy()
        basis = ("single participant in room (other seat empty)" if r.co_tested_with == "n/a"
                 else "investigator-confirmed seating plan (D07)")
        g = g.sort_values("time_unix")
        out = pd.DataFrame({"time": unix_to_iso(g.time_unix), "time_unix": g.time_unix.round(3),
                            **{c: g[c].to_numpy() for c in KEEP}})
        base = f"sub-{r.participant}_ses-{r.session}"
        write_tsv(out, OUT / f"sub-{r.participant}" / f"ses-{r.session}" / f"{base}_desc-facetemp_timeseries.tsv")
        t = tl[(tl.participant_id == r.participant_id) & (tl.session_id == r.session_id)]
        b = t[t.phase == "baseline"]
        b0, b1 = (float(b.onset.iloc[0]), float(b.offset.iloc[0])) if len(b) else (np.nan, np.nan)
        inb = g[(g.time_unix >= b0) & (g.time_unix <= b1)]
        expected = (b1 - b0) / 10.07 if np.isfinite(b0) else np.nan
        n_valid_b = int(inb.valid.sum())
        cov.append({"participant_id": r.participant_id, "session_id": r.session_id, "session_folder": folder,
                    "seat": r.seat, "link_basis": basis, "n_snapshots": len(g), "n_valid": int(g.valid.sum()),
                    "first_snapshot": unix_to_iso([g.time_unix.min()])[0], "last_snapshot": unix_to_iso([g.time_unix.max()])[0],
                    "baseline_valid_snapshots": n_valid_b, "baseline_expected_snapshots": round(expected, 1),
                    "baseline_usable": bool(np.isfinite(expected) and n_valid_b >= 0.5 * expected)})
    covd = pd.DataFrame(cov)
    write_tsv(covd, OUT / "thermal_coverage.tsv")
    for f in ["thermal_frames.csv", "thermal_streams.csv", "seat_vacancy_events.csv", "other_person_episodes.csv",
              "ocr_manual_check.csv", "occupancy_visual_census.csv"]:
        d = pd.read_csv(SRC / f)
        write_tsv(d, OUT / "quality_control" / f.replace(".csv", ".tsv"))
    shutil.copy2(SRC / "thermal_validation.json", OUT / "quality_control" / "thermal_validation.json")
    shutil.copy2(paths.CONFIG / "thermal_method_README.md", OUT / "quality_control" / "README.md")
    (OUT / "quality_control" / "figures").mkdir(parents=True, exist_ok=True)
    for png in (SRC / "qc_figures").glob("*.png"):
        if png.name not in ("ocr_examples.png",):          # crops of camera frames are not distributed
            shutil.copy2(png, OUT / "quality_control" / "figures" / png.name)
    write_json({"Name": "SYNAPSE thermal derivatives (FLIR AX8)", "BIDSVersion": "1.10.0", "DatasetType": "derivative",
                "GeneratedBy": [{"Name": "synapse_curation", "Version": __version__,
                                 "Description": "OCR of the on-screen temperature scale, CIELAB palette inversion (236 levels) of rendered (non-radiometric) snapshots every ~10 s, seat ROIs; validated against the camera spot meter"}],
                "SourceDatasets": [{"Description": "raw thermal video/snapshots (restricted, D26)"}],
                "CameraSettings": decisions()["D16_thermal_camera"]["value"]},
               OUT / "dataset_description.json")
    write_json({"time": {"Description": "Snapshot time (ISO 8601, CEST), from a linear fit of file times (residual SD ~0.6 s)"},
                "seat": {"Description": "Seat ROI (R wall-side, L other, C centre)"},
                "valid": {"Description": "true if the frame passed OCR, palette, spot-meter and ROI checks"},
                "invalid_reason": {"Description": "Reason a frame was rejected"},
                "occupied": {"Description": "Seat occupancy score > Otsu threshold (participant present)"},
                "T_median": {"Description": "Median temperature in the head ROI (relative use only: non-radiometric source, emissivity 0.95 assumed)", "Units": "degC"},
                "T_median_minus_bg": {"Description": "T_median minus scene background temperature (recommended for within-session change)", "Units": "degC"},
                "T_mean": {"Description": "Mean head-ROI temperature", "Units": "degC"}, "T_p95": {"Description": "95th percentile head-ROI temperature", "Units": "degC"},
                "T_max": {"Description": "Maximum head-ROI temperature (saturates at the scale maximum; do not compare seats)", "Units": "degC"},
                "T_background": {"Description": "Scene background temperature", "Units": "degC"},
                "sat_frac": {"Description": "Fraction of ROI pixels at the top of the scale"}, "roi_area_px": {"Description": "ROI area", "Units": "pixels"},
                "valid_frac": {"Description": "Fraction of ROI pixels successfully inverted"}},
               OUT / "desc-facetemp_timeseries.json")
    print(covd[["participant_id", "session_id", "seat", "n_valid", "baseline_valid_snapshots", "baseline_expected_snapshots", "baseline_usable"]].to_string())


if __name__ == "__main__":
    main()
