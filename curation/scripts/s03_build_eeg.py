"""Step 03: EEG -> BIDS (BrainVision int16, bit-exact) + channels/events/sidecars, lossless host-time
companion in sourcedata, and QC derivatives (coverage, gaps, stalls, inventory)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synapse_curation import eeg, paths  # noqa: E402
from synapse_curation.common import NA, decisions, unix_to_iso, write_json, write_tsv  # noqa: E402

B = paths.BIDS
TASK = "cogbattery"


def main() -> None:
    man = pd.read_csv(paths.WORK / "eeg_inventory.tsv", sep="\t")
    assign = pd.read_csv(paths.CONFIG / "eeg_assignment.tsv", sep="\t", dtype=str, keep_default_na=False)
    tl = pd.read_pickle(paths.WORK / "timeline.pkl")
    recs = eeg.assemble(man, assign)
    qc, gaps = [], []
    for (pid, ses), rec in sorted(recs.items()):
        d = rec["data"]
        folder = B / pid / ses / "eeg"
        base = f"{pid}_{ses}_task-{TASK}"
        t_sess = tl[(tl.participant_id == pid) & (tl.session_id == ses)]
        s0 = t_sess.loc[t_sess.phase == "T0", "offset"].min()
        s1 = np.nanmax([t_sess.loc[t_sess.phase == "T3", "offset"].max(), t_sess["offset"].max()])
        # events: phases and blocks from the platform timeline + recording discontinuities
        ev = t_sess[t_sess.phase != "T0"].copy()
        ev["trial_type"] = np.where(ev.phase == "block", "task_" + ev["task"].astype(str), ev["phase"])
        on = eeg.host_to_file_time(rec, ev["onset"].to_numpy(float))
        off = eeg.host_to_file_time(rec, ev["offset"].to_numpy(float))
        ev = ev.assign(onset_file=on, offset_file=off)
        ev = ev[np.isfinite(ev.onset_file)]
        ev_out = pd.DataFrame({"onset": ev.onset_file.round(4), "duration": (ev.offset_file - ev.onset_file).round(4),
                               "trial_type": ev.trial_type, "round": ev["round"], "soundscape": ev["soundscape"],
                               "onset_time": unix_to_iso(ev["onset"]), "offset_time": unix_to_iso(ev["offset"]),
                               "offset_outside_recording": ~np.isfinite(ev.offset_file)})
        segs = rec["segments"]
        bnd = pd.DataFrame({"onset": [round(s.i0 / eeg.FS, 4) for s in segs[1:]], "duration": 0.0, "trial_type": "boundary",
                            "round": np.nan, "soundscape": NA,
                            "onset_time": unix_to_iso([s.start for s in segs[1:]]),
                            "offset_time": NA, "offset_outside_recording": False,
                            "gap_duration_s": [round(b.start - a.end, 3) for a, b in zip(segs, segs[1:])]})
        events = pd.concat([ev_out, bnd], ignore_index=True).sort_values("onset")
        write_tsv(events, folder / f"{base}_events.tsv")
        eeg.write_brainvision(rec, folder, f"{base}_eeg",
                              extra_events=[{"onset": int(round(o * eeg.FS)), "duration": 1, "description": t, "type": "Comment"}
                                            for o, t in zip(ev.onset_file, ev.trial_type)])
        q = eeg.qc_row(rec, pid, ses, s0, s1)
        notes = ["under the left headphone ear cup", "", "", "under the right headphone ear cup"]
        status_description = []
        for c, note in zip(eeg.CHANNELS, notes):
            rail = q[f"rail_fraction_{c}"]
            parts = [note] if note else []
            if rail >= 0.01:
                parts.append(f"saturated (+-1000 uV) in {rail:.0%} of in-session samples")
            status_description.append("; ".join(parts) or NA)
        write_tsv(pd.DataFrame({"name": eeg.CHANNELS, "type": "EEG", "units": "uV",
                                "low_cutoff": NA, "high_cutoff": NA, "reference": "Fpz",
                                "placement": ["left temporal-parietal (behind the left ear)", "left anterior-frontal (forehead)",
                                              "right anterior-frontal (forehead)", "right temporal-parietal (behind the right ear)"],
                                "status": "good",
                                "status_description": status_description}),
                  folder / f"{base}_channels.tsv")
        dur = d[-1, 0] - d[0, 0]
        write_json({
            "TaskName": "SYNAPSE cognitive battery",
            "TaskDescription": "Whole session: T0 questionnaire, 5-min eyes-open baseline, T1, round 1 (n-back letters, word Stroop, visual counting, arrow flanker) with soundscape 1, T2, round 2 (n-back digits, emoji Stroop, shape comparison, digit flanker) with soundscape 2, T3 (sessions sub-P01/ses-A and sub-P02/ses-A: emotional-word and emotional-image Stroop instead of word and emoji Stroop). Phase and block onsets in _events.tsv.",
            "Manufacturer": "Interaxon", "ManufacturersModelName": "Muse S",
            "SoftwareVersions": "BrainFlow (board MUSE_S_BOARD; version not recorded) via raw/code muse-eeg.py",
            "SamplingFrequency": eeg.FS, "EEGChannelCount": 4, "EOGChannelCount": 0, "ECGChannelCount": 0, "EMGChannelCount": 0,
            "MISCChannelCount": 0, "TriggerChannelCount": 0,
            "EEGReference": "Fpz (device reference electrode)", "EEGGround": "Muse S device ground (forehead)",
            "EEGPlacementScheme": "International 10-20 positions TP9, AF7, AF8, TP10 (dry electrodes integrated in the headband)",
            "CapManufacturer": "Interaxon", "CapManufacturersModelName": "Muse S headband",
            "PowerLineFrequency": 50, "SoftwareFilters": "n/a", "HardwareFilters": "n/a",
            "RecordingType": "discontinuous" if len(rec["segments"]) > 1 else "continuous",
            "RecordingDuration": round(len(d) / eeg.FS, 3),
            "SubjectArtefactDescription": "Dry electrodes; TP9/TP10 under over-ear headphones. Values saturate at +-1000 uV (device rail). No artefact correction applied.",
            "InstitutionName": "Technical University of Denmark", "InstitutionAddress": "Kongens Lyngby, Denmark", "InstitutionalDepartmentName": "Department of Environmental and Resource Engineering",
            "Instructions": "Sit at the workstation, keep the headband in place, and follow the on-screen instructions of the questionnaires and tasks; during the baseline sit still with eyes open.",
            "SYNAPSE_TimeBase": "Sample times = BrainFlow host timestamps (Unix s, acquisition-laptop clock, not synchronised with other devices; D11). File time 0 = first sample; segments are concatenated and each discontinuity is a 'boundary' event with its true gap duration. Per-sample host timestamps: sourcedata/eeg_host_timestamps/.",
            "SYNAPSE_HostClockSpanSeconds": round(dur, 3),
            "SYNAPSE_Resolution_uV": eeg.RESOLUTION_UV,
            "SYNAPSE_DeDuplication": "Each sample was written ~18x by the acquisition script; exact duplicates (same host timestamp and values) were collapsed, no conflicts occurred; values are bit-exact.",
            "SYNAPSE_SourceFiles": rec["sources"], "SYNAPSE_AssignmentEvidence": rec["evidence"],
            "SYNAPSE_AssignmentEvidenceText": rec["evidence_text"],
            "SYNAPSE_SessionCoverageFraction": q["session_coverage_fraction"],
        }, folder / f"{base}_eeg.json")
        hp = B / "sourcedata" / "eeg_host_timestamps" / pid / ses
        hp.mkdir(parents=True, exist_ok=True)
        t0 = eeg.write_hosttime(rec, hp / f"{base}_hosttime.tsv.gz")
        write_json({"Description": "BrainFlow host timestamp of every EEG sample (acquisition-laptop clock; D11). Row i = sample i of the BrainVision file; no header.",
                    "Columns": ["host_time"], "host_time": {"Description": "Host time relative to the first sample", "Units": "s"},
                    "StartTimeUnix": round(t0, 6), "StartTimeLocal": unix_to_iso([t0])[0], "NumberOfSamples": len(d)},
                   hp / f"{base}_hosttime.json")
        qc.append(q)
        gaps.append(eeg.gap_table(rec, pid, ses))
        print(pid, ses, f"{len(d)/eeg.FS/60:.1f} min", len(rec["segments"]), "segments, coverage", q["session_coverage_fraction"], flush=True)
    der = B / "derivatives" / "quality_control"
    write_tsv(pd.DataFrame(qc), der / "eeg_quality.tsv")
    write_tsv(pd.concat(gaps, ignore_index=True), der / "eeg_gaps.tsv")
    inv = man.copy()
    inv["assigned_to"] = inv["source"].map(lambda s: ";".join(
        f"{r.participant_id}/{r.session_id}" for r in assign[assign.source == s[4:]].itertuples() if r.participant_id != "n/a"))
    inv["excluded_reason"] = inv["source"].map(lambda s: ";".join(assign.loc[(assign.source == s[4:]) & (assign.participant_id == "n/a"), "evidence"]))
    write_tsv(inv.drop(columns=["max_quantisation_error_steps"]), der / "eeg_source_inventory.tsv")
    pd.DataFrame(qc)[["participant_id", "session_id"]].assign(has_eeg=True).to_pickle(paths.WORK / "availability_eeg.pkl")


if __name__ == "__main__":
    main()
