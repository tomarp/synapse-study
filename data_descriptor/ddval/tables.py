"""LaTeX tables of the Data Descriptor (booktabs), with counts taken from the dataset."""
from __future__ import annotations

import json

import pandas as pd

from .common import BIDS, GEN, NUM, behaviour, tsv


def _tex(rows: list[list[str]], header: list[str], spec: str) -> str:
    def esc(s) -> str:
        s = str(s).replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
        return s.replace("_", "\\_\\allowbreak{}").replace("/", "/\\allowbreak{}")
    spec = spec.replace("p{", ">{\\raggedright\\arraybackslash}p{")      # narrow columns: ragged right, no underfull boxes
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(f"\\textbf{{{esc(h)}}}" for h in header) + " \\\\", "\\midrule"]
    out += [" & ".join(esc(c) for c in r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def count(pattern: str) -> int:
    return len(list(BIDS.glob(pattern)))


def streams() -> str:
    eq = tsv("derivatives/quality_control/eeg_quality.tsv")
    wc = tsv("derivatives/quality_control/wearable_coverage.tsv")
    th = tsv("derivatives/facial_temperature/thermal_coverage.tsv")
    room = tsv("environment/room_monitor_1min.tsv")
    rows = [
        ["Task responses", "Browser task platform (v1, 2025)", "per trial", "sub-*/ses-*/beh/*_beh.tsv",
         f"20 ({len(behaviour()):,} trials)"],
        ["EEG (TP9, AF7, AF8, TP10)", "Interaxon Muse S, BrainFlow", "256 Hz", "sub-*/ses-*/eeg/ (BrainVision)",
         f"20 ({eq.n_samples.sum() / 256 / 3600:.1f} h)"],
        ["PPG (2 green, 2 ambient)", "ActiGraph LEAP v2, non-dominant wrist", "25 Hz", "*_recording-ppg_physio.tsv.gz",
         f"{int((wc.status == 'available').sum())}"],
        ["Acceleration (3 axes)", "ActiGraph LEAP v2", "32 Hz", "*_recording-acc_physio.tsv.gz", f"{int((wc.status == 'available').sum())}"],
        ["Skin temperature", "ActiGraph LEAP v2", "1/12 Hz", "*_recording-skintemp_physio.tsv.gz", f"{int((wc.status == 'available').sum())}"],
        ["IBI, heart rate, HRV", "ActiLife 7.1 (device software)", "beat, 1 min, 300 s", "derivatives/wristband_vendor_metrics/",
         f"{int((wc.status == 'available').sum())}"],
        ["Facial temperature (relative)", "FLIR AX8, rendered video", "~0.1 Hz", "derivatives/facial_temperature/", f"{len(th)}"],
        ["Room conditions", "ATMOCUBE", "1 min", "environment/room_monitor_1min.tsv", f"20 ({len(room):,} records)"],
        ["Questionnaires T0--T3", "Google Forms in the task platform", "per form", "phenotype/", "20 (80 forms)"],
        ["Background questionnaire", "Paper/online form", "once", "background_questionnaire/", "9 of 10 (unlinked)"],
    ]
    return _tex(rows, ["Stream", "Device / source", "Sampling", "Location in the dataset", "Sessions"], "p{2.9cm}p{3.3cm}p{1.5cm}p{4.6cm}p{1.8cm}")


def questionnaires() -> str:
    rows = []
    spec = {"T0": ("T0_pre_session", "Before the baseline", "Alertness, energy, motivation, stress; clothing; caffeine and meal; cold symptoms; sleep; initial IEQ ratings"),
            "T1": ("T1_ieq", "After the baseline", "Thermal, air-quality, lighting and acoustic sensation, satisfaction and preference"),
            "T2": ("T2_task_affect_ieq", "After round 1", "Task difficulty, focus, relaxation/alertness, emotional stability, engagement, cognitive load, valence, arousal; T1 items"),
            "T3": ("T3_task_affect_ieq", "After round 2", "As T2")}
    for k, (stem, when, content) in spec.items():
        cols = pd.read_csv(BIDS / "phenotype" / f"{stem}_sessionA.tsv", sep="\t", nrows=0).columns
        items = [c for c in cols if c.endswith("_code")]
        rows.append([k, when, content, str(len(items))])
    forms = json.loads((BIDS / "sourcedata/questionnaire_forms/form_T0.json").read_text())
    NUM.set("TZeroHealthItems", sum(not i["responses_distributed"] for i in forms["Items"]))
    return _tex(rows, ["Form", "When", "Content", "Coded items"], "p{0.9cm}p{2.3cm}p{9.1cm}p{1.4cm}")


def records() -> str:
    rows = [
        ["participants.tsv/.json", "Participant table: group, co-tested participant, seat, lighting order, data availability", "1"],
        ["sub-*/sub-*_sessions.tsv", "Session table: lighting, soundscape per round, times, task configuration, operator notes", f"{count('sub-*/sub-*_sessions.tsv')}"],
        ["sub-*/ses-*/beh/*_beh.tsv", "Trial tables, one per task and session", f"{count('sub-*/ses-*/beh/*_beh.tsv')}"],
        ["sub-*/ses-*/beh/*_physio.tsv.gz", "Wrist signals (PPG, acceleration, skin temperature)", f"{count('sub-*/ses-*/beh/*_physio.tsv.gz')}"],
        ["sub-*/ses-*/eeg/", "EEG (.vhdr, .vmrk, .eeg), channels and events", f"{count('sub-*/ses-*/eeg/*_eeg.eeg')} recordings"],
        ["phenotype/", "Questionnaires T0--T3 per session", f"{count('phenotype/*.tsv')}"],
        ["background_questionnaire/", "Generalised, unlinked background questionnaire and sample summary", f"{count('background_questionnaire/*.tsv')}"],
        ["environment/", "Room monitor, 1-min records", f"{count('environment/*.tsv')}"],
        ["stimuli/", "Image provenance, visual-item answer-key review, soundscape characterisation", f"{count('stimuli/*.*') // 1}"],
        ["derivatives/session_timeline/", "Onset and offset of every phase and task block", f"{count('derivatives/session_timeline/*.tsv')}"],
        ["derivatives/facial_temperature/", "Facial temperature series, seat links, extraction quality control", f"{count('derivatives/facial_temperature/sub-*/ses-*/*.tsv')} series"],
        ["derivatives/wristband_vendor_metrics/", "Device-computed IBI, heart rate and HRV", f"{count('derivatives/wristband_vendor_metrics/sub-*/ses-*/*.tsv')}"],
        ["derivatives/quality_control/", "EEG quality and gaps, wearable coverage, clock checks, linkage, task configuration", f"{count('derivatives/quality_control/*.tsv')}"],
        ["sourcedata/", "Raw-file inventory, curation decisions, EEG file assignment, operator-log times, form definitions, EEG host timestamps", f"{sum(1 for f in (BIDS / 'sourcedata').rglob('*') if f.is_file())}"],
        ["DATA_DICTIONARY.tsv", "Every column of every table with description, units and levels", "1"],
    ]
    NUM.set("NFiles", sum(1 for f in BIDS.rglob("*") if f.is_file()))
    NUM.set("SizeMB", sum(f.stat().st_size for f in BIDS.rglob("*") if f.is_file()) / 1e6, "{:.0f}")
    NUM.set("NColumns", len(pd.read_csv(BIDS / "DATA_DICTIONARY.tsv", sep="\t")))
    return _tex(rows, ["Path", "Content", "Files"], "p{5.2cm}p{7.8cm}p{1.4cm}")


def sample() -> None:
    summ = tsv("background_questionnaire/sample_summary.tsv").set_index("statistic")["value"]
    NUM.set("SampleSex", f"{int(summ['female'])} female, {int(summ['male'])} male")
    NUM.set("SampleAge", f"{summ['age_mean_years']:.1f}~$\\pm$~{summ['age_sd_years']:.1f} years, range "
                         f"{int(summ['age_min_years'])}--{int(summ['age_max_years'])}")
    NUM.set("NTrialsTotal", f"{len(behaviour()):,}".replace(",", "{,}"))
    snd = json.loads((BIDS / "stimuli" / "soundscapes.json").read_text())["files"]
    NUM.set("TrafficRelDB", next(s["level_A_relative_to_babble_dB"] for s in snd if "traffic" in s["file"].lower()), "{:.1f}")


def make() -> None:
    sample()
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "table_streams.tex").write_text(streams())
    (GEN / "table_questionnaires.tex").write_text(questionnaires())
    (GEN / "table_records.tex").write_text(records())
