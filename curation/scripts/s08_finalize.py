"""Step 08: dataset-level files: dataset_description, participants (with availability), scans.tsv,
README, CHANGES, LICENSE, .bidsignore, sourcedata (raw-archive manifest, deployed form definitions,
decision register), data dictionary."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synapse_curation import __version__, participants, paths, questionnaires, zenodo  # noqa: E402
from synapse_curation.common import NA, decisions, sessions, sha256, write_json, write_tsv  # noqa: E402

B = paths.BIDS
AUTHORS = ["Puneet Tomar", "Tanmay Dave", "Pawel Wargocki", "Anna Laura Pisello"]

RAW_CLASS = [  # (path prefix, status, note)
    ("dataforms/", "used", "task responses and platform event log"),
    ("tasks/", "duplicate", "byte-identical copy of dataforms/ task files"),
    ("pilot/", "excluded", "pilot day 23 June 2025, different people (D02)"),
    ("surveys/", "used", "questionnaires T0-T3 (health-screening items withheld)"),
    ("ParticipantInformation.xlsx", "restricted", "background questionnaire with direct identifiers; released only unlinked and generalised (D08)"),
    ("eeg/", "used", "Muse S EEG (see derivatives/quality_control/eeg_source_inventory.tsv)"),
    ("muse/", "used", "Muse S EEG (see derivatives/quality_control/eeg_source_inventory.tsv)"),
    ("leap_csv/Monday", "excluded", "pilot day (D02)"),
    ("leap_csv/", "used", "ActiGraph LEAP ActiLife exports"),
    ("leap_raw/Monday", "excluded", "pilot day (D02)"),
    ("leap_raw/", "inventoried", "vendor binary files (.agsd/.agd), identical to those inside leap_csv archives; contain the acquisition host name"),
    ("thermal/", "restricted", "FLIR AX8 rendered video and snapshots showing faces (D26); derivatives distributed"),
    ("atmo/", "used", "ATMOCUBE room monitor"),
    ("metadata/deployed_google_forms/", "used", "deployed questionnaire definitions (copied to sourcedata)"),
    ("metadata/", "inventoried", "questionnaire design drafts and stimulus prompts"),
    ("stimuli/", "used", "soundscape recordings (copied to stimuli/)"),
    ("experiment_log.txt", "restricted", "free-text operator log (D26); transcribed into sessions.tsv"),
]


def classify(rel: str) -> tuple[str, str]:
    for pre, status, note in RAW_CLASS:
        if rel.startswith(pre):
            return status, note
    return "inventoried", ""


def raw_manifest() -> pd.DataFrame:
    cache_p = paths.WORK / "raw_hash_cache.tsv"
    cache = pd.read_csv(cache_p, sep="\t").set_index("path") if cache_p.exists() else pd.DataFrame()
    eeg = pd.read_csv(paths.WORK / "eeg_inventory.tsv", sep="\t").set_index("source")["sha256"]
    rows = []
    for f in sorted(paths.RAW.rglob("*")):
        if not f.is_file() or f.name == ".DS_Store":
            continue
        rel = f.relative_to(paths.RAW).as_posix()
        st = f.stat()
        key = f"raw/{rel}"
        if key in eeg.index:
            h = eeg[key]
        elif rel in cache.index and cache.loc[rel, "bytes"] == st.st_size and cache.loc[rel, "mtime"] == int(st.st_mtime):
            h = cache.loc[rel, "sha256"]
        else:
            h = sha256(f)
        status, note = classify(rel)
        shown = re.sub(r"/(P\d\d)-[A-Za-z]+/", r"/\1-[label]/", rel)        # operator folder labels redacted
        if shown != rel:
            note = (note + "; " if note else "") + "folder label redacted in this listing"
        rows.append({"path": shown, "raw_path": rel, "bytes": st.st_size, "mtime": int(st.st_mtime), "sha256": h, "status": status, "note": note})
    d = pd.DataFrame(rows)
    d.drop(columns="path").rename(columns={"raw_path": "path"}).to_csv(cache_p, sep="\t", index=False)
    return d.drop(columns=["mtime", "raw_path"])


def availability() -> pd.DataFrame:
    a = pd.read_pickle(paths.WORK / "availability_core.pkl")
    for f in ("availability_eeg.pkl", "availability_physio.pkl"):
        a = a.merge(pd.read_pickle(paths.WORK / f), on=["participant_id", "session_id"], how="left")
    th = pd.read_csv(B / "derivatives" / "facial_temperature" / "thermal_coverage.tsv", sep="\t")
    a = a.merge(th[["participant_id", "session_id"]].assign(has_thermal=True), on=["participant_id", "session_id"], how="left")
    return a.fillna(False).infer_objects()


def scans() -> None:
    for sub in sorted(B.glob("sub-*")):
        for ses in sorted(sub.glob("ses-*")):
            rows = []
            for f in sorted(ses.rglob("*")):
                if f.suffix in (".tsv", ".vhdr", ".gz") and not f.name.endswith(("_events.tsv", "_channels.tsv", "_scans.tsv")):
                    rel = f.relative_to(ses).as_posix()
                    if f.name.endswith("_physio.tsv.gz"):
                        start = json.loads(Path(str(f).replace(".tsv.gz", ".json")).read_text())["SYNAPSE_StartTimeLocal"][:19]
                    elif f.suffix == ".vhdr":
                        start = pd.read_csv(f.parent / f.name.replace("_eeg.vhdr", "_events.tsv"), sep="\t")
                        start = pd.Timestamp(start["onset_time"].iloc[0]) - pd.Timedelta(seconds=float(start["onset"].iloc[0]))
                        start = start.strftime("%Y-%m-%dT%H:%M:%S")
                    else:
                        t = pd.read_csv(f, sep="\t", usecols=["submitted_time"])["submitted_time"]
                        start = str(t.iloc[0])[:19]
                    rows.append({"filename": rel, "acq_time": start})
            write_tsv(pd.DataFrame(rows), ses / f"{sub.name}_{ses.name}_scans.tsv")
    write_json({"acq_time": {"Description": "Acquisition time of the first sample/response in the file (local time, CEST, UTC+02:00)"}},
               B / "scans.json")


def main() -> None:
    dec = decisions()
    a = availability()
    write_tsv(participants.participants_table(a), B / "participants.tsv")
    p = json.loads((B / "participants.json").read_text())
    for c in ("behaviour", "eeg", "wristphysio", "thermal"):
        p[f"n_sessions_{c}"] = {"Description": f"Number of sessions (of 2) with {c} data, complete or partial; completeness per recording in derivatives/quality_control"}
    write_json(p, B / "participants.json")
    scans()
    write_json({
        "Name": zenodo.TITLE,
        "BIDSVersion": "1.10.0", "DatasetType": "raw", "License": "CC-BY-4.0", "Authors": AUTHORS,
        "Acknowledgements": "We thank the participants. Experiments were conducted in the field-labs of the Department of Environmental and Resource Engineering, Technical University of Denmark.",
        "HowToAcknowledge": "Cite the dataset (DOI below) and the associated Data Descriptor.",
        "Funding": ["European Union Horizon Europe, Marie Sklodowska-Curie Actions Doctoral Network MuSIC, grant agreement No. 101073357"],
        "EthicsApprovals": ["No Ethics Review Board approval was required under the procedures of the Technical University of Denmark for this type of study; all participants gave signed informed consent, including to the sharing of pseudonymised data (decision D24)"],
        "ReferencesAndLinks": ["https://doi.org/10.17504/protocols.io.m7vznm25rvx1/v1", "https://doi.org/10.5281/zenodo.23768059", "https://synapse.ptomar.com", "https://doi.org/10.1016/j.buildenv.2026.115068"],
        "DatasetDOI": "doi:" + dec["D27_doi"]["value"].split(" ")[0],
        "SourceDatasets": [{"Description": "SYNAPSE raw archive (Level 0); manifest with SHA-256 in sourcedata/raw_file_inventory.tsv"}],
        "GeneratedBy": [{"Name": "synapse_curation", "Version": __version__, "Description": "Cleaning and organisation of the SYNAPSE raw recordings into BIDS; provenance in sourcedata/"}],
        "Keywords": ["indoor environmental quality", "correlated colour temperature", "soundscape", "cognitive performance",
                     "EEG", "photoplethysmography", "skin temperature", "infrared thermography", "multimodal"],
    }, B / "dataset_description.json")
    (B / ".bidsignore").write_text("/environment/**\n/environment\n/background_questionnaire/**\n/background_questionnaire\n/DATA_DICTIONARY.tsv\n/CITATION.cff\n")
    shutil.copy2(paths.PKG_ROOT / "config" / "LICENSE_dataset.txt", B / "LICENSE")
    # sourcedata: decision register, deployed forms, raw manifest, session table
    sd = B / "sourcedata"
    forms_dir = sd / "questionnaire_forms"
    shutil.rmtree(forms_dir, ignore_errors=True)
    for key, names, title in (("T0", questionnaires.T0_NAMES, "T0 pre-session questionnaire"),
                              ("T1", questionnaires.T1_NAMES, "T1 indoor-environment questionnaire"),
                              ("T23", questionnaires.T23_NAMES, "T2/T3 task-experience, affect and indoor-environment questionnaire")):
        items = questionnaires.item_table(key, names)
        items["responses_distributed"] = ~items["variable"].isin(questionnaires.HEALTH)
        write_json({"Questionnaire": title,
                    "Description": "Items of the Google Form as deployed in the task platform: variable name in the phenotype tables, "
                                   "question wording, help text, response options and whether an answer was required. "
                                   "Health-screening answers are not distributed (responses_distributed = false).",
                    "Items": items.to_dict(orient="records")},
                   forms_dir / f"form_{'T2_T3' if key == 'T23' else key}.json")
    shutil.copy2(paths.CONFIG / "decisions.yaml", sd / "curation_decisions.yaml")
    for src, dst in (("sessions.tsv", "operator_log_transcription.tsv"), ("eeg_assignment.tsv", "eeg_file_assignment.tsv")):
        write_tsv(pd.read_csv(paths.CONFIG / src, sep="\t", dtype=str, keep_default_na=False), sd / dst)
    write_tsv(raw_manifest(), sd / "raw_file_inventory.tsv")
    subprocess.run([sys.executable, "-B", str(paths.PKG_ROOT / "scripts" / "s09_docs.py")], check=True)


if __name__ == "__main__":
    main()
