# synapse_curation — raw recordings → BIDS dataset

Reproducible curation of the SYNAPSE study (10 participants × 2 sessions, DTU field laboratory, 24–26 June 2025) from
the read-only raw recordings into the published BIDS 1.10 dataset (https://doi.org/10.5281/zenodo.23458777). The raw
recordings are not public (they contain identifiable material), so this part documents and reproduces the curation for
the authors; every curated value can be traced to its raw source through the dataset's `sourcedata/`.

## Run

```bash
uv venv -p 3.12 .venv && uv pip sync -p .venv/bin/python requirements.lock
PY=.venv/bin/python SYNAPSE_RAW_ROOT=/path/to/raw SYNAPSE_BIDS_ROOT=/path/to/SYNAPSE_dataset ./run_curation.sh
# ~10 min; the first run reads ~35 GB of EEG and caches it in SYNAPSE_WORK_ROOT
```

Requirements: Python 3.12, `ffprobe` (thermal AVI metadata), `deno` (BIDS validator 3.0.2). Paths are set in
`config/paths.toml` and can be overridden with `SYNAPSE_RAW_ROOT`, `SYNAPSE_BIDS_ROOT`, `SYNAPSE_WORK_ROOT`,
`SYNAPSE_PLATFORM_ROOT`.

## Inputs that are not raw data (all under `config/`)

| File | Content |
|---|---|
| `decisions.yaml` | **Register of every non-measured fact and every correction** (D01–D28) with its basis (investigator, data-derived, literature, convention, unavailable) and evidence. Copied into the dataset (`sourcedata/decisions.yaml`). |
| `sessions.tsv` | Transcription of the operator log (one row per participant-session), with corrections labelled by decision ID |
| `eeg_assignment.tsv` | Which raw EEG file content (and time window) belongs to which participant-session, with evidence |
| `visual_item_review.tsv` | Key-validity rating of the AI-generated visual-task images (one rater) |
| `thermal_method_README.md` | Method description of the thermal extraction |
| `LICENSE_dataset.txt` | CC BY 4.0 legal code for the dataset |

## Pipeline

| Step | Script | Output |
|---|---|---|
| 01 | `s01_eeg_inventory.py` | SHA-256 of every EEG file; lossless de-duplication cache; stream clustering |
| 02 | `s02_build_core.py` | trial tables (`beh/`), questionnaires (`phenotype/`), participants/sessions, timeline, room monitor, stimuli |
| 03 | `s03_build_eeg.py` | BrainVision int16 (bit-exact), channels/events/sidecars, host timestamps, EEG QC |
| 04 | `s04_thermal.py` | thermal snapshot extraction (method of release 1.0.0, outputs reproduced byte-identically) |
| 05 | `s05_build_physio.py` | wrist PPG/acceleration/skin temperature (BIDS physio), vendor IBI/HR/HRV derivatives |
| 06 | `s06_build_thermal.py` | seat→participant link, per-session facial temperature series, baseline usability |
| 07 | `s07_clock_checks.py` | post-hoc clock-consistency bounds between devices |
| 08–09 | `s08_finalize.py`, `s09_docs.py` | dataset_description, scans, raw-file inventory, README, CHANGES, data dictionary |
| tests | `tests/test_curation.py` | row accounting, EEG bit-exactness, physio integrity, no direct identifiers, decision IDs |

## Principles

* The raw archive is never modified (two additions were made to it on 2026-10-27 and are listed in its manifest:
  `stimuli/soundscapes/` — the original WAV files previously held only in a derived folder — and
  `metadata/deployed_google_forms/` — the deployed questionnaire definitions retrieved from the live forms).
* No value that was not measured is created. Missing data are documented (`n/a`), never imputed.
* Every curated row/recording traces to its raw source (file + line, or file + SHA-256 + time window).
* Identifiable material (faces, names, free-text health, host names) is never written to the dataset;
  `tests/test_curation.py::test_no_direct_identifiers` enforces this.

Licence: MIT (code). Author: Puneet Tomar.
