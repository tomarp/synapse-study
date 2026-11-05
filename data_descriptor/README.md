# Technical validation — SYNAPSE Data Descriptor

Code behind the figures, tables and numbers of the Data Descriptor *Multimodal dataset of cognitive performance,
physiology and indoor environment under lighting and soundscapes* (Tomar, Dave, Wargocki & Pisello; submitted to
*Scientific Data*). It runs on the published dataset only (https://doi.org/10.5281/zenodo.23458777).

Every number quoted in the paper is computed here and written as a LaTeX macro, so that the text contains no hand-typed
values.

## Setup

Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

Download `SYNAPSE_dataset.zip` from the dataset record and unzip it, then:

```bash
SYNAPSE_BIDS_ROOT=/path/to/SYNAPSE_dataset SYNAPSE_DD_OUT=/path/to/output .venv/bin/python run.py
```

| Variable | Meaning |
|---|---|
| `SYNAPSE_BIDS_ROOT` | unzipped dataset (`SYNAPSE_dataset/`) |
| `SYNAPSE_DD_OUT` | output folder |

Without these variables, paths default to the authors' project layout (`output/{code,dataset,deliverables}`).

Figures are written to `<SYNAPSE_DD_OUT>/figures/` (PDF and PNG), and tables and numbers to
`<SYNAPSE_DD_OUT>/generated/`. The EEG epoch quality is computed from the BrainVision files on the first run and cached;
`--recompute-eeg` forces a new computation, and `--only fig1 fig5 tables` restricts the run to selected outputs.

## Modules

The modules in `ddval/` read every stream of the dataset and can serve as examples of loading and aligning the data.

| Module | Output |
|---|---|
| `fig_design.py` | Fig. 1, study design |
| `fig_example.py` | Fig. 2, all streams of one session on the common timeline |
| `fig_quality.py` | Fig. 3, completeness and quality per session |
| `fig_behaviour.py` | Fig. 4, behavioural data |
| `eeg.py`, `fig_eeg.py` | Fig. 5, EEG epoch quality and baseline spectra |
| `physio.py`, `fig_physio.py` | Fig. 6, wrist and thermal validation (PPG-derived versus device heart rate; reconstructed versus spot temperature) |
| `fig_environment.py` | Fig. 7, room conditions and perceived-environment ratings |
| `tables.py` | Tables 1–3 |
| `common.py` | paths, data loading, figure style, number registry |

Licence: MIT (see the repository's `LICENSE`).
