<p align="center"><img src="assets/SYNAPSE_logo.svg" alt="SYNAPSE study logo" width="220"></p>

# SYNAPSE study

*SYnergistic Neurophysiological Assessment of Perception in Sensory Environments*

SYNAPSE is a laboratory study of how the indoor environment affects people at work. Ten adults completed computerised
cognitive tasks in 20 sessions in a simulated office at the Technical University of Denmark (June 2025) under warm or
cool LED lighting and two urban soundscapes, while their cognitive performance, four-channel EEG, wrist physiology,
facial thermography, perceived environment and the room conditions were recorded.

This repository is the home of the study. It holds the study's code, and its publications and further materials are
added here as they become available.

## Resources

| Resource | Link |
|---|---|
| Dataset (BIDS 1.10, CC BY 4.0) | https://doi.org/10.5281/zenodo.23458777 |
| Experimental protocol | https://doi.org/10.17504/protocols.io.m7vznm25rvx1/v1 |
| Experiment platform (task and questionnaire software) | https://github.com/tomarp/synapse-platform, https://doi.org/10.5281/zenodo.23768060 |

## Publications

| Publication | Status |
|---|---|
| Tomar, P., Dave, T., Wargocki, P. & Pisello, A. L. *Multimodal dataset of cognitive performance, physiology and indoor environment under lighting and soundscapes.* Data Descriptor. | submitted to *Scientific Data* |

## Contents

| Folder | Purpose | Needs |
|---|---|---|
| [`curation/`](curation/README.md) | Curation pipeline: raw recordings → BIDS dataset (behaviour, EEG, wrist physiology, facial thermography, questionnaires, room data, session timeline, quality control), with integrity tests and the register of curation decisions | the raw recordings (restricted) |
| `data_descriptor/` | Technical validation: the figures, tables and numbers of the Data Descriptor | the published dataset |
| `assets/` | Study logo (SVG and PNG) | |

Every number quoted in the Data Descriptor is computed by this code and written as a LaTeX macro, so that the text
contains no hand-typed values.

## Code

### Setup

Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r data_descriptor/requirements.txt
```

The curation pipeline has its own locked environment (`curation/requirements.lock`) and needs `ffprobe` and `deno`.

Paths default to the authors' project layout (`output/{code,dataset,platform,deliverables}`) and can be set
with environment variables:

| Variable | Meaning | Used by |
|---|---|---|
| `SYNAPSE_BIDS_ROOT` | unzipped dataset (`SYNAPSE_dataset/`) | `curation/`, `data_descriptor/` |
| `SYNAPSE_DD_OUT` | output folder of the Data Descriptor figures and numbers | `data_descriptor/` |
| `SYNAPSE_RAW_ROOT`, `SYNAPSE_WORK_ROOT` | raw recordings, cache | `curation/` |
| `SYNAPSE_PLATFORM_ROOT` | clone of `synapse-platform` (task code and stimulus images) | `curation/` |

### Reproducing the technical validation

Download `SYNAPSE_dataset.zip` from the dataset record and unzip it.

```bash
cd data_descriptor
SYNAPSE_BIDS_ROOT=/path/to/SYNAPSE_dataset SYNAPSE_DD_OUT=/path/to/output ../.venv/bin/python run.py
```

Figures are written to `<SYNAPSE_DD_OUT>/figures/` (PDF and PNG) and tables and numbers to
`<SYNAPSE_DD_OUT>/generated/`. The EEG epoch quality is computed from the BrainVision files on the first run and cached;
`--recompute-eeg` forces a new computation, and `--only fig1 fig5 tables` restricts the run to selected outputs. The
modules in `data_descriptor/ddval/` read every stream of the dataset and can serve as examples of loading and aligning
the data:

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

### Checking the dataset

`curation/tests/` checks the integrity of the published dataset (complete layout, well-formed tables, physiological
files matching their sidecars, valid decision identifiers):

```bash
.venv/bin/pip install pytest
SYNAPSE_BIDS_ROOT=/path/to/SYNAPSE_dataset .venv/bin/python -m pytest curation/tests
```

The tests that compare the dataset with the raw recordings are skipped when these are not available.

### Rebuilding the dataset

The curation pipeline documents every step from the raw recordings to the published dataset; see
[`curation/README.md`](curation/README.md) for the steps, the configuration files and the principles applied. Every
non-measured value and every correction is recorded with its basis in
[`curation/config/decisions.yaml`](curation/config/decisions.yaml), which is also distributed with the dataset.

## Citation

If you use material from this repository, please cite the Data Descriptor and the dataset (see `CITATION.cff`).

## Licence

Code: MIT (see `LICENSE`). Study logo (`assets/`): CC BY 4.0. The dataset is licensed separately under CC BY 4.0.

## Funding

This work was funded by the European Union's Horizon Europe research and innovation programme under the Marie
Skłodowska-Curie Doctoral Network MuSIC (grant agreement No. 101073357).
