# Results paper — analysis

Analysis of the published SYNAPSE dataset (https://doi.org/10.5281/zenodo.23458777) for the paper on correlated colour
temperature and urban soundscapes (*Building and Environment*, in preparation).

- [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) — the analysis plan, registered (commit `7b7c552`) before any comparison
  between conditions was made.
- `rp/` — the analysis: `data.py` (analysis tables from the dataset), `models.py` (Bayesian models in PyMC),
  `analyses.py` (registered, sensitivity and exploratory analyses), `figures.py` (figures and tables),
  `common.py` (paths, coding, blinding, number registry).
- `run.py` — runs everything.

## Run

Python 3.12.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
SYNAPSE_BIDS_ROOT=/path/to/SYNAPSE_dataset SYNAPSE_RP_OUT=/path/to/output .venv/bin/python -I -B run.py
```

The analysis is **blind by default**: condition labels are scrambled (ANALYSIS_PLAN.md, section 8) and the outputs go to
`<SYNAPSE_RP_OUT>/blind/`. `SYNAPSE_UNBLIND=1` runs it on the true labels. A full run fits about 120 models
(roughly 30 min on a laptop); fitted posteriors are cached in `<out>/cache/`. Outputs: `figures/` (PDF, PNG) and
`generated/` (`results.tsv` with every estimand, `convergence.tsv`, `deviations.tsv`, LaTeX tables and the number
macros used in the manuscript).

Licence: MIT (see the repository's `LICENSE`).
