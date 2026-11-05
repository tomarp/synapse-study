#!/usr/bin/env bash
# Build the curated SYNAPSE dataset (BIDS) from the raw recordings.
#   SYNAPSE_RAW_ROOT / SYNAPSE_BIDS_ROOT / SYNAPSE_WORK_ROOT / SYNAPSE_PLATFORM_ROOT override config/paths.toml
#   PY  python of the environment (default SYNAPSE/.venv; create with: uv venv -p 3.12 ../../../.venv && uv pip sync -p ../../../.venv/bin/python requirements.lock)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../../.venv/bin/python}
export PYTHONDONTWRITEBYTECODE=1                    # no __pycache__ folders in output/
BIDS=$($PY -c "import sys; sys.path.insert(0,'src'); from synapse_curation import paths; print(paths.BIDS)")
WORK=$($PY -c "import sys; sys.path.insert(0,'src'); from synapse_curation import paths; print(paths.WORK)")
step() { echo; echo "=== $1"; }
step "00 clean output";                                   rm -rf "${BIDS:?}"
step "01 EEG inventory + de-duplication (~35 GB, cached)"; $PY -I -B scripts/s01_eeg_inventory.py > /dev/null
step "02 behaviour, questionnaires, participants, room, stimuli"; $PY -I -B scripts/s02_build_core.py
step "03 EEG -> BrainVision";                             $PY -I -B scripts/s03_build_eeg.py
step "04 thermal extraction";                             SYNAPSE_THERMAL_TMP=${SYNAPSE_THERMAL_TMP:-$WORK/thermal_cache} $PY -I -B scripts/s04_thermal.py --workers 8 --reuse
step "05 wrist physiology";                               $PY -I -B scripts/s05_build_physio.py > /dev/null
step "06 thermal derivatives";                            $PY -I -B scripts/s06_build_thermal.py > /dev/null
step "07 clock checks";                                   $PY -I -B scripts/s07_clock_checks.py > /dev/null
step "08 dataset metadata, README, dictionary"; $PY -I -B scripts/s08_finalize.py
step "09 tests";                                          $PY -m pytest -q -p no:cacheprovider tests
step "10 BIDS validator";                                 deno run -A jsr:@bids/validator@3.0.2 "$BIDS" --json > "$WORK/bids_validation.json" || true
echo; echo "Done: $BIDS"
