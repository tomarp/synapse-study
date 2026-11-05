"""Invariants of the curated SYNAPSE dataset (run after run_curation.sh)."""
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synapse_curation import paths  # noqa: E402
from synapse_curation.common import decisions  # noqa: E402

B = paths.BIDS
SUBS = [f"sub-P{i:02d}" for i in range(1, 11)]
# tests that compare the dataset with the raw recordings (restricted) run only where these are available
needs_raw = pytest.mark.skipif(not paths.RAW.is_dir(), reason="raw recordings not available")
needs_cache = pytest.mark.skipif(not (paths.WORK / "eeg_inventory.tsv").is_file(), reason="EEG inventory cache not available")


def test_layout_complete():
    for s in SUBS:
        for ses in ("ses-A", "ses-B"):
            assert len(list((B / s / ses / "beh").glob("*_beh.tsv"))) == 8, (s, ses)
            assert (B / s / ses / "eeg" / f"{s}_{ses}_task-cogbattery_eeg.vhdr").exists()
        assert (B / s / f"{s}_sessions.tsv").exists()


@needs_raw
def test_every_raw_task_row_accounted():
    qc = pd.read_csv(B / "derivatives/quality_control/behaviour_exclusion_counts.tsv", sep="\t", keep_default_na=False)
    raw = {"nback": "nBackTask", "stroop": "StroopTask", "visual": "VisualTask", "flanker": "FlankerTask"}
    for fam, stem in raw.items():
        n_raw = len(pd.read_csv(paths.RAW / "dataforms" / f"{stem}-Responses.csv"))
        assert qc.loc[qc.task_family == fam, "rows"].sum() == n_raw
        n_out = sum(len(pd.read_csv(f, sep="\t")) for f in B.glob(f"sub-*/ses-*/beh/*_task-*_beh.tsv")
                    if fam in f.name.replace("emotional", ""))
        n_keep = qc.loc[(qc.task_family == fam) & qc.exclusion.isin(["analysable", "superseded_run"]), "rows"].sum()
        assert n_out == n_keep, fam


@needs_raw
@needs_cache
def test_eeg_bit_exact():
    """BrainVision values equal the de-duplicated raw samples; host-time file has one row per sample."""
    from synapse_curation import eeg
    recs = eeg.assemble(pd.read_csv(paths.WORK / "eeg_inventory.tsv", sep="\t"),
                        pd.read_csv(paths.CONFIG / "eeg_assignment.tsv", sep="\t", dtype=str, keep_default_na=False))
    for v in sorted(B.glob("sub-*/ses-*/eeg/*_eeg.vhdr")):
        res = float(re.search(r"Ch1=TP9,[^,]*,([0-9.]+)", v.read_text()).group(1))
        x = np.fromfile(v.with_suffix(".eeg"), dtype="<i2").reshape(-1, 4) * res
        sub, ses = v.name.split("_")[:2]
        y = recs[(sub, ses)]["data"][:, 1:]
        assert np.array_equal(x, y), v.name
        with gzip.open(B / "sourcedata/eeg_host_timestamps" / sub / ses / v.name.replace("_eeg.vhdr", "_hosttime.tsv.gz"), "rt") as fh:
            assert sum(1 for _ in fh) == len(x), v.name


def test_physio_matches_sidecar():
    for f in sorted(B.glob("sub-*/ses-*/beh/*_physio.tsv.gz")):
        side = json.loads(Path(str(f).replace(".tsv.gz", ".json")).read_text())
        with gzip.open(f, "rt") as fh:
            first = fh.readline().rstrip("\n").split("\t")
        assert len(first) == len(side["Columns"]), f.name


@needs_raw
def test_no_direct_identifiers():
    bg = pd.read_excel(paths.RAW / "ParticipantInformation.xlsx", dtype=str)
    tokens = set()
    for v in bg.iloc[:, 0].dropna():
        tokens |= {t for t in re.split(r"\s+", v.strip()) if len(t) >= 4}
    tokens |= {v.strip() for v in bg.iloc[:, 30].dropna() if "@" in str(v)}
    tokens |= {d.name for d in paths.RAW.rglob("P[0-9][0-9]-*") if d.is_dir() and re.fullmatch(r"P\d\d-[A-Za-z]+", d.name)}
    pat = re.compile("|".join(re.escape(t) for t in tokens), re.I)
    hits = []
    for f in B.rglob("*"):
        if f.is_file() and f.suffix in (".tsv", ".json", ".md", ".yaml", ".txt", ".vhdr", ".vmrk") or f.name in ("README.md", "CHANGES"):
            if pat.search(f.read_text(errors="ignore")):
                hits.append(f.relative_to(B).as_posix())
    assert not hits, f"identifier tokens found in {len(hits)} files"


def test_decision_ids_exist():
    ids = set(decisions())
    text = " ".join(f.read_text(errors="ignore") for f in B.rglob("*.json"))
    for ref in set(re.findall(r"\bD(\d\d)\b", text)):
        assert any(k.startswith(f"D{ref}_") for k in ids), ref


def test_tables_well_formed():
    """Every TSV is rectangular, unquoted, uses n/a for missing values and has a sidecar describing its columns."""
    problems = []
    for f in B.rglob("*.tsv"):
        lines = f.read_text(encoding="utf-8").rstrip("\n").split("\n")
        header = lines[0].split("\t")
        for i, line in enumerate(lines[1:], 2):
            cells = line.split("\t")
            if len(cells) != len(header) or "" in cells or any(c.startswith('"') for c in cells):
                problems.append(f"{f.relative_to(B)}:{i}")
                break
        side = f.with_suffix(".json")
        if f.name not in ("DATA_DICTIONARY.tsv",) and not f.name.endswith(("_channels.tsv", "_scans.tsv")):
            if not side.exists():
                problems.append(f"{f.relative_to(B)}: no sidecar")
            else:
                described = json.loads(side.read_text())
                missing = [c for c in header if c not in described and c not in ("participant_id", "session_id", "onset", "duration")]
                if missing:
                    problems.append(f"{f.relative_to(B)}: undescribed {missing[:3]}")
    assert not problems, problems[:10]
