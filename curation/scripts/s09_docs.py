"""Step 09: README.md (numbers computed from the curated files), CHANGES and DATA_DICTIONARY.tsv."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synapse_curation import __version__, paths  # noqa: E402
from synapse_curation.common import decisions, write_tsv  # noqa: E402

B = paths.BIDS
QC = B / "derivatives" / "quality_control"


def stats() -> dict:
    beh = pd.concat([pd.read_csv(f, sep="\t") for f in B.glob("sub-*/ses-*/beh/*_beh.tsv")], ignore_index=True)
    eq = pd.read_csv(QC / "eeg_quality.tsv", sep="\t")
    wc = pd.read_csv(QC / "wearable_coverage.tsv", sep="\t")
    th = pd.read_csv(B / "derivatives" / "facial_temperature" / "thermal_coverage.tsv", sep="\t")
    room = pd.read_csv(B / "environment" / "room_monitor_1min.tsv", sep="\t")
    summ = pd.read_csv(B / "background_questionnaire" / "sample_summary.tsv", sep="\t").set_index("statistic")["value"]
    links = pd.read_csv(QC / "questionnaire_linkage.tsv", sep="\t")
    snd = json.loads((B / "stimuli" / "soundscapes.json").read_text())["files"]
    return {
        "n_trials": int((~beh.superseded_run).sum()), "n_superseded": int(beh.superseded_run.sum()),
        "eeg_hours": round(sum(eq.n_samples) / 256 / 3600, 1),
        "eeg_partial": ", ".join(f"{r.participant_id}/{r.session_id} ({r.session_coverage_fraction:.0%})"
                                 for r in eq.itertuples() if r.session_coverage_fraction < 0.9),
        "wear_sessions": int((wc.status == "available").sum()),
        "th_unusable_baseline": ", ".join(f"{r.participant_id}/{r.session_id}" for r in th.itertuples() if not r.baseline_usable),
        "room_records": len(room),
        "age": f"{summ['age_mean_years']:.1f} ± {summ['age_sd_years']:.1f} years, range {int(summ['age_min_years'])}–{int(summ['age_max_years'])}",
        "sex": f"{int(summ['female'])} female, {int(summ['male'])} male",
        "links_weak": ", ".join(f"{r.questionnaire} {r.participant_id}/{r.session_id}" for r in links.itertuples() if r.link_quality in ("weak", "moderate")),
        "snd_rel": next(s["level_A_relative_to_babble_dB"] for s in snd if "traffic" in s["file"].lower()),
        "availability": availability_table(), "eeg_saturation": eeg_saturation(), "room_gaps": room_gaps(),
        "task_rows": task_rows(), **response_strategies(),
    }


def task_rows() -> str:
    from synapse_curation.columns import TASK_INFO
    rounds = {"nbackletter": "1", "stroopword": "1", "visualcount": "1", "flankerarrow": "1", "stroopemotionalword": "1",
              "nbackdigit": "2", "stroopemoji": "2", "visualshape": "2", "flankerdigit": "2", "stroopemotionalimage": "2"}
    early = ("stroopemotionalword", "stroopemotionalimage")
    return "\n".join(f"| `{k}` | {rounds[k]} | **{name}**: {re.sub(r' *\(D\d\d\)', '', desc)} |"
                     for k, (name, desc) in sorted(TASK_INFO.items(), key=lambda kv: (kv[0] in early, rounds[kv[0]])))


def availability_table() -> str:
    """One markdown row per session: what is available and how complete it is."""
    eq = pd.read_csv(QC / "eeg_quality.tsv", sep="\t").set_index(["participant_id", "session_id"])
    wc = pd.read_csv(QC / "wearable_coverage.tsv", sep="\t").set_index(["participant_id", "session_id"])
    th = pd.read_csv(B / "derivatives" / "facial_temperature" / "thermal_coverage.tsv", sep="\t").set_index(["participant_id", "session_id"])
    links = pd.read_csv(QC / "questionnaire_linkage.tsv", sep="\t")
    weak = {(r.participant_id, r.session_id, r.questionnaire) for r in links.itertuples() if r.link_quality in ("weak", "moderate")}
    rows = ["| Session | Trials | EEG coverage | Wrist | Facial temperature (valid snapshots) | T0 | T1 | T2 | T3 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for sub in sorted(p.name for p in B.glob("sub-*")):
        for ses in ("ses-A", "ses-B"):
            beh = pd.concat([pd.read_csv(f, sep="\t", usecols=["superseded_run"]) for f in (B / sub / ses / "beh").glob("*_beh.tsv")])
            k = (sub, ses)
            t = th.loc[k]
            face = f"{t.n_valid / t.n_snapshots:.0%}" + ("" if t.baseline_usable else ", no baseline")
            wrist = "yes" if wc.loc[k, "status"] == "available" else "missing"
            q = ["yes" + (" (weak link)" if (sub, ses, n) in weak else "") for n in ("T0", "T1", "T2", "T3")]
            rows.append(f"| {sub}/{ses} | {int((~beh.superseded_run).sum())} | {eq.loc[k, 'session_coverage_fraction']:.0%} | {wrist} | {face} | " + " | ".join(q) + " |")
    return "\n".join(rows)


def response_strategies() -> dict:
    """Sessions in which a task was performed with a systematically different response rule (documented, not altered)."""
    def load(task):
        out = []
        for f in B.glob(f"sub-*/ses-*/beh/*_task-{task}_beh.tsv"):
            d = pd.read_csv(f, sep="\t")
            d = d[~d.superseded_run.astype(bool)]
            d["key"] = "/".join(f.name.split("_")[:2]).replace("_", "/")
            out.append(d)
        return pd.concat(out)
    e = load("stroopemoji")
    e["word_response"] = e.response == e.word
    g = e.groupby("key").agg(word=("word_response", "mean"), acc=("correct", "mean"))
    g = g[(g.word >= 0.2) & ((g.word + g.acc).round(3) == 1)].sort_index()
    f = load("flankerdigit")
    f["opposite"] = f.response != f.expected_response
    h = f.groupby("key").opposite.mean()
    h = h[h >= 0.5].sort_index()
    return {"emoji_word_sessions": ", ".join(g.index), "emoji_word_min": f"{g.word.min():.0%}", "emoji_word_max": f"{g.word.max():.0%}",
            "flanker_reversed": ", ".join(f"{k} ({v:.0%} of responses)" for k, v in h.items())}


def eeg_saturation() -> str:
    eq = pd.read_csv(QC / "eeg_quality.tsv", sep="\t")
    out = []
    for r in eq.itertuples():
        bad = [f"{c} {getattr(r, f'rail_fraction_{c}'):.0%}" for c in ("TP9", "AF7", "AF8", "TP10") if getattr(r, f"rail_fraction_{c}") >= 0.10]
        if bad:
            out.append(f"{r.participant_id}/{r.session_id} ({', '.join(bad)})")
    return "; ".join(out)


def room_gaps() -> str:
    room = pd.read_csv(B / "environment" / "room_monitor_1min.tsv", sep="\t")
    t = pd.to_datetime(room["time"])
    gap = t.diff().dt.total_seconds() / 60
    big = gap[gap > 10]
    return "; ".join(f"{t[i - 1]:%d %B %H:%M}–{t[i]:%H:%M}" for i in big.index)


README = """# SYNAPSE dataset

**{subtitle}**

Version {version} · BIDS {bids} · CC BY 4.0 · [doi:{doi}](https://doi.org/{doi})

Tomar P., Dave T., Wargocki P., Pisello A.L. (University of Perugia; Technical University of Denmark)

## Overview

Ten adults worked on a cognitive test battery in a simulated office at the Technical University of Denmark (24–26 June 2025). Each participant completed two sessions on the same day, one under **warm** (2014 K) and one under **cool** (6250 K) LED lighting. Within each session, two rounds of tasks were performed while listening to two soundscapes over headphones: **cafeteria babble** and **road traffic**. Brain activity (EEG), wrist physiology, facial thermography and room conditions were recorded throughout, and participants rated their state and the indoor environment in questionnaires.

| | |
|---|---|
| Participants | 10 ({sex}, 1 not reported; age {age}) |
| Sessions | 20 (2 per participant: `ses-A`, `ses-B`) |
| Factors | lighting (warm, cool) × soundscape (babble, traffic), both within participants |
| Cognitive tasks | 4 per round, 8 per session; {n_trials} trials |
| EEG | Muse S, 4 channels (TP9, AF7, AF8, TP10), 256 Hz; 20 sessions, {eeg_hours} h |
| Wrist | ActiGraph LEAP: PPG 25 Hz, acceleration 32 Hz, skin temperature 1/12 Hz; {wear_sessions} sessions |
| Face | FLIR AX8 thermal camera; relative head temperature every ~10 s, derived from the thermal video; 20 sessions |
| Room | ATMOCUBE: air temperature, humidity, CO₂, particles, VOC, NOx, sound level, illuminance; 1-min records, 20–27 June 2025 |
| Questionnaires | T0 before the session; T1 after the baseline; T2 and T3 after each task round; background questionnaire |
| Size | {size} in {n_files} files |

## Study design

Each session lasted about 40 min (range 34–55 min). A participant's two sessions were separated by a 10–15 min break.

| Phase | Content | Soundscape |
|---|---|---|
| T0 | Pre-session questionnaire: alertness, energy, stress, sleep, caffeine, food, clothing, initial comfort | none |
| Baseline | 5 min, eyes open | none |
| T1 | Indoor-environment questionnaire: thermal, air quality, lighting, acoustics | none |
| Round 1 | n-back (letters), word Stroop, visual counting, arrow flanker | soundscape 1 |
| T2 | Task experience, affect and indoor environment | none |
| Round 2 | n-back (digits), emoji Stroop, shape comparison, digit flanker | soundscape 2 |
| T3 | Task experience, affect and indoor environment | none |

- Lighting order: cool first for 6 participants, warm first for 4. Soundscape order was reversed between a participant's two sessions.
- Participants were tested in pairs at two workstations (sub-P01 and sub-P02, sub-P03 and sub-P04, sub-P05 and sub-P06, sub-P08 and sub-P09) or alone (sub-P07, sub-P10).
- Each round always used the same task variants, so task variant is confounded with round, and lighting with session order.
- The colour theme of the task screens followed the lighting (cool: dark; warm: light).

## Tasks

Each task table (`sub-*/ses-*/beh/*_task-<task>_beh.tsv`) has one row per trial.

| Task label | Round | Task |
|---|---|---|
{task_rows}

The number of trials varies between sessions because the task settings changed during data collection (see *Missing data and data quality*); `derivatives/quality_control/task_configuration.tsv` gives the settings and trial counts of every session. Each task ran until its trials were completed or the 5-min page timer allowed the participant to continue.

## Dataset schema

```
participants.tsv                         1 row per participant              key: participant_id
sub-<ID>/sub-<ID>_sessions.tsv           1 row per session                  key: session_id; lighting, soundscapes, times, task settings
sub-<ID>/ses-<A|B>/
    beh/*_task-<task>_beh.tsv            1 row per trial                    round, soundscape, stimulus, response, rt_ms, correct
    beh/*_recording-<ppg|acc|skintemp>_physio.tsv.gz   continuous wrist signals (no header; columns in the .json)
    eeg/*_eeg.vhdr|.vmrk|.eeg            continuous EEG (BrainVision)
    eeg/*_events.tsv                     1 row per phase, task block or recording gap (onset in s from recording start)
    eeg/*_channels.tsv                   1 row per channel, with signal-quality notes
phenotype/<T0|T1|T2|T3>_*_session<A|B>.tsv   1 row per participant      keys: participant_id, session_id
derivatives/session_timeline/timeline.tsv    1 row per phase or task block  keys: participant_id, session_id
derivatives/facial_temperature/sub-<ID>/ses-<A|B>/*_desc-facetemp_timeseries.tsv   1 row per snapshot (~10 s)
derivatives/wristband_vendor_metrics/sub-<ID>/ses-<A|B>/*_desc-<ibi|heartrate|hrv>_timeseries.tsv
environment/room_monitor_1min.tsv        1 row per minute                   key: time (join to sessions by time)
stimuli/visual_items.tsv                 1 row per task image               key: image (= image in the visual task tables)
background_questionnaire/                1 row per respondent               not linked to participant_id
```

All tables of one participant and session share `participant_id` (`sub-P01` … `sub-P10`) and `session_id` (`ses-A`, `ses-B`); for files inside `sub-*/ses-*/` both are given by the folder. The lighting condition of a session is in `sub-*_sessions.tsv` (`light`); the soundscape of each trial is in the task tables (`soundscape`).

## Files

Every table has a JSON file of the same name that defines each column (description, units, levels); `DATA_DICTIONARY.tsv` lists all {n_columns} columns of all tables in one place.

| Path | Content |
|---|---|
| `dataset_description.json`, `participants.tsv`, `CITATION.cff`, `CHANGES`, `LICENSE` | Dataset metadata, participants, citation, version history, licence |
| `sub-*/sub-*_sessions.tsv` | Lighting (`light`, `cct_setpoint_K`), soundscape per round, session start and end, task settings (`cfg_*`), operator notes |
| `sub-*/ses-*/beh/` | Task tables (see *Tasks*) and wrist signals: `ppg_green_ch0/1` and `ppg_ambient_ch0/1` (ADC counts), `acc_x/y/z` (g), `skin_temperature` (°C) |
| `sub-*/ses-*/eeg/` | EEG in µV (BrainVision), events and channels |
| `phenotype/` | Questionnaires T0–T3, one table per questionnaire and session. Each item has the answer as given and a numeric code (`*_code`); the JSON gives the question wording and the codes |
| `background_questionnaire/` | Background questionnaire (9 of 10 returned) in generalised form, and a sample summary |
| `environment/room_monitor_1min.tsv` | `air_temperature`, `relative_humidity`, `absolute_humidity`, `air_pressure`, `co2`, `pm1`–`pm10`, `tvoc`, `voc_index`, `nox_index`, `sound_level`, `illuminance` |
| `stimuli/` | `visual_items.tsv` (answer key and its validity for each image), `image_provenance.tsv` (generator and checksum of each image), `soundscapes.json` (level and spectrum of both recordings) |
| `derivatives/session_timeline/` | Start and end of every phase and task block |
| `derivatives/facial_temperature/` | Relative head-region temperature per session (`T_median`, `T_median_minus_bg`, `valid`), seat-to-participant links (`thermal_coverage.tsv`), method and validation (`quality_control/README.md`) |
| `derivatives/wristband_vendor_metrics/` | Inter-beat intervals, 1-min heart rate and 300-s HRV computed by the device software |
| `derivatives/quality_control/` | EEG coverage, gaps and saturation; wrist coverage; clock checks; questionnaire linkage; task settings; excluded rows |
| `sourcedata/` | How the dataset was built from the raw recordings: raw-file inventory with checksums, curation decisions (D01–D29, cited in the JSON files), EEG file assignment, operator-log times, questionnaire item definitions, per-sample EEG timestamps |

## Conventions

- Tables are tab-separated UTF-8 text with a header row; missing values are `n/a`; logical values are `true`/`false`.
- Times are local time (CEST, UTC+02:00) in ISO 8601 format; `*_unix` columns give the same moment in seconds since 1970-01-01 UTC.
- EEG event onsets and wrist-signal samples are relative to the start of their file; the start time is in the JSON file (wrist: `SYNAPSE_StartTimeLocal`; sample *i* is at start + *i* / sampling rate) and in `*_scans.tsv`.
- Questionnaire scales: T2/T3 task and affect items 9-point (focus 7-point); indoor-environment items 5-point; preferences coded −1 / 0 / +1.

## Data availability

{availability}

Trials exclude earlier attempts of restarted task blocks. EEG coverage is the fraction of the session covered by the recording. A *weak link* means that the anonymous questionnaire was assigned to the participant by submission time with less certainty (`derivatives/quality_control/questionnaire_linkage.tsv`).

## Missing data and data quality

Missing values are never imputed.

- **Wrist data** are missing for sub-P06 (device worn, data not retrieved).
- **EEG** recordings are incomplete where the table above shows less than 100 %, because of signal dropouts; all gaps are listed in `derivatives/quality_control/eeg_gaps.tsv` and marked as `boundary` events. Channels saturated (±1000 µV) for at least 10 % of the session: {eeg_saturation}. Saturation per channel is given in `eeg_quality.tsv` and in each `*_channels.tsv`. TP9 and TP10 were under the headphone ear cups.
- **Facial temperature** of the baseline is missing for {th_unusable_baseline} (camera failure).
- **Room monitor** has no records during {room_gaps} (before the sessions of that day); `tvoc`, `voc_index` and `nox_index` are missing in a few records.
- **Questionnaires:** all T0–T3 forms were returned. `clothing_clo_est` is missing when the clothing checklist was incomplete (sub-P10).
- **Background questionnaire:** 9 of 10 returned; it cannot be linked to participant codes.
- **Task settings** changed during data collection: 30, 60 or 20 n-back trials per level and 15, 60 or 20 flanker trials per variant (`cfg_*` columns). Sessions sub-P01/ses-A and sub-P02/ses-A used an early configuration with emotional Stroop tasks and fixed n-back lists; 20 n-back responses of that configuration were re-scored because of an answer-key error.
- **Visual-task answer keys:** the images are AI-generated, and some counting images have no well-defined count; two shape images (image3_same, image12_dif) are identical with contradictory keys. `key_validity` rates each key, and `count_log_ratio_consensus` scores counting responses against the participants' consensus.
- **Stroop:** all word-Stroop trials are incongruent; for emoji Stroop the emoji shown was not recorded.
- **Response strategies:** in {emoji_word_sessions}, {emoji_word_min}–{emoji_word_max} of emoji-Stroop responses named the word instead of the ink colour, and every error in these sessions was of this kind. In {flanker_reversed}, digit-flanker responses gave the opposite parity, consistent with a reversed response mapping. These responses are scored as given (`correct = 0`) and are not altered.
- **Sound and light levels** were not measured at the participant. The babble soundscape through the same headphones measured 53.6 ± 2.5 dB at the ear cup in a previous study (Dave et al., 2026); the traffic recording is {snd_rel:+.1f} dB(A) louder. Colour temperatures are luminaire setpoints. The room monitor's `sound_level` does not include headphone playback and its `illuminance` is uncalibrated.
- **Clocks:** each device used its own clock. They agree within about 1 min (task computers within 1 s of server time; `derivatives/quality_control/clock_checks.tsv`), and no correction is applied. Task block onsets are accurate to about ±2 s.

## Using the data

- **Trials:** use rows with `superseded_run = false` and score with `correct` (re-scored from the stimulus; the platform's own score is kept as `correct_recorded`). Implausible response times are marked in `rt_flag` (`ok`, `fast_lt150ms`, `slow_gt10s`, `slow_gt120s`), not removed.
- **Conditions:** take lighting from `sub-*_sessions.tsv` and soundscape from the task tables; `round` separates the two task rounds.
- **Time alignment:** use `derivatives/session_timeline/timeline.tsv` to cut any signal into baseline, rounds and task blocks.
- **EEG** is unfiltered and not re-referenced (reference Fpz), at the device resolution of 0.49 µV; per-sample computer timestamps are in `sourcedata/eeg_host_timestamps/`.
- **Facial temperature** values are relative: use changes within a session, preferably `T_median` or `T_median_minus_bg`, and only rows with `valid = true`.

## Not included

- Raw thermal video and images (they show faces), the free-text operator log, health-screening answers and the identifiable background questionnaire. These are available from the authors under a data-use agreement, within the terms of the participants' consent.
- The task images, which are distributed with the SYNAPSE experiment platform (doi:10.5281/zenodo.23768059) and identified here by file name and checksum.
- The soundscape audio, whose source was not recorded and which therefore cannot be redistributed; `stimuli/soundscapes.json` describes both recordings.

## Ethics, licence and funding

All participants gave signed informed consent, which covered the EEG, wrist-device and thermal-camera recordings and the sharing of pseudonymised data. The study followed the procedure of the field-laboratory studies at the Technical University of Denmark (Dave et al., 2026); personal information was pseudonymised in accordance with the GDPR, and under the university's procedures no Ethics Review Board approval was required.

Licence: CC BY 4.0. Funded by the European Union's Horizon Europe programme under the Marie Skłodowska-Curie Doctoral Network MuSIC (grant agreement No. 101073357).

## Citation

Tomar P., Dave T., Wargocki P., Pisello A.L. (2026). {title} (v{version}). Zenodo. https://doi.org/{doi}

## References

- Experimental protocol: Tomar P., Dave T., Wargocki P., Pisello A.L. (2026). Multimodal assessment of cognitive performance, physiology and perceived indoor environment under warm and cool LED lighting and urban soundscapes (SYNAPSE). protocols.io. https://doi.org/10.17504/protocols.io.m7vznm25rvx1/v1
- Task images: Tomar P. (2026). SYNAPSE experiment platform. Zenodo. https://doi.org/10.5281/zenodo.23768059
- Dave T., Andersen R.K., Wargocki P. (2026). Sensory influence and overall comfort: combined effects of thermal, acoustic, and visual stimulus intensity in office environments. Building and Environment 304, 115068. https://doi.org/10.1016/j.buildenv.2026.115068
"""


BIDS_STANDARD = {"name": "Channel label", "type": "Channel type", "units": "Physical unit of the channel",
                 "low_cutoff": "Frequency of the high-pass filter (n/a = none)", "high_cutoff": "Frequency of the low-pass filter (n/a = none)",
                 "reference": "Reference electrode", "status": "Channel status (good/bad)", "status_description": "Free-text note on the channel status"}
EXTRA_SIDECARS: dict[str, str] = {}


def sidecar(f: Path) -> dict:
    """Resolve the JSON sidecar of a TSV following the BIDS inheritance principle (same name first,
    then names with sub-/ses- entities removed, walking up to the dataset root)."""
    stem = f.name.replace(".tsv", "")
    parts = stem.split("_")
    names = [stem, "_".join(p for p in parts if not p.startswith(("sub-", "ses-")))]
    d = f.parent
    while True:
        for n in names:
            if n and (d / f"{n}.json").exists():
                return json.loads((d / f"{n}.json").read_text())
        if d == B:
            break
        d = d.parent
    for k, v in EXTRA_SIDECARS.items():
        if k in f.parts:
            return json.loads((B / v).read_text())
    return {}


def data_dictionary() -> pd.DataFrame:
    rows = []
    for f in sorted(p for p in B.rglob("*.tsv") if p.name != "DATA_DICTIONARY.tsv"):
        rel = f.relative_to(B).as_posix()
        if rel.startswith(("sub-", "derivatives/wristband_vendor_metrics/sub-", "derivatives/facial_temperature/sub-")) and not rel.endswith("_sessions.tsv"):
            # one representative file per file type
            key = "_".join(p for p in f.name.split("_") if not p.startswith(("sub-", "ses-")))
            if any(r["file_pattern"].endswith(key) for r in rows):
                continue
            pattern = "sub-*/ses-*/.../*_" + key
        else:
            pattern = rel
        cols = pd.read_csv(f, sep="\t", nrows=0).columns
        side = sidecar(f)
        for c in cols:
            m = side.get(c, {}) if isinstance(side.get(c, {}), dict) else {}
            if not m.get("Description") and c in BIDS_STANDARD:
                m = {"Description": BIDS_STANDARD[c] + " (BIDS standard column)"}
            rows.append({"file_pattern": pattern, "column": c, "description": m.get("Description", ""),
                         "units": m.get("Units", ""), "levels": json.dumps(m["Levels"], ensure_ascii=False) if "Levels" in m else ""})
    return pd.DataFrame(rows)


def write_sidecars() -> None:
    from synapse_curation.sidecars_extra import SIDECARS, THERMAL_QC
    for key, cols in SIDECARS.items():
        merge = key.endswith("+")
        path = B / key.rstrip("+")
        old = json.loads(path.read_text()) if merge and path.exists() else {}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**old, **cols}, indent=2, ensure_ascii=False) + "\n")
    for name in THERMAL_QC:
        f = B / "derivatives" / "facial_temperature" / "quality_control" / f"{name}.tsv"
        cols = pd.read_csv(f, sep="\t", nrows=0).columns
        (f.with_suffix(".json")).write_text(json.dumps(
            {c: {"Description": f"Thermal extraction QC variable '{c}'; definitions in derivatives/facial_temperature/quality_control/README.md"} for c in cols},
            indent=2) + "\n")


def _stem(f: Path) -> str:
    return f.name.removesuffix(".gz").removesuffix(".tsv")


def localise_sidecars() -> None:
    """Replace shared (inherited) sidecars by one sidecar next to each data file, so that every table has its
    own description and no shared sidecars remain at the top of the dataset or of a derivative folder."""
    shared = [j for j in B.rglob("*.json") if not j.name.startswith("sub-") and j.name not in ("dataset_description.json", "participants.json")
              and not (j.with_suffix(".tsv")).exists()]
    for j in shared:
        targets = [f for f in j.parent.rglob("sub-*") if f.name.endswith((".tsv", ".tsv.gz"))
                   and "_".join(p for p in _stem(f).split("_") if not p.startswith(("sub-", "ses-"))) == j.stem]
        if not targets:
            continue
        inherited = json.loads(j.read_text())
        for f in targets:
            own = f.parent / f"{_stem(f)}.json"
            specific = json.loads(own.read_text()) if own.exists() else {}
            own.write_text(json.dumps({**inherited, **specific}, indent=2, ensure_ascii=False) + "\n")
        j.unlink()


def main() -> None:
    dec = decisions()
    write_sidecars()
    localise_sidecars()
    s = stats()
    doi = dec["D27_doi"]["value"].split(" ")[0]
    from synapse_curation import zenodo
    (B / "CITATION.cff").write_text(zenodo.citation_cff())
    for junk in B.rglob(".DS_Store"):
        junk.unlink()
    (B / "CHANGES").write_text(f"{__version__} 2026-10-27\n  - First public release.\n")
    dd = data_dictionary()
    write_tsv(dd, B / "DATA_DICTIONARY.tsv")
    files = [f for f in B.rglob("*") if f.is_file() and f != B / "README.md"]
    size = sum(f.stat().st_size for f in files) / 1e6
    (B / "README.md").write_text(README.format(version=__version__, bids="1.10.0", doi=doi,
                                               n_columns=len(dd), n_files=len(files) + 1, title=zenodo.TITLE,
                                               subtitle=(lambda t: t[0].upper() + t[1:])(zenodo.TITLE.split(": ", 1)[1]), size=f"{size:.0f} MB", **s))
    print(f"README, CHANGES, DATA_DICTIONARY ({len(dd)} columns, {(dd.description == '').sum()} undescribed)")


if __name__ == "__main__":
    main()
