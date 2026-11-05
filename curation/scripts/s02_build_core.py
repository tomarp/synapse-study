"""Step 02: behaviour, questionnaires (phenotype), participants/sessions, timeline, room monitor,
stimuli and the related QC derivatives."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synapse_curation import behavior, participants, paths, questionnaires, room, timeline  # noqa: E402
from synapse_curation.columns import COMMON_BEH, TASK_BEH, TASK_INFO  # noqa: E402
from synapse_curation.common import (NA, decisions, sessions, unix_to_iso, write_json,  # noqa: E402
                                     write_tsv)
from synapse_curation import stimuli  # noqa: E402

B = paths.BIDS
DER = B / "derivatives"
FAMILY_OF = {"nbackletter": "nback", "nbackdigit": "nback", "flankerarrow": "flanker", "flankerdigit": "flanker",
             "visualcount": "visual", "visualshape": "visual"}


def family(task: str) -> str:
    return FAMILY_OF.get(task, "stroop")


# ------------------------------------------------------------------------------------ behaviour
def write_behaviour(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rename = {"participant_raw": "participant_id_raw", "session_raw": "session_id_raw"}
    log = []
    for fam, d in tables.items():
        log.append(d.groupby("exclusion", dropna=False).size().rename("rows").reset_index().assign(task_family=fam))
        keep = d[d["exclusion"].isin(["", "superseded_run"])].copy()
        keep["superseded_run"] = keep["exclusion"] == "superseded_run"
        keep["submitted_time"] = unix_to_iso(keep["submitted_unix"], ms=False)
        keep = keep.rename(columns=rename)
        cols = (["trial_index", "run", "superseded_run", "round", "soundscape"] + list(TASK_BEH[fam]) +
                ["response", "rt_ms", "rt_flag", "correct", "correct_recorded", "score_discrepancy",
                 "submitted_time", "submitted_unix", "outside_logged_window",
                 "participant_id_raw", "session_id_raw", "src_file", "src_row"])
        for (p, s, task), g in keep.groupby(["participant", "session", "task"]):
            g = g.sort_values("src_row").copy()
            g["trial_index"] = np.arange(1, len(g) + 1)
            g["round"] = g["round"].astype(int)
            out = B / f"sub-{p}" / f"ses-{s}" / "beh" / f"sub-{p}_ses-{s}_task-{task}_beh.tsv"
            write_tsv(g[[c for c in cols if c in g.columns]], out)
    for task, (name, desc) in TASK_INFO.items():
        fam = family(task)
        side = {"TaskName": name, "TaskDescription": desc,
                "Instructions": "Task instructions were shown on screen by the platform before each task (text in the platform code; protocol on protocols.io).",
                "InstitutionName": "Technical University of Denmark", "InstitutionAddress": "Kongens Lyngby, Denmark", "InstitutionalDepartmentName": "Department of Environmental and Resource Engineering",
                **COMMON_BEH, **TASK_BEH[fam]}
        write_json(side, B / f"task-{task}_beh.json")
    return pd.concat(log, ignore_index=True)


# ------------------------------------------------------------------------------------ phenotype
def write_phenotype(q: dict[str, pd.DataFrame]) -> pd.DataFrame:
    link_rows = []
    specs = {"T0": ("T0_pre_session", q["items_T0"]), "T1": ("T1_ieq", q["items_T1"]),
             "T2": ("T2_task_affect_ieq", q["items_T23"]), "T3": ("T3_task_affect_ieq", q["items_T23"])}
    for k, (fname, items) in specs.items():
        d = q[k][q[k]["exclusion"] == ""].copy()
        d.insert(0, "participant_id", "sub-" + d["participant"])
        d.insert(1, "session_id", "ses-" + d["session"].astype(str))
        d["submitted_time"] = unix_to_iso(d["submitted_unix"], ms=False)
        meta = {"participant_id": {"Description": "Participant"}, "session_id": {"Description": "Session"},
                "submitted_time": {"Description": "Google-server submission time (ISO 8601, CEST)"}}
        cols = ["participant_id", "session_id", "submitted_time"]
        if k == "T0":
            cols += ["session_self_reported", "session_corrected"]
            meta["session_self_reported"] = {"Description": "Session letter typed by the participant"}
            meta["session_corrected"] = {"Description": "true if session_id was assigned from the logged session window because the typed letter was wrong"}
        else:
            cols += ["link_quality", "link_residual_s", "link_alt_cost_ratio"]
            meta["link_quality"] = {"Description": "Strength of the timing-based link of this anonymous form to the participant",
                                    "Levels": {"single_participant": "only one participant in the room", "causally_unique": "any other assignment is causally impossible",
                                               "strong": "alternative >= 4x worse", "moderate": "alternative 1.5-4x worse", "weak": "alternative < 1.5x worse (treat with caution)"}}
            meta["link_residual_s"] = {"Description": "Seconds between form submission and the participant's next/previous platform activity", "Units": "s"}
            meta["link_alt_cost_ratio"] = {"Description": "Cost of the best alternative assignment / cost of the chosen one (1e6 = no causal alternative)"}
        for r in items.itertuples():
            if r.variable in questionnaires.HEALTH or r.variable in ("participant_raw", "session_raw"):
                continue            # health screening withheld (D26)
            entry = {"Description": r.title, "LongName": r.export_header.replace("\n", " ")}
            if r.options and r.type != "checkboxes":
                entry["Levels"] = {o: o for o in r.options}
            cols.append(r.variable)
            meta[r.variable] = entry
            if r.variable + "_code" in d.columns:
                cols.append(r.variable + "_code")
                if r.variable in questionnaires.PREF_CODES:
                    lv = {str(v): kk for kk, v in questionnaires.PREF_CODES[r.variable].items()}
                else:
                    lv = {str(i + 1): o for i, o in enumerate(r.options)}
                meta[r.variable + "_code"] = {"Description": f"Numeric code of {r.variable} (position of the option in the deployed form)", "Levels": lv}
        if k == "T0":
            cols += ["clothing_clo_est", "clothing_note"]
            meta["clothing_clo_est"] = {"Description": "Estimated clothing insulation incl. office chair (ASHRAE 55 / ISO 9920 garment values; +-0.1 clo); n/a if the checklist lacks an upper- or lower-body garment", "Units": "clo"}
            meta["clothing_note"] = {"Description": "Reason clothing_clo_est is missing"}
        d = d.sort_values(["participant_id", "session_id"])
        for ses, g in d.groupby("session_id"):      # BIDS phenotype tables are indexed by participant_id only
            name = f"{fname}_session{ses[4:]}"
            write_tsv(g[cols], B / "phenotype" / f"{name}.tsv")
            write_json({"MeasurementToolMetadata": {"Description": f"Questionnaire {k}, session {ses[4:]} (Google Form embedded in the task platform; definition archived in sourcedata/questionnaire_forms)"}, **meta},
                       B / "phenotype" / f"{name}.json")
        if k != "T0":
            link_rows.append(d[["participant_id", "session_id", "submitted_time", "link_quality", "link_residual_s", "link_alt_cost_ratio"]].assign(questionnaire=k))
    return pd.concat(link_rows, ignore_index=True)


def write_background() -> None:
    u, summary = participants.background_unlinked()
    vc = u["education_field"].value_counts()
    u.loc[u["education_field"].isin(vc[vc < 2].index), "education_field"] = "other"     # small-cell suppression
    out = B / "background_questionnaire"
    write_tsv(u, out / "responses_unlinked.tsv")
    write_tsv(summary, out / "sample_summary.tsv")
    write_json({"Description": "Pre-study background questionnaire, 9 of 10 participants. Rows are NOT linked to participant_id (no key exists; decision D08) and are shuffled. Ages are 5-year bands; categories with < 2 respondents in education_field are merged into 'other'; names, e-mail, residence and free-text health information are withheld.",
                "respondent": {"Description": "Arbitrary respondent label (not a participant code)"}},
               out / "responses_unlinked.json")


# ------------------------------------------------------------------------------------ participants / sessions
def write_participants_sessions(tl: pd.DataFrame, cfg: pd.DataFrame, avail: pd.DataFrame) -> None:
    p = participants.participants_table(avail)
    write_tsv(p, B / "participants.tsv")
    write_json({
        "participant_id": {"Description": "Pseudonymous participant code"},
        "group": {"Description": "Test group (participants tested together)"},
        "co_tested_with": {"Description": "Participant tested concurrently at the other workstation"},
        "seat": {"Description": "Workstation (decision D07)", "Levels": {"R": "wall-side seat (right in the thermal image)", "L": "other seat", "C": "centre (single participant, camera re-aimed)"}},
        "lighting_order": {"Description": "Lighting condition of the first session", "Levels": {"cool_first": "cool then warm", "warm_first": "warm then cool"}},
    }, B / "participants.json")
    s = sessions()
    span = timeline.session_span(tl)
    s = s.merge(span, on=["participant_id", "session_id"], how="left").merge(cfg, on=["participant_id", "session_id"], how="left")
    sw = tl[tl.phase == "soundscape_switch"][["participant_id", "session_id", "onset", "offset"]]
    s = s.merge(sw.rename(columns={"onset": "sw0", "offset": "sw1"}), on=["participant_id", "session_id"], how="left")
    for pid, g in s.groupby("participant_id"):
        out = pd.DataFrame({
            "session_id": g["session_id"], "acq_time": [t[:19] for t in unix_to_iso(g["session_start_unix"])],
            "session_order": g["session_order"].astype(int), "light": g["light"], "cct_setpoint_K": g["cct_setpoint_K"],
            "cct_logged_K": g["cct_logged_K"], "ui_theme": g["ui_theme"], "ui_theme_basis": g["ui_theme_basis"],
            "soundscape_round1": g["sound_round1"], "soundscape_round2": g["sound_round2"],
            "soundscape_switch_window_start": unix_to_iso(g["sw0"], ms=False), "soundscape_switch_window_end": unix_to_iso(g["sw1"], ms=False),
            "session_start": unix_to_iso(g["session_start_unix"], ms=False), "session_end": unix_to_iso(g["session_end_unix"], ms=False),
            "log_start": g["log_start"], "log_end": g["log_end"], "log_end_raw": g["log_end_raw"],
            "platform_configuration": np.where((g["participant"].isin(["P01", "P02"])) & (g["session"] == "A"), "early", "main"),
            "cfg_nback_trials_per_level": g["cfg_nback_trials_per_level"], "cfg_flanker_trials_per_variant": g["cfg_flanker_trials_per_variant"],
            "cfg_stroop_paradigm": g["cfg_stroop_paradigm"], "operator_notes": g["log_notes"].replace("", NA), "corrections": g["corrections"].replace("", NA),
        })
        write_tsv(out, B / pid / f"{pid}_sessions.tsv")
    write_json({
        "acq_time": {"Description": "Session start (T0 submission), local time CEST"},
        "session_order": {"Description": "1 = first session of the day, 2 = second (after a ~10-15 min break)"},
        "light": {"Description": "Lighting condition", "Levels": {"warm": "warm-white LED preset", "cool": "cool-white LED preset"}},
        "cct_setpoint_K": {"Description": "Luminaire CCT setpoint (decision D04; not measured at the eye)", "Units": "K"},
        "cct_logged_K": {"Description": "CCT written in the operator log (n/a = not logged)", "Units": "K"},
        "ui_theme": {"Description": "Task platform colour theme", "Levels": {"dark": "dark theme", "light": "light theme"}},
        "ui_theme_basis": {"Description": "Source of ui_theme (D06)"},
        "soundscape_round1": {"Description": "Soundscape during round 1"}, "soundscape_round2": {"Description": "Soundscape during round 2"},
        "soundscape_switch_window_start": {"Description": "Earliest time the soundscape could have been switched (end of round 1)"},
        "soundscape_switch_window_end": {"Description": "Latest time (start of round 2); the switch was not time-stamped (D19)"},
        "session_start": {"Description": "Data-derived start: T0 submission"}, "session_end": {"Description": "Data-derived end: T3 submission or last response"},
        "log_start": {"Description": "Start time in the operator log (HH:MM)"}, "log_end": {"Description": "End time in the operator log, corrected where documented"},
        "log_end_raw": {"Description": "End time as written in the operator log"},
        "platform_configuration": {"Description": "Platform build", "Levels": {"early": "early configuration (fixed n-back lists with key errors, emotional Stroop, 15 flanker trials)", "main": "main configuration"}},
        "cfg_nback_trials_per_level": {"Description": "Trials per n-back level in this session (reconstructed, D17/D18)"},
        "cfg_flanker_trials_per_variant": {"Description": "Flanker trials per variant in this session (reconstructed, D17/D18)"},
        "cfg_stroop_paradigm": {"Description": "Stroop variants administered"},
        "operator_notes": {"Description": "Operator log notes (paraphrased, non-identifying)"},
        "corrections": {"Description": "Corrections applied to logged values (decision IDs)"},
    }, B / "sessions.json")


# ------------------------------------------------------------------------------------ room
def write_room() -> None:
    d = room.load()
    write_tsv(d, B / "environment" / "room_monitor_1min.tsv")
    write_json({"Description": "ATMOCUBE room monitor at the centre of the room between the workstations, 1-min records 20-27 June 2025 (decision D15). Covers all sessions; 65-min gap on 24 June 10:23-11:28 (before testing).",
                "Manufacturer": "Atmotube", "ManufacturersModelName": "ATMOCUBE", **room.sidecar()},
               B / "environment" / "room_monitor_1min.json")


def main() -> None:
    items = pd.read_csv(paths.CONFIG / "visual_item_review.tsv", sep="\t")
    beh = behavior.load_all(items)
    q = questionnaires.load_linked()
    tl = timeline.build(beh, q)
    cfg = behavior.task_configuration(beh)
    paths.ensure(B, DER)
    excl = write_behaviour(beh)
    links = write_phenotype(q)
    write_background()
    write_room()
    stimuli.write(B / "stimuli", beh["visual"], items)
    # derivatives: timeline and QC
    tlo = tl.copy()
    tlo["onset_time"] = unix_to_iso(tlo["onset"])
    tlo["offset_time"] = unix_to_iso(tlo["offset"])
    tlo = tlo.rename(columns={"onset": "onset_unix", "offset": "offset_unix"})
    write_tsv(tlo[["participant_id", "session_id", "phase", "task", "round", "soundscape", "onset_time", "offset_time",
                   "onset_unix", "offset_unix", "duration_s", "n_responses"]], DER / "session_timeline" / "timeline.tsv")
    paths.ensure(paths.WORK)
    tl.to_pickle(paths.WORK / "timeline.pkl")
    write_tsv(cfg, DER / "quality_control" / "task_configuration.tsv")
    write_tsv(links, DER / "quality_control" / "questionnaire_linkage.tsv")
    write_tsv(excl.assign(exclusion=excl["exclusion"].replace("", "analysable")), DER / "quality_control" / "behaviour_exclusion_counts.tsv")
    # availability table for participants.tsv (behaviour/questionnaires here; EEG/physio added in later steps)
    s = sessions()
    avail = s[["participant_id", "session_id"]].copy()
    avail["has_behaviour"] = True
    avail.to_pickle(paths.WORK / "availability_core.pkl")
    write_participants_sessions(tl, cfg, avail)
    print("core done:", {k: int((v.exclusion == "").sum()) for k, v in beh.items()})


if __name__ == "__main__":
    main()
