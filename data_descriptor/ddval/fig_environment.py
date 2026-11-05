"""Figure 7: measured indoor conditions during every session and the distribution of perceived-environment ratings."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import BIDS, C, DOUBLE, NUM, SESSIONS, panel, room, save, session_window, sessions, tsv

VARS = [("air_temperature", "Air temperature (°C)"), ("relative_humidity", "Relative humidity (%)"),
        ("co2", "CO$_2$ (ppm)"), ("sound_level", "Sound level (dB)"), ("illuminance", "Illuminance (sensor, lx)")]
IEQ = [("thermal_sensation", "Thermal sensation"), ("thermal_satisfaction", "Thermal satisfaction"),
       ("air_quality_assessment", "Air quality"), ("air_quality_satisfaction", "Air quality satisfaction"),
       ("light_assessment", "Light assessment"), ("light_satisfaction", "Light satisfaction"),
       ("acoustic_assessment", "Acoustic assessment"), ("acoustic_satisfaction", "Acoustic satisfaction")]


def session_room() -> pd.DataFrame:
    r = room()
    s = sessions().set_index(["participant_id", "session_id"])
    rows = []
    for sub, ses in SESSIONS:
        t0, t1 = session_window(sub, ses)
        w = r[(r.time_unix >= t0) & (r.time_unix <= t1)]
        rows.append(w.assign(participant_id=sub, session_id=ses, light=s.loc[(sub, ses), "light"], start=t0))
    return pd.concat(rows, ignore_index=True)


def make() -> None:
    d = session_room()
    keys = d.groupby(["participant_id", "session_id"]).start.first().sort_values()
    fig = plt.figure(figsize=(DOUBLE, 118 / 25.4))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.05], hspace=0.62, wspace=0.55)
    for j, (col, lab) in enumerate(VARS):
        ax = fig.add_subplot(gs[0, j])
        for i, k in enumerate(keys.index):
            v = d.set_index(["participant_id", "session_id"]).loc[k, col].dropna().to_numpy()
            light = d.set_index(["participant_id", "session_id"]).loc[k, "light"].iloc[0]
            ax.plot([v.min(), v.max()], [i, i], color=C[light], lw=0.7)
            ax.plot(np.median(v), i, "o", color=C[light], ms=2.2)
        ax.set_yticks(range(len(keys)), [f"{a}/{b}" for a, b in keys.index] if j == 0 else [], fontsize=4.6)
        ax.invert_yaxis()
        ax.set_xlabel(lab, fontsize=5.8)
        ax.tick_params(axis="x", labelsize=5.2)
        if j == 0:
            panel(ax, "a", x=-0.85, y=1.02)
            ax.text(0.0, 1.03, "warm", color=C["warm"], transform=ax.transAxes, fontsize=5.5)
            ax.text(0.35, 1.03, "cool", color=C["cool"], transform=ax.transAxes, fontsize=5.5)
    for col, name in [("air_temperature", "Temp"), ("relative_humidity", "RH"), ("co2", "COtwo"), ("sound_level", "Sound"), ("illuminance", "Lux")]:
        med = d.groupby(["participant_id", "session_id"])[col].median()
        NUM.set(f"Room{name}Min", med.min(), "{:.0f}" if col in ("co2", "illuminance") else "{:.1f}")
        NUM.set(f"Room{name}Max", med.max(), "{:.0f}" if col in ("co2", "illuminance") else "{:.1f}")
    NUM.set("RoomRecordsSession", len(d))
    # the room sensor sees daylight from the window as well as the luminaires: compare it between lighting conditions
    lux = d.groupby(["participant_id", "session_id", "light"]).illuminance.median().groupby("light").median()
    NUM.set("RoomLuxWarm", lux["warm"], "{:.0f}")
    NUM.set("RoomLuxCool", lux["cool"], "{:.0f}")
    # the operator log is the only record of the luminaire preset; logged values are rounded in some sessions
    s = sessions()
    logged = s.dropna(subset=["cct_logged_K"])
    same_side = (logged.cct_logged_K > 4000) == (logged.light == "cool")
    if not same_side.all():
        raise ValueError("logged CCT contradicts the lighting condition")
    NUM.set("CCTLogged", len(logged))

    # b: perceived IEQ ratings, all T1-T3 forms pooled
    ax = fig.add_subplot(gs[1, :])
    forms = pd.concat([tsv(f.relative_to(BIDS)).assign(form=f.name[:2]) for f in sorted(BIDS.glob("phenotype/T[123]_*_session?.tsv"))])
    meta = json.loads((BIDS / "phenotype" / "T1_ieq_sessionA.json").read_text())
    cmap = plt.get_cmap("RdBu_r")
    for i, (item, lab) in enumerate(IEQ):
        levels = meta[f"{item}_code"]["Levels"]
        k = len(levels)
        counts = forms[f"{item}_code"].value_counts().reindex(range(1, k + 1), fill_value=0)
        frac = counts / counts.sum()
        left = 0.0
        for code, fv in frac.items():
            col = cmap(0.1 + 0.8 * (code - 1) / (k - 1))
            ax.barh(i, fv, left=left, color=col, ec="white", lw=0.3, height=0.72)
            if fv >= 0.06:
                ax.text(left + fv / 2, i, levels[str(code)].split(": ", 1)[1].replace("Slightly ", "Sl. ")
                        .replace("Very ", "V. ").replace("satisfied", "sat.").replace("dissatisfied", "dissat.")
                        .replace("acceptable", "acc."), ha="center", va="center", fontsize=4.2)
            left += fv
    ax.set_yticks(range(len(IEQ)), [l for _, l in IEQ], fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
    ax.set_xlabel(f"Responses (%), questionnaires T1–T3 of all sessions pooled (n = {len(forms)} forms)")
    panel(ax, "b", x=-0.12, y=1.02)
    NUM.set("IEQForms", len(forms))
    save(fig, "fig7_environment")
