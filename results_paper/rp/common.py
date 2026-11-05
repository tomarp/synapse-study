"""Shared paths, condition coding, blinding, figure style and the registry of numbers quoted in the results paper.

Every number in the manuscript text is computed by this package from the published dataset and written to
``generated/numbers.tex`` as a LaTeX macro. While the analysis is blind (the default), condition labels are scrambled
(ANALYSIS_PLAN.md, section 8) and all outputs go to ``<out>/blind/``.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]                      # <repo>/results_paper
OUTPUT = HERE.parents[1]                                          # project layout: output/{code,dataset,deliverables}
BIDS = Path(os.environ.get("SYNAPSE_BIDS_ROOT", OUTPUT / "dataset" / "SYNAPSE_dataset")).resolve()
UNBLIND = os.environ.get("SYNAPSE_UNBLIND") == "1"
_OUT = Path(os.environ.get("SYNAPSE_RP_OUT", OUTPUT / "deliverables" / "03_results_paper")).resolve()
PAPER = _OUT if UNBLIND else _OUT / "blind"
FIG, GEN, CACHE = PAPER / "figures", PAPER / "generated", PAPER / "cache"
SEED = 20251105

SUBS = [f"sub-P{i:02d}" for i in range(1, 11)]
PAIR = {"sub-P01": "D1", "sub-P02": "D1", "sub-P03": "D2", "sub-P04": "D2", "sub-P05": "D3", "sub-P06": "D3",
        "sub-P08": "D4", "sub-P09": "D4", "sub-P07": "S7", "sub-P10": "S10"}   # participants tested together

# task variants analysed for performance (ANALYSIS_PLAN.md, family A), in presentation order
TASKS = ["nbackletter", "nbackdigit", "flankerarrow", "flankerdigit", "stroopword", "stroopemoji"]
TASK_LABEL = {"nbackletter": "n-back, letters", "nbackdigit": "n-back, digits", "flankerarrow": "Flanker, arrows",
              "flankerdigit": "Flanker, digits", "stroopword": "Stroop, words", "stroopemoji": "Stroop, emoji",
              "visualshape": "Shape comparison"}
# task blocks whose response strategy departed from the instruction (dataset README)
STRATEGY_BLOCKS = {("sub-P04", "ses-A", "stroopemoji"), ("sub-P06", "ses-A", "stroopemoji"),
                   ("sub-P06", "ses-B", "stroopemoji"), ("sub-P09", "ses-A", "stroopemoji"),
                   ("sub-P09", "ses-B", "stroopemoji"), ("sub-P04", "ses-A", "flankerdigit")}
EARLY = {("sub-P01", "ses-A"), ("sub-P02", "ses-A")}             # early platform configuration

PERCEPTION = ["thermal_sensation", "thermal_satisfaction", "air_quality_assessment", "air_quality_satisfaction",
              "light_assessment", "light_satisfaction", "acoustic_assessment", "acoustic_satisfaction"]
PREFERENCE = ["thermal_preference", "air_movement_preference", "light_preference", "acoustic_preference"]
AFFECT = ["task_difficulty", "task_focus", "relaxation_alertness", "emotional_stability", "task_engagement",
          "cognitive_load", "emotional_valence", "emotional_arousal"]
ITEM_LABEL = {
    "thermal_sensation": "Thermal sensation (cold → hot)", "thermal_satisfaction": "Thermal satisfaction",
    "air_quality_assessment": "Air quality (very bad → very good)", "air_quality_satisfaction": "Air-quality satisfaction",
    "light_assessment": "Light (bright → dark)", "light_satisfaction": "Light satisfaction",
    "acoustic_assessment": "Acoustics (very noisy → very quiet)", "acoustic_satisfaction": "Acoustic satisfaction",
    "thermal_preference": "Thermal preference (cooler → warmer)",
    "air_movement_preference": "Air movement (less → more)", "light_preference": "Light (dimmer → brighter)",
    "acoustic_preference": "Acoustics (louder → quieter)",
    "task_difficulty": "Task difficulty", "task_focus": "Focus", "relaxation_alertness": "Alertness (relaxed → alert)",
    "emotional_stability": "Emotional stability", "task_engagement": "Engagement", "cognitive_load": "Cognitive load",
    "emotional_valence": "Valence (unhappy → happy)", "emotional_arousal": "Arousal (calm → excited)"}
DOMAIN = {"thermal": "Thermal", "air": "Air quality", "light": "Lighting", "acoustic": "Acoustics"}

# Okabe-Ito colour-blind-safe palette (as in the Data Descriptor)
C = {"warm": "#E69F00", "cool": "#0072B2", "people": "#009E73", "traffic": "#CC79A7", "grey": "#7F7F7F",
     "black": "#000000", "sky": "#56B4E9", "red": "#D55E00", "light": "#0072B2", "sound": "#CC79A7"}
MM = 1 / 25.4
SINGLE, ONEHALF, DOUBLE = 90 * MM, 140 * MM, 190 * MM          # Elsevier column widths


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "legend.frameon": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 150, "savefig.dpi": 600, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def panel(ax, letter: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


# ------------------------------------------------------------------------------------------- numbers registry
class Numbers:
    """Values quoted in the manuscript; written as JSON (for checking) and LaTeX macros (for the text)."""

    DIGITS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def set(self, key: str, value, fmt: str = "{}") -> None:
        key = re.sub(r"\d", lambda m: self.DIGITS[int(m.group())], key)
        if not re.fullmatch(r"[A-Za-z]+", key):
            raise ValueError(f"macro names must be letters only: {key}")
        self.values[key] = fmt.format(value) if not isinstance(value, str) else value

    def write(self) -> None:
        GEN.mkdir(parents=True, exist_ok=True)
        old = json.loads((GEN / "numbers.json").read_text()) if (GEN / "numbers.json").exists() else {}
        merged = {**old, **self.values}
        (GEN / "numbers.json").write_text(json.dumps(dict(sorted(merged.items())), indent=1, ensure_ascii=False) + "\n")
        (GEN / "numbers.tex").write_text(
            "% Generated by output/code/results_paper; do not edit by hand.\n"
            + "".join(f"\\newcommand{{\\n{k}}}{{{v}}}\n" for k, v in sorted(merged.items())))


NUM = Numbers()


def tsv(path: Path | str, **kw) -> pd.DataFrame:
    return pd.read_csv(BIDS / path if not Path(path).is_absolute() else path, sep="\t", na_values=["n/a"],
                       keep_default_na=False, **kw)


# ------------------------------------------------------------------------------------------------ blinding
def scramble(sessions: pd.DataFrame) -> pd.DataFrame:
    """Swap the lighting of a participant's two sessions and the soundscapes of a session's two rounds, each with
    probability 1/2 (ANALYSIS_PLAN.md, section 8). Session order and everything else are left unchanged."""
    rng = np.random.default_rng(SEED)
    s = sessions.copy()
    for pid in SUBS:
        if rng.random() < 0.5:
            i = s.index[s.participant_id == pid]
            s.loc[i, "light"] = s.loc[i, "light"].values[::-1]
    for i in s.index:
        if rng.random() < 0.5:
            s.loc[i, ["soundscape_round1", "soundscape_round2"]] = s.loc[i, ["soundscape_round2", "soundscape_round1"]].values
    return s
