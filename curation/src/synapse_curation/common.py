"""Shared helpers: time handling, session table, decision register, TSV/JSON writers."""
from __future__ import annotations

import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import paths

TZ = "Europe/Copenhagen"                 # CEST (UTC+02:00) on all study days (decision D28)
STUDY_DATES = ("2025-06-24", "2025-06-25", "2025-06-26")
FIRST_SESSION = pd.Timestamp("2025-06-24 12:40", tz=TZ)
NA = "n/a"


# ----------------------------------------------------------------------------- time
def local_to_unix(ts) -> np.ndarray:
    """Naive local (CEST) timestamps -> Unix seconds."""
    s = pd.to_datetime(pd.Series(ts))
    if s.dt.tz is None:
        s = s.dt.tz_localize(TZ)
    return (s.dt.tz_convert("UTC") - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def unix_to_iso(t, ms: bool = True) -> list[str]:
    s = pd.to_datetime(pd.Series(np.asarray(t, dtype=float)), unit="s", utc=True).dt.tz_convert(TZ)
    fmt = "%Y-%m-%dT%H:%M:%S.%f" if ms else "%Y-%m-%dT%H:%M:%S"
    out = s.dt.strftime(fmt)
    if ms:
        out = out.str[:-3]
    return (out + "+02:00").tolist()


# ----------------------------------------------------------------------------- config
@lru_cache
def decisions() -> dict:
    return yaml.safe_load((paths.CONFIG / "decisions.yaml").read_text())


@lru_cache
def sessions() -> pd.DataFrame:
    """Operator-log session table (config/sessions.tsv) with derived columns."""
    s = pd.read_csv(paths.CONFIG / "sessions.tsv", sep="\t", dtype=str, keep_default_na=False)
    s["participant"] = s["participant_id"].str[4:]
    s["session"] = s["session_id"].str[4:]
    s["log_start_unix"] = local_to_unix(s["date"] + " " + s["log_start"])
    s["log_end_unix"] = local_to_unix(s["date"] + " " + s["log_end"])
    cct = decisions()["D04_lighting"]["value"]
    s["cct_setpoint_K"] = s["light"].map(lambda x: cct[x]["cct_setpoint_K"])
    s["ui_theme"] = s["light"].map({"cool": "dark", "warm": "light"})        # D06
    s["ui_theme_basis"] = np.where(s["ui_theme_logged"] == NA, "investigator (D06)", "operator log")
    return s


def session_of(participant: str, t_unix: float, margin_s: float = 1200) -> str | None:
    """Session letter whose logged window (+- margin) contains t, else None."""
    s = sessions()
    m = s[(s.participant == participant) & (t_unix >= s.log_start_unix - margin_s)
          & (t_unix <= s.log_end_unix + margin_s)]
    return m["session"].iloc[0] if len(m) == 1 else None


# ----------------------------------------------------------------------------- io
def sha256(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            h.update(chunk)
    return h.hexdigest()


def write_tsv(df: pd.DataFrame, path: Path, **kw) -> None:
    """BIDS TSV: tab-separated, 'n/a' for missing (also for empty strings), whole-number columns as
    integers, booleans as true/false, one line per row (no quoting), no index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == bool:
            out[c] = out[c].map({True: "true", False: "false"})
        elif pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].map(lambda v: " ".join(v.split()) if isinstance(v, str) and any(ch in v for ch in "\t\n\r") else v)
            out[c] = out[c].replace("", np.nan)
        elif pd.api.types.is_float_dtype(out[c]):
            v = out[c].dropna()
            if len(v) and np.isfinite(v).all() and (v == v.round()).all() and v.abs().max() < 2**53:
                out[c] = out[c].astype("Int64")
    out.to_csv(path, sep="\t", index=False, na_rep=NA, lineterminator="\n", quoting=csv.QUOTE_NONE, **kw)


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_default) + "\n")
