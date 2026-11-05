"""Central path configuration (config/paths.toml, overridable via SYNAPSE_<KEY>)."""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]          # output/code/curation/
CONFIG = PKG_ROOT / "config"
_CFG = tomllib.loads((CONFIG / "paths.toml").read_text())


def _resolve(key: str) -> Path:
    env = os.environ.get(f"SYNAPSE_{key.upper()}")
    return (Path(env) if env else PKG_ROOT / _CFG[key]).resolve()


RAW = _resolve("raw_root")
PLATFORM = _resolve("platform_root")
BIDS = _resolve("bids_root")
WORK = _resolve("work_root")


def ensure(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
