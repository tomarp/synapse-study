"""Stimuli: tables describing the task images (distributed with the experiment platform, referenced by path and
SHA-256) and the two soundscape recordings (not redistributable; characterised from the raw archive)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

from . import paths
from .common import sha256, write_json, write_tsv


def a_weighting(fs: float):
    """IEC 61672 A-weighting as a digital filter (bilinear transform of the analogue prototype)."""
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    a1000 = 1.9997
    nums = [(2 * np.pi * f4) ** 2 * 10 ** (a1000 / 20), 0, 0, 0, 0]
    dens = np.polymul([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2], [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2])
    dens = np.polymul(np.polymul(dens, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2])
    return signal.bilinear(nums, dens, fs)


def characterise(path) -> dict:
    x, fs = sf.read(path, always_2d=True)
    b, a = a_weighting(fs)
    xa = signal.lfilter(b, a, x, axis=0)
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))          # noqa: E731
    # 1-s LAeq-like levels re full scale, for the level distribution
    n = int(fs)
    frames = xa[: len(xa) // n * n].reshape(-1, n, x.shape[1])
    l1s = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=(1, 2))) + 1e-12)
    return {"file": f"media/audio/{path.name}", "sha256": sha256(path), "sampling_rate_hz": fs, "channels": x.shape[1],
            "duration_s": round(len(x) / fs, 2), "subtype": sf.info(path).subtype,
            "level_dBFS_unweighted": round(20 * np.log10(rms(x)), 2),
            "level_dBFS_A_weighted": round(20 * np.log10(rms(xa)), 2),
            "level_dBFS_A_1s_p10": round(float(np.percentile(l1s, 90)), 2),
            "level_dBFS_A_1s_p50": round(float(np.percentile(l1s, 50)), 2),
            "level_dBFS_A_1s_p90": round(float(np.percentile(l1s, 10)), 2),
            "interaural_level_difference_dB": round(20 * np.log10(rms(x[:, 0]) / rms(x[:, 1])), 2) if x.shape[1] == 2 else None,
            "third_octave_band_levels_dBFS": third_octave(x.mean(axis=1), fs)}


THIRD_OCTAVE_HZ = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
                   2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000]


def third_octave(x: np.ndarray, fs: float) -> dict:
    """Unweighted 1/3-octave band levels (IEC 61260 base-10 band edges) of the mono mix, dB re full scale,
    from the Welch power spectrum (Hann, 1-s segments)."""
    f, p = signal.welch(x, fs=fs, window="hann", nperseg=int(fs), scaling="spectrum")
    out = {}
    for fc in THIRD_OCTAVE_HZ:
        exact = 1000 * 10 ** (round(10 * np.log10(fc / 1000)) / 10)
        lo, hi = exact * 10 ** (-1 / 20), exact * 10 ** (1 / 20)
        band = p[(f >= lo) & (f < hi)].sum()
        out[str(fc)] = round(10 * np.log10(band + 1e-20), 1)
    return out


GENERATORS = [(b"Image Creator from Designer", "Microsoft Designer Image Creator (DALL-E 3)"), (b"GPT-4o", "OpenAI GPT-4o (ChatGPT)")]


def image_provenance(f) -> dict:
    """Generator and creation time read from the C2PA content credentials embedded in the image file."""
    b = f.read_bytes()
    gen = next((name for key, name in GENERATORS if key in b), "n/a")
    when = re.search(rb"when.?(20\d\d-\d\d-\d\dT\d\d:\d\d:\d\dZ)", b)
    return {"generator": gen, "ai_generated": "yes" if b"trainedAlgorithmicMedia" in b else "n/a",
            "c2pa_credentials": "yes" if b"c2pa" in b else "no", "created_utc": when.group(1).decode() if when else "n/a"}


def write(out, visual: pd.DataFrame, items: pd.DataFrame) -> None:
    # images
    src = paths.PLATFORM / "media"
    out.mkdir(parents=True, exist_ok=True)
    prov = []
    for sub in ("images/visual_count", "images/visual_shape", "images/stroop_emotional"):
        for f in sorted((src / sub).iterdir()):
            if f.is_file() and f.suffix.lower() in (".png", ".jpeg", ".jpg"):
                prov.append({"file": f"media/{sub}/{f.name}", **image_provenance(f), "sha256": sha256(f)})
    write_tsv(pd.DataFrame(prov), out / "image_provenance.tsv")
    write_json({"Description": "Task images used in the experiment, read from the C2PA content credentials embedded in the files (decision D29). "
                               "The images and the Stroop image prompts (media/images/stroop_emotional/generation_prompts.txt) are distributed with the experiment platform.",
                "file": {"Description": "Path of the image within the experiment platform"},
                "generator": {"Description": "Image generator named in the credentials"},
                "ai_generated": {"Description": "Credentials declare digital source type trainedAlgorithmicMedia (IPTC)"},
                "c2pa_credentials": {"Description": "File carries C2PA content credentials"},
                "created_utc": {"Description": "Creation time in the credentials, where recorded"},
                "sha256": {"Description": "SHA-256 of the image file"}}, out / "image_provenance.json")
    # soundscapes: original recordings (raw archive only; not redistributable, D29)
    chars = []
    for f in sorted((paths.RAW / "stimuli" / "soundscapes").glob("*.wav")):
        c = characterise(f)
        c["file"] = f.name
        chars.append(c)
    ref = next(c for c in chars if c["file"].startswith("People"))
    for c in chars:
        c["level_A_relative_to_babble_dB"] = round(c["level_dBFS_A_weighted"] - ref["level_dBFS_A_weighted"], 2)
    write_json({"Description": "Characterisation of the two soundscape recordings (computed from the original files; the audio is not distributed because its source on YouTube was not recorded, decision D29). Presentation level was not measured in SYNAPSE; reference: the babble recording (People_Defi.wav) played at maximum tablet volume through the same JBL Tune 770NC headphones measured 53.6 +/- 2.5 dB at the ear cup (Dave et al., Build. Environ. 304 (2026) 115068). If both files were played at the same device volume, the traffic level at the ear differs from the babble level by level_A_relative_to_babble_dB (assumption, decision D19).",
                "Levels_dBFS": "dB re digital full scale; A = IEC 61672 A-weighting applied digitally; 1-s percentiles p10 (loud), p50, p90 (quiet); third_octave_band_levels_dBFS: unweighted band levels of the mono mix, keyed by nominal centre frequency (Hz)",
                "files": chars}, out / "soundscapes.json")
    # item table with empirical response statistics
    v = visual[visual["exclusion"] == ""]
    rows = []
    for r in items.itertuples():
        g = v[v["image"] == r.image]
        row = {"image": r.image, "task": r.task, "key_nominal": r.key_nominal, "key_validity": r.key_validity,
               "rater_A_note": r.rater_A_note, "file": f"media/images/visual_{'count' if r.task == 'visualcount' else 'shape'}/{r.image}",
               "sha256": sha256(src / "images" / ("visual_count" if r.task == "visualcount" else "visual_shape") / r.image),
               "n_responses": len(g), "rt_median_s": round(g["rt_ms"].median() / 1000, 2) if len(g) else np.nan}
        if r.task == "visualcount":
            c = g["count_response"]
            row.update(response_median=c.median(), response_q25=c.quantile(.25), response_q75=c.quantile(.75),
                       prop_exact_nominal=round(float((c == float(r.key_nominal)).mean()), 3),
                       prop_within_10pct_nominal=round(float((abs(c - float(r.key_nominal)) <= 0.1 * float(r.key_nominal)).mean()), 3))
        else:
            row.update(prop_same=round(float((g["response"].str.upper() == "SAME").mean()), 3),
                       prop_agree_nominal=round(float((g["response"].str.upper() == r.key_nominal).mean()), 3))
        rows.append(row)
    t = pd.DataFrame(rows)
    write_tsv(t, out / "visual_items.tsv")
    write_json({"Description": "Visual task items: nominal keys (file-name generation targets), key-validity rating by rater A (curator; pending confirmation by a second rater), and empirical response statistics (analysable trials, both sessions). See decision D21.",
                "key_validity": {"Levels": {"nominal": "no indication of error", "duplicate_consistent": "byte-identical duplicate with the same key",
                                            "indeterminate": "byte-identical duplicate with the opposite key", "ambiguous": "count depends on interpretation",
                                            "ill_defined": "recursive/impossible geometry; no well-defined count"}}},
               out / "visual_items.json")
