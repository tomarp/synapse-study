"""Zenodo deposit: metadata (single source for CITATION.cff and the upload guide) and the archive plan.

Zenodo accepts at most 100 files per record, so the dataset folder is uploaded as one zip of exactly that folder
(SYNAPSE_dataset.zip); README.md is also uploaded unpacked so that it can be read on the Zenodo page.
"""
from __future__ import annotations

from . import __version__
from .common import decisions

CREATORS = [
    {"family": "Tomar", "given": "Puneet", "orcid": "0000-0002-5168-5471",
     "affiliations": ["University of Perugia, Department of Engineering, CIRIAF", "Technical University of Denmark, Department of Environmental and Resource Engineering"]},
    {"family": "Dave", "given": "Tanmay", "orcid": "0009-0004-1255-9140",
     "affiliations": ["Technical University of Denmark, Department of Environmental and Resource Engineering"]},
    {"family": "Wargocki", "given": "Pawel", "orcid": "0000-0003-3865-3560",
     "affiliations": ["Technical University of Denmark, Department of Environmental and Resource Engineering"]},
    {"family": "Pisello", "given": "Anna Laura", "orcid": "0000-0002-4527-6444",
     "affiliations": ["University of Perugia, Department of Engineering, CIRIAF"]},
]
TITLE = ("SYNAPSE: cognitive performance, EEG, wrist physiology, facial thermography and indoor environment "
         "under warm and cool LED lighting and urban soundscapes")
KEYWORDS = ["indoor environmental quality", "multi-domain IEQ", "building informatics", "occupant-centric building operation", "correlated colour temperature", "lighting", "soundscape", "noise",
            "cognitive performance", "n-back", "Stroop", "flanker", "EEG", "Muse S", "photoplethysmography",
            "heart rate variability", "skin temperature", "infrared thermography", "BIDS", "multimodal dataset"]
GRANT = {"funder": "European Commission (Horizon Europe, MSCA Doctoral Networks)", "number": "101073357", "acronym": "MuSIC"}
LOCATION = {"place": "Technical University of Denmark, Kongens Lyngby, Denmark", "lat": 55.786, "lon": 12.522}
RELATED = [
    ("https://doi.org/10.17504/protocols.io.m7vznm25rvx1/v1", "IsDocumentedBy", "Protocol on protocols.io (SYNAPSE experimental procedure)"),
    ("https://doi.org/10.1016/j.buildenv.2026.115068", "References", "Journal article (same field-labs, same cafeteria-babble stimulus; Dave et al. 2026)"),
]



def orcid_ok(orcid: str) -> bool:
    """ISO 7064 MOD 11-2 check digit of an ORCID iD."""
    digits = orcid.replace("-", "")
    total = 0
    for ch in digits[:-1]:
        total = (total + int(ch)) * 2
    check = (12 - total % 11) % 11
    return digits[-1] == ("X" if check == 10 else str(check))


def doi() -> str:
    return decisions()["D27_doi"]["value"].split(" ")[0]


def citation_cff() -> str:
    lines = ["cff-version: 1.2.0", "message: \"If you use this dataset, please cite it as below.\"", "type: dataset",
             f"title: \"{TITLE}\"", f"version: \"{__version__}\"", f"doi: \"{doi()}\"",
             "license: CC-BY-4.0", "authors:"]
    for c in CREATORS:
        lines += [f"  - family-names: \"{c['family']}\"", f"    given-names: \"{c['given']}\"",
                  f"    orcid: \"https://orcid.org/{c['orcid']}\"", f"    affiliation: \"{'; '.join(c['affiliations'])}\""]
    lines += ["keywords:"] + [f"  - \"{k}\"" for k in KEYWORDS]
    lines += ["references:",
              "  - type: generic",
              "    title: \"Multimodal assessment of cognitive performance, physiology and perceived indoor environment under warm and cool LED lighting and urban soundscapes (SYNAPSE)\"",
              "    authors:", "      - family-names: \"Tomar\"", "        given-names: \"Puneet\"",
              "    publisher:", "      name: \"protocols.io\"", "    year: 2026", "    doi: \"10.17504/protocols.io.m7vznm25rvx1/v1\"",
              "  - type: software", "    title: \"SYNAPSE experiment platform\"",
              "    authors:", "      - family-names: \"Tomar\"", "        given-names: \"Puneet\"",
              "    publisher:", "      name: \"Zenodo\"", "    year: 2026", "    doi: \"10.5281/zenodo.23768059\"", "    url: \"https://synapse.ptomar.com\""]
    return "\n".join(lines) + "\n"
