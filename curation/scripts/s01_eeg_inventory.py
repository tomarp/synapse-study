"""Step 01: hash and de-duplicate every raw EEG file (cache in the work folder, config/paths.toml), write the inventory."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synapse_curation import eeg, paths  # noqa: E402

man = eeg.inventory()
paths.ensure(paths.WORK)
man.to_csv(paths.WORK / "eeg_inventory.tsv", sep="\t", index=False)
print(man.drop(columns=["sha256"]).to_string())
