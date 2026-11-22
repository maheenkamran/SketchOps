"""Download a fixed subset of the Sketch2Code dataset (license ODC-By, see data/raw/sketch2code/SOURCE.md).
    python -m pipeline.download_sketch2code
Same seed + same dataset commit = the same files on every machine.
Output: data/raw/sketch2code/{sketches,webpages}/ and manifest.csv (one row per sketch)."""
import csv
import random
import shutil
from collections import defaultdict

from huggingface_hub import HfApi, snapshot_download

from shared import config

REPO = "SALT-NLP/Sketch2Code"
REVISION = "ce28bf4e62f9242cde5c5beaa8775efccfa8502b"  # pinned dataset commit
N_WEBPAGES = 120
SEED = 42
OUT = config.ROOT / "data" / "raw" / "sketch2code"


def pick_subset():
    """All sketches of N_WEBPAGES randomly chosen webpages -> {webpage_id: [sketch file, ...]}."""
    files = HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION)
    pages = defaultdict(list)
    for f in files:
        if f.startswith("sketches/") and f.endswith(".png"):  # sketches/<webpage_id>_<sketch_id>.png
            pages[f.split("/")[1].split("_")[0]].append(f)
    chosen = random.Random(SEED).sample(sorted(pages), N_WEBPAGES)
    return {w: sorted(pages[w]) for w in sorted(chosen, key=int)}


def main():
    subset = pick_subset()
    wanted = ["rick.jpg"]  # placeholder image used by the webpages' HTML
    for w, sketches in subset.items():
        wanted += sketches + [f"webpages/{w}.png", f"webpages/{w}.html"]
    snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=OUT, allow_patterns=wanted)
    shutil.rmtree(OUT / ".cache", ignore_errors=True)  # download bookkeeping, not data

    with open(OUT / "manifest.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["webpage_id", "sketch_id", "sketch_path", "reference_png", "reference_html"])
        for w, sketches in subset.items():
            for s in sketches:
                sid = s.split("_")[-1].removesuffix(".png")
                out.writerow([w, f"{w}_{sid}", f"sketch2code/{s}",
                              f"sketch2code/webpages/{w}.png", f"sketch2code/webpages/{w}.html"])
    n = sum(map(len, subset.values()))
    print(f"{len(subset)} webpages, {n} sketches -> {OUT}")


if __name__ == "__main__":
    main()
