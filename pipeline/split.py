"""Stage 2: put every WEBPAGE (not every sketch) into train, val or test.
    python -m pipeline.split
All sketches of one webpage end up in the same split, so the model is never tested on a page it trained on.
Our own photos always go to test (real-world check). Writes data/split/webpages.csv and reports/split.json."""
import json
import random

import pandas as pd
import yaml

from shared import config

MANIFEST = config.ROOT / "data" / "validated" / "manifest.csv"
OUT = config.ROOT / "data" / "split" / "webpages.csv"
REPORT = config.ROOT / "reports" / "split.json"


def assign(webpages, seed, val, test):
    """Shuffle the webpage ids with a fixed seed and cut them into train/val/test -> {webpage_id: split}."""
    pages = sorted(webpages)
    random.Random(seed).shuffle(pages)
    n_test, n_val = round(len(pages) * test), round(len(pages) * val)
    return {w: "test" if i < n_test else "val" if i < n_test + n_val else "train" for i, w in enumerate(pages)}


def check_no_leakage(m):
    """Every webpage in exactly one split, and no identical image in two splits."""
    per_page = m.groupby("webpage_id")["split"].nunique()
    assert (per_page == 1).all(), f"webpages in several splits: {list(per_page[per_page > 1].index)}"
    per_image = m.groupby("sketch_md5")["split"].nunique()
    assert (per_image == 1).all(), "the same sketch image is in two splits"


def main():
    p = yaml.safe_load((config.ROOT / "params.yaml").read_text())["split"]
    m = pd.read_csv(MANIFEST, dtype=str)
    s2c = m.loc[m["source"] == "sketch2code", "webpage_id"].unique()
    split = assign(s2c, p["seed"], p["val"], p["test"])
    split.update({w: "test" for w in m.loc[m["source"] == "own", "webpage_id"].unique()})
    m["split"] = m["webpage_id"].map(split)
    check_no_leakage(m)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    m[["webpage_id", "split"]].drop_duplicates().sort_values("webpage_id").to_csv(OUT, index=False)
    report = {s: dict(webpages=int(g["webpage_id"].nunique()), sketches=len(g)) for s, g in m.groupby("split")}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == "__main__":
    main()
