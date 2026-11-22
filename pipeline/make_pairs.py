"""Stage 3: build pairs.csv = (sketch, screenshot, label) rows for the scorer to learn from.
    python -m pipeline.make_pairs
Three kinds of pair:
  reference  sketch + the real screenshot of its own webpage            label 1 (automatic)
  mismatch   sketch + the real screenshot of ANOTHER webpage            label 0 (automatic)
             the other webpage is picked from the SAME split, so no screenshot is shared between splits
  agent      sketch + a page our agent built (data/renders)             label from data/labels/render_labels.csv
             (only rounds you labelled; added once renders and labels exist)
Paths are relative to data/. Writes data/pairs/pairs.csv and reports/pairs.json."""
import json
import random

import pandas as pd
import yaml

from shared import config

DATA = config.ROOT / "data"
MANIFEST = DATA / "validated" / "manifest.csv"
SPLIT = DATA / "split" / "webpages.csv"
RENDERS = DATA / "renders" / "renders.csv"
LABELS = DATA / "labels" / "render_labels.csv"
OUT = DATA / "pairs" / "pairs.csv"
REPORT = config.ROOT / "reports" / "pairs.json"


def _row(s, kind, render_path, label, key):
    return dict(pair_id=f"{s.sketch_id}__{kind}__{key}", webpage_id=s.webpage_id, sketch_id=s.sketch_id,
                sketch_path=f"raw/{s.sketch_path}", render_path=render_path, source=s.source,
                kind=kind, label=int(label), split=s.split)


def auto_pairs(m, seed, n_mismatch):
    rng = random.Random(seed)
    refs = m[m["reference_png"] != ""].drop_duplicates("webpage_id")
    rows = []
    for s in m.itertuples():
        if s.reference_png:
            rows.append(_row(s, "reference", f"raw/{s.reference_png}", 1, s.webpage_id))
        others = sorted(refs.loc[(refs["split"] == s.split) & (refs["webpage_id"] != s.webpage_id), "webpage_id"])
        for w in rng.sample(others, min(n_mismatch, len(others))):
            rows.append(_row(s, "mismatch", f"raw/{refs.set_index('webpage_id').at[w, 'reference_png']}", 0, w))
    return rows


def agent_pairs(m):
    if not (RENDERS.exists() and LABELS.exists()):
        return []
    r = pd.read_csv(RENDERS, dtype=str)[["sketch_id", "round", "render_png"]]
    r = r.merge(pd.read_csv(LABELS, dtype=str)[["sketch_id", "round", "label"]], on=["sketch_id", "round"])
    r = r[r["label"].isin(["0", "1"])].merge(m, on="sketch_id")  # skip rows marked unsure
    return [_row(x, "agent", f"renders/{x.render_png}", x.label, f"r{x.round}") for x in r.itertuples()]


def main():
    p = yaml.safe_load((config.ROOT / "params.yaml").read_text())["pairs"]
    m = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False).merge(pd.read_csv(SPLIT, dtype=str), on="webpage_id")
    pairs = pd.DataFrame(auto_pairs(m, p["seed"], p["mismatches_per_sketch"]) + agent_pairs(m))

    assert pairs["pair_id"].is_unique
    # leakage check: a screenshot of a real webpage may only appear in that webpage's split
    ref = pairs[pairs["kind"] != "agent"].copy()
    ref["shot_split"] = ref["render_path"].map(
        pairs[pairs["kind"] == "reference"].drop_duplicates("render_path").set_index("render_path")["split"])
    assert (ref["shot_split"].isna() | (ref["shot_split"] == ref["split"])).all(), "screenshot shared across splits"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT, index=False)
    report = pairs.groupby(["split", "kind"])["label"].agg(["count", "mean"]).round(3)
    REPORT.write_text(json.dumps({f"{s}/{k}": dict(n=int(r["count"]), pass_rate=r["mean"])
                                  for (s, k), r in report.iterrows()}, indent=2))
    print(report)


if __name__ == "__main__":
    main()
