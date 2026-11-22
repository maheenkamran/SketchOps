"""Stage 1 of dvc.yaml: check the raw data before anything uses it.
    python -m pipeline.validate
Reads  data/raw/sketch2code/ (manifest.csv + images) and data/raw/own/ (our photos, optional).
Writes data/validated/manifest.csv (one row per sketch, both sources) and reports/validate.json.
Every problem is collected and listed; if there is any, the stage fails (exit 1) and writes no manifest."""
import hashlib
import json
import sys

import pandas as pd
from PIL import Image

from shared import config

RAW = config.ROOT / "data" / "raw"
OUT = config.ROOT / "data" / "validated" / "manifest.csv"
REPORT = config.ROOT / "reports" / "validate.json"

COLUMNS = ["webpage_id", "sketch_id", "sketch_path", "reference_png", "reference_html"]
PHOTO_TYPES = {".jpg", ".jpeg", ".png"}
MIN_SKETCH_SIDE = 300  # px; smaller is unreadable for the LLM and SigLIP
MIN_REFERENCE_WIDTH = 1024  # references must be desktop pages (we render at 1280)


def check_image(path, errors, min_side=0, min_width=0):
    """Open the image fully and check its size and transparency. Returns (w, h) or None."""
    try:
        with Image.open(path) as im:
            im.load()  # decodes every pixel, so truncated files fail here
            w, h = im.size
            if "A" in im.getbands() and im.getchannel("A").getextrema()[0] < 255:
                errors.append(f"{path.name}: has transparent pixels (would turn black when featurized)")
    except Exception as e:
        errors.append(f"{path.name}: cannot open image ({e})")
        return None
    if min(w, h) < min_side:
        errors.append(f"{path.name}: too small {w}x{h} (shortest side < {min_side}px)")
    if w < min_width:
        errors.append(f"{path.name}: width {w} < {min_width}px, not a desktop page")
    return w, h


def load_sketch2code(errors):
    folder = RAW / "sketch2code"
    m = pd.read_csv(folder / "manifest.csv", dtype=str)
    if list(m.columns) != COLUMNS:
        errors.append(f"sketch2code manifest columns are {list(m.columns)}, expected {COLUMNS}")
        return pd.DataFrame()
    if m.isna().any().any():
        errors.append(f"sketch2code manifest has empty cells in rows {list(m.index[m.isna().any(axis=1)])}")
        m = m.dropna()
    for r in m.itertuples():
        if not r.sketch_id.startswith(f"{r.webpage_id}_"):
            errors.append(f"{r.sketch_id}: does not belong to webpage {r.webpage_id}")
    for ref in m.drop_duplicates("webpage_id").itertuples():  # each webpage once
        if check_image(RAW / ref.reference_png, errors, min_width=MIN_REFERENCE_WIDTH) is None:
            continue
        html = RAW / ref.reference_html
        if not html.exists() or "<body" not in html.read_text(errors="ignore").lower():
            errors.append(f"{html.name}: missing or has no <body>")
    m["webpage_id"] = "s2c_" + m["webpage_id"]  # prefix so ids can never clash with our own photos
    m["sketch_id"] = "s2c_" + m["sketch_id"]
    m["source"] = "sketch2code"
    return m


def load_own(errors):
    """Our own photos: data/raw/own/<name>.jpg. Each photo is its own webpage, with no reference page."""
    folder = RAW / "own"
    rows = []
    for p in sorted(folder.glob("*")) if folder.exists() else []:
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in PHOTO_TYPES:
            errors.append(f"{p.name}: unsupported type {p.suffix} (convert HEIC to JPG)")
            continue
        rows.append(dict(webpage_id=f"own_{p.stem}", sketch_id=f"own_{p.stem}_0",
                         sketch_path=f"own/{p.name}", reference_png="", reference_html="", source="own"))
    return pd.DataFrame(rows)


def main():
    errors = []
    m = pd.concat([load_sketch2code(errors), load_own(errors)], ignore_index=True)

    sizes, hashes = [], []
    for r in m.itertuples():
        path = RAW / r.sketch_path
        size = check_image(path, errors, min_side=MIN_SKETCH_SIDE) if path.exists() else None
        if not path.exists():
            errors.append(f"{r.sketch_path}: file missing")
        sizes.append(size or (0, 0))
        hashes.append(hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else "")
    m["width"], m["height"] = [s[0] for s in sizes], [s[1] for s in sizes]
    m["sketch_md5"] = hashes

    if m["sketch_id"].duplicated().any():
        errors.append(f"duplicate sketch_id: {sorted(m.loc[m['sketch_id'].duplicated(), 'sketch_id'])}")
    dup = m[(m["sketch_md5"] != "") & m["sketch_md5"].duplicated(keep=False)]
    for _, g in dup.groupby("sketch_md5"):  # same image twice -> could leak between train and test
        errors.append(f"identical images: {list(g['sketch_id'])}")

    report = dict(ok=not errors, n_sketches=len(m), n_webpages=m["webpage_id"].nunique(),
                  by_source=m["source"].value_counts().to_dict(), n_errors=len(errors), errors=errors)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))

    if errors:
        print(f"FAILED: {len(errors)} problem(s)", *errors[:20], sep="\n  ")
        OUT.unlink(missing_ok=True)  # never leave an old manifest that looks valid
        sys.exit(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT, index=False)
    print(f"OK: {len(m)} sketches from {report['n_webpages']} webpages {report['by_source']} -> {OUT}")


if __name__ == "__main__":
    main()
