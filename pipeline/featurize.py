"""Stage 4: turn every (sketch, screenshot) pair into numbers the models can learn from.
    python -m pipeline.featurize
Features per pair (all compare the sketch with the screenshot):
  siglip_sim       cosine similarity of frozen SigLIP embeddings (what the image "looks like" overall)
  ssim             structural similarity of the two images in grey, same size (pixel-level baseline)
  aspect_diff      |log of (screenshot height/width) / (sketch height/width)|: 0 = same page length
  row_corr         correlation of the edge profile from top to bottom (are the sections at the same heights?)
  col_corr         correlation of the edge profile from left to right (same columns?)
  grid_corr        correlation of a grid x grid map of edge density (same layout blocks?)
Edges are used for the layout features because a pencil sketch and a coloured page share outlines, not colours.
Writes data/features/features.csv (pair info + features + label + split)
   and data/features/embeddings.npz (SigLIP vectors of both images, for the neural-network model)."""
import io

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy import ndimage

from shared import config

DATA = config.ROOT / "data"
PAIRS = DATA / "pairs" / "pairs.csv"
OUT = DATA / "features" / "features.csv"
EMB = DATA / "features" / "embeddings.npz"
FEATURES = ["siglip_sim", "ssim", "aspect_diff", "row_corr", "col_corr", "grid_corr"]


def load_grey(path, side):
    """Open as white-backed RGB, then grey, shrunk to `side` px wide (keeps the aspect ratio). Values 0..1."""
    im = Image.open(path)
    w, h = im.size
    bg = Image.new("RGB", im.size, "white")
    bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").getchannel("A"))
    g = bg.convert("L").resize((side, max(1, round(h * side / w))), Image.BILINEAR)
    return np.asarray(g, dtype=np.float64) / 255.0, h / w


def edges(g):
    """Edge strength (Sobel), scaled so each image's strongest edges are ~1."""
    e = np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))
    return e / (np.percentile(e, 99) + 1e-8)


def ssim(a, b):
    """Standard SSIM (Gaussian window, sigma 1.5) of two same-size grey images in 0..1."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    f = lambda x: ndimage.gaussian_filter(x, 1.5)
    ma, mb = f(a), f(b)
    va, vb, cov = f(a * a) - ma ** 2, f(b * b) - mb ** 2, f(a * b) - ma * mb
    return float(np.mean((2 * ma * mb + c1) * (2 * cov + c2) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))))


def _resample(v, n):
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v)), v)


def _corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def _grid(e, n):
    """Average edge strength in an n x n grid over the whole page."""
    rows = np.array_split(e, n, axis=0)
    return np.array([[c.mean() for c in np.array_split(r, n, axis=1)] for r in rows])


def layout_features(sketch, shot, p):
    """All non-SigLIP features for one pair (both arguments are file paths)."""
    gs, ars = load_grey(sketch, p["side"])
    gr, arr = load_grey(shot, p["side"])
    es, er = edges(gs), edges(gr)
    sq = lambda g: np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((p["side"], p["side"])),
                              dtype=np.float64) / 255.0
    return dict(
        ssim=ssim(sq(gs), sq(gr)),
        aspect_diff=float(abs(np.log(arr / ars))),
        row_corr=_corr(_resample(es.mean(1), p["bins"]), _resample(er.mean(1), p["bins"])),
        col_corr=_corr(_resample(es.mean(0), p["bins"]), _resample(er.mean(0), p["bins"])),
        grid_corr=_corr(_grid(es, p["grid"]), _grid(er, p["grid"])),
    )


def siglip_embeddings(paths, batch=16):
    """{path: unit-length SigLIP vector}, each distinct image embedded once."""
    from scoring.clip_score import embed

    out = {}
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        vecs = embed([(DATA / x).read_bytes() for x in chunk]).numpy().astype(np.float32)
        out.update(zip(chunk, vecs))
        print(f"  SigLIP {min(i + batch, len(paths))}/{len(paths)} images", end="\r")
    print()
    return out


def main():
    p = yaml.safe_load((config.ROOT / "params.yaml").read_text())["featurize"]
    pairs = pd.read_csv(PAIRS, dtype={"webpage_id": str})
    emb = siglip_embeddings(sorted(set(pairs["sketch_path"]) | set(pairs["render_path"])))

    rows = []
    for i, r in enumerate(pairs.itertuples(), 1):
        f = layout_features(DATA / r.sketch_path, DATA / r.render_path, p)
        f["siglip_sim"] = float(emb[r.sketch_path] @ emb[r.render_path])
        rows.append(f)
        if i % 50 == 0 or i == len(pairs):
            print(f"  layout features {i}/{len(pairs)}", end="\r")
    print()
    feats = pd.concat([pairs, pd.DataFrame(rows)[FEATURES]], axis=1)
    assert feats[FEATURES].notna().all().all(), "a feature came out as NaN"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUT, index=False)
    np.savez_compressed(EMB, pair_id=pairs["pair_id"].to_numpy(),
                        sketch=np.stack([emb[x] for x in pairs["sketch_path"]]),
                        render=np.stack([emb[x] for x in pairs["render_path"]]))
    print(feats.groupby(["kind", "label"])[FEATURES].mean().round(3).to_string())


if __name__ == "__main__":
    main()
