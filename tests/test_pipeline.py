"""split / make_pairs / featurize: no leakage between splits, and features that behave sensibly."""
import pandas as pd
import pytest
from PIL import Image, ImageDraw

from pipeline import featurize, make_pairs, split


# ---- split ----
def test_split_is_by_webpage_and_repeatable():
    pages = [str(i) for i in range(100)]
    a = split.assign(pages, 42, 0.15, 0.15)
    assert a == split.assign(pages, 42, 0.15, 0.15)  # same seed -> same split
    counts = pd.Series(a).value_counts()
    assert counts["train"] == 70 and counts["val"] == 15 and counts["test"] == 15


def test_split_leakage_check_fails_on_shared_page():
    m = pd.DataFrame(dict(webpage_id=["1", "1"], sketch_md5=["a", "b"], split=["train", "test"]))
    with pytest.raises(AssertionError, match="several splits"):
        split.check_no_leakage(m)


# ---- pairs ----
def test_mismatch_partner_comes_from_same_split():
    m = pd.DataFrame(dict(
        webpage_id=["1", "2", "3", "4"], sketch_id=["1_0", "2_0", "3_0", "4_0"],
        sketch_path=[f"s/{i}.png" for i in "1234"], reference_png=[f"w/{i}.png" for i in "1234"],
        source="sketch2code", split=["train", "train", "test", "test"]))
    rows = pd.DataFrame(make_pairs.auto_pairs(m, seed=0, n_mismatch=1))
    assert len(rows) == 8 and set(rows.groupby("kind")["label"].first()) == {0, 1}
    mis = rows[rows["kind"] == "mismatch"].set_index("sketch_id")["render_path"]
    assert mis["1_0"] == "raw/w/2.png" and mis["3_0"] == "raw/w/4.png"  # only possible same-split partners


# ---- features ----
def _page(path, boxes, size=(600, 900)):
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    for b in boxes:
        d.rectangle(b, outline="black", width=4)
    im.save(path)
    return path


P = dict(side=128, bins=32, grid=8)


def test_identical_images_score_perfectly(tmp_path):
    a = _page(tmp_path / "a.png", [(20, 20, 580, 100), (20, 150, 580, 500)])
    f = featurize.layout_features(a, a, P)
    assert f["ssim"] == pytest.approx(1.0) and f["aspect_diff"] == pytest.approx(0.0)
    assert f["row_corr"] == pytest.approx(1.0) and f["grid_corr"] == pytest.approx(1.0)


def test_same_layout_beats_different_layout(tmp_path):
    sketch = _page(tmp_path / "s.png", [(20, 20, 580, 100), (20, 150, 580, 500), (20, 550, 580, 880)])
    same = _page(tmp_path / "same.png", [(30, 25, 570, 105), (30, 160, 570, 510), (30, 555, 570, 870)])
    other = _page(tmp_path / "o.png", [(20, 20, 280, 880), (320, 20, 580, 880)], size=(600, 1500))
    good, bad = featurize.layout_features(sketch, same, P), featurize.layout_features(sketch, other, P)
    for k in ("row_corr", "grid_corr"):
        assert good[k] > bad[k]
    assert good["aspect_diff"] < bad["aspect_diff"]


def test_transparent_png_is_read_as_white(tmp_path):
    Image.new("RGBA", (100, 100), (0, 0, 0, 0)).save(tmp_path / "t.png")
    g, _ = featurize.load_grey(tmp_path / "t.png", 64)
    assert g.min() == pytest.approx(1.0)
