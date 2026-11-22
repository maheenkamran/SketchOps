"""pipeline/validate.py must pass clean data and catch each kind of broken data."""
import json

import pytest
from PIL import Image

from pipeline import validate


@pytest.fixture
def raw(tmp_path, monkeypatch):
    """A tiny clean dataset: 2 webpages, 3 sketches, laid out like data/raw/."""
    s2c = tmp_path / "raw" / "sketch2code"
    (s2c / "sketches").mkdir(parents=True)
    (s2c / "webpages").mkdir()
    rows = ["webpage_id,sketch_id,sketch_path,reference_png,reference_html"]
    for w, n in ((1, 2), (2, 1)):
        Image.new("RGB", (1280, 900), "white").save(s2c / "webpages" / f"{w}.png")
        (s2c / "webpages" / f"{w}.html").write_text("<html><body>hi</body></html>")
        for i in range(n):
            Image.new("RGB", (600, 400 + 10 * w + i), "white").save(s2c / "sketches" / f"{w}_{i}.png")  # sizes differ -> no duplicates
            rows.append(f"{w},{w}_{i},sketch2code/sketches/{w}_{i}.png,"
                        f"sketch2code/webpages/{w}.png,sketch2code/webpages/{w}.html")
    (s2c / "manifest.csv").write_text("\n".join(rows) + "\n")
    monkeypatch.setattr(validate, "RAW", tmp_path / "raw")
    monkeypatch.setattr(validate, "OUT", tmp_path / "validated" / "manifest.csv")
    monkeypatch.setattr(validate, "REPORT", tmp_path / "validate.json")
    return tmp_path / "raw"


def _errors(raw):
    with pytest.raises(SystemExit):
        validate.main()
    assert not validate.OUT.exists()  # a failed run leaves no manifest behind
    return " ".join(json.loads(validate.REPORT.read_text())["errors"])


def test_clean_data_passes(raw):
    (raw / "own").mkdir()
    Image.new("RGB", (800, 600), "white").save(raw / "own" / "cafe.jpg")
    validate.main()
    m = validate.pd.read_csv(validate.OUT)
    assert len(m) == 4 and m["webpage_id"].nunique() == 3
    assert set(m["source"]) == {"sketch2code", "own"}
    assert "own_cafe" in set(m["webpage_id"]) and "s2c_1" in set(m["webpage_id"])


def test_missing_file(raw):
    (raw / "sketch2code" / "sketches" / "1_0.png").unlink()
    assert "file missing" in _errors(raw)


def test_corrupt_image(raw):
    (raw / "sketch2code" / "sketches" / "1_0.png").write_bytes(b"not a png")
    assert "cannot open" in _errors(raw)


def test_too_small_and_transparent(raw):
    Image.new("RGB", (100, 80)).save(raw / "sketch2code" / "sketches" / "1_0.png")
    Image.new("RGBA", (600, 400), (0, 0, 0, 0)).save(raw / "sketch2code" / "sketches" / "1_1.png")
    err = _errors(raw)
    assert "too small" in err and "transparent" in err


def test_duplicate_image_across_webpages(raw):
    s = raw / "sketch2code" / "sketches"
    (s / "2_0.png").write_bytes((s / "1_0.png").read_bytes())
    assert "identical images" in _errors(raw)


def test_mobile_reference_and_heic(raw):
    Image.new("RGB", (390, 900), "white").save(raw / "sketch2code" / "webpages" / "2.png")
    (raw / "own").mkdir()
    (raw / "own" / "IMG_001.HEIC").write_bytes(b"x")
    err = _errors(raw)
    assert "not a desktop page" in err and "HEIC" in err
