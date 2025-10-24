from pathlib import Path
from typing import List, Tuple

import pytest
from PIL import Image

from tinyvision.model import predict_topk

ASSETS_DIR = Path(__file__).resolve().parents[1] / "tinyvision" / "assets"

EXPECTED_KEYWORDS = {
    "absolute_cinema.png": ["book jacket", "comic book", "bow tie"],
    "goldfish.JPG": ["goldfish", "puffer", "tench"],
    "horse.jpeg": ["sorrel", "redbone", "ox", "horse"],
    "dog.jpg": ["vizsla", "rhodesian ridgeback", "redbone", "dog"],
}


def _open_rgb(path: Path) -> Image.Image:
    """Open image as RGB"""
    with Image.open(path) as im:
        return im.convert("RGB")


@pytest.mark.parametrize("fname", sorted(EXPECTED_KEYWORDS.keys()))
def test_real_images_top_prediction(fname: str):
    """
    Check that the model correctly processes real images from tinyvision/assets
    and that at least one expected keyword is present in the top-3 predictions.
    """
    img_path = ASSETS_DIR / fname
    assert img_path.exists(), f"file not found: {img_path}"

    pil_img = _open_rgb(img_path)

    topk: List[Tuple[str, float]] = predict_topk(pil_img, k=3)
    assert isinstance(topk, list) and len(topk) == 3, f"Expected top-3, got: {topk}"

    labels = [lbl.lower() for (lbl, _p) in topk]
    probs = [float(p) for (_l, p) in topk]

    assert all(0.0 <= p <= 1.0 for p in probs), f"Probs out of range [0,1]: {probs}"
    assert probs == sorted(probs, reverse=True), f"Top-3 not sorted by prob: {probs}"
    assert all(isinstance(lbl, str) and lbl for lbl in labels), f"Bad labels: {labels}"

    expected = [kw.lower() for kw in EXPECTED_KEYWORDS[fname]]

    if expected:
        matched = any(any(kw in lbl for kw in expected) for lbl in labels)
        assert matched, (
            f"Expected one of {expected} in top-3 for {fname}, "
            f"got labels={labels} topk={topk}"
        )
    else:
        assert True


def test_assets_directory_structure():
    """Check that all required files are present in the repo."""
    missing = [name for name in EXPECTED_KEYWORDS if not (ASSETS_DIR / name).exists()]
    assert not missing, f"file not found: {missing}"
