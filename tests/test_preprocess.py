from tinyvision.model import preprocess
from PIL import Image

def test_preprocess_shape():
    img = Image.new("RGB", (64, 64), color=(200, 200, 200))
    t = preprocess(img)
    assert t.shape == (1, 3, 224, 224)
