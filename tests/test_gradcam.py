from tinyvision.model import get_model_and_device, preprocess
from tinyvision.gradcam import GradCAM
from PIL import Image
import numpy as np

def test_gradcam_basic():
    model, _ = get_model_and_device()
    target_layer = model.features[-1]
    gc = GradCAM(model, target_layer)
    img = Image.new("RGB", (64, 64), color=(180, 180, 180))
    t = preprocess(img)
    cam, cls = gc(t)
    assert isinstance(cls, int)
    assert isinstance(cam, np.ndarray)
    assert cam.ndim == 2
    assert 0.0 <= float(cam.min()) <= 1.0
    assert 0.0 <= float(cam.max()) <= 1.0
