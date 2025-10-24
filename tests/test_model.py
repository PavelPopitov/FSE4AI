from tinyvision.model import predict_topk
from PIL import Image

def test_predict_top5_types():
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    preds = predict_topk(img, k=5)
    assert len(preds) == 5
    assert isinstance(preds[0][0], str)
    assert isinstance(preds[0][1], float)
