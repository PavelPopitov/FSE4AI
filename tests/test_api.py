from fastapi.testclient import TestClient
from tinyvision.app import app
from PIL import Image
import io
import base64

client = TestClient(app)

def _png_bytes(w=64, h=64):
    img = Image.new("RGB", (w, h), color=(200, 200, 200))
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()

def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

def test_predict_top5():
    files = {"file": ("x.png", _png_bytes(), "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    js = r.json()
    assert "predictions" in js and len(js["predictions"]) == 5

def test_predict_with_gradcam():
    files = {"file": ("x.png", _png_bytes(), "image/png")}
    r = client.post("/predict?gradcam=true", files=files)
    assert r.status_code == 200
    js = r.json()
    assert "gradcam_png_b64" in js
    png = base64.b64decode(js["gradcam_png_b64"])
    assert png[:8] == b'\x89PNG\r\n\x1a\n'
