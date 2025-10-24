from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import numpy as np
from .ui import build_ui

from .model import predict_topk, preprocess, get_model_and_device, LABELS
from .gradcam import GradCAM

app = FastAPI(title="TinyVision", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/labels")
def labels():
    return {"classes": LABELS}

def _heatmap_overlay_rgba(cam: np.ndarray, base_img: Image.Image) -> bytes:
    """
    cam: np.array [H, W] в [0,1]
    base_img: PIL RGB
    return PNG imgs with RGBA heatmap overlay
    """
    cam_img = Image.fromarray((cam * 255).astype("uint8"), "L").resize(base_img.size)
    arr = np.asarray(cam_img, dtype=np.float32) / 255.0
    r = (arr * 255.0)
    g = (np.clip(arr * 1.5, 0, 1) * 255.0)
    b = np.zeros_like(r)
    a = (np.clip(arr * 0.6 + 0.2, 0, 1) * 255.0)
    rgba = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
    heat_rgba = Image.fromarray(rgba, "RGBA")
    over = Image.alpha_composite(base_img.convert("RGBA"), heat_rgba)
    buf = io.BytesIO()
    over.save(buf, format="PNG")
    return buf.getvalue()

@app.post("/predict")
async def predict(file: UploadFile = File(...), gradcam: bool = False):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image")

    preds = predict_topk(img, k=5)
    out = {"predictions": [{"label": label, "prob": prob} for label, prob in preds]}

    if gradcam:
        model, _ = get_model_and_device()
        target_layer = model.features[-1]
        gc = GradCAM(model, target_layer)
        t = preprocess(img)
        cam, cls_idx = gc(t)
        overlay_png = _heatmap_overlay_rgba(cam, img)
        out["gradcam_png_b64"] = base64.b64encode(overlay_png).decode()
        out["gradcam_for_class"] = cls_idx

    return JSONResponse(out)

app = build_ui(app)