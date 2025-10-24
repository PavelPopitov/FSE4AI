import gradio as gr
from PIL import Image
import io
import base64

def build_ui(fastapi_app):
    def infer(image: Image.Image, show_gradcam: bool):
        import httpx
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        files = {"file": ("img.png", buf.getvalue(), "image/png")}
        params = {"gradcam": "true" if show_gradcam else "false"}
        with httpx.Client() as client:
            r = client.post("http://localhost:8000/predict", files=files, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
        text = "\n".join([f"{i+1}. {p['label']} — {p['prob']:.3f}" for i, p in enumerate(data["predictions"])])
        grad = None
        if show_gradcam and "gradcam_png_b64" in data:
            grad = Image.open(io.BytesIO(base64.b64decode(data["gradcam_png_b64"])))
        return text, grad

    with gr.Blocks() as demo:
        gr.Markdown("# TinyVision — MobileNetV3 + Grad-CAM")
        with gr.Row():
            img = gr.Image(type="pil", label="Изображение", height=320)
            chk = gr.Checkbox(label="Показать Grad-CAM", value=True)
        btn = gr.Button("Классифицировать")
        out_text = gr.Textbox(label="Top-5 (label — prob)")
        out_img = gr.Image(label="Grad-CAM overlay")
        btn.click(infer, [img, chk], [out_text, out_img])

    # доступно в gradio>=4: монтирование UI внутрь FastAPI
    gr.mount_gradio_app(fastapi_app, demo, path="/ui")
    return fastapi_app
