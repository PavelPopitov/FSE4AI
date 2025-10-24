from typing import List, Tuple
import torch
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

# _device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
_model = models.mobilenet_v3_small(weights=_weights).to(_device).eval()
_preproc = _weights.transforms()
LABELS: List[str] = _weights.meta["categories"]

def preprocess(pil_img):
    """PIL.Image -> tensor [1,3,224,224] to _device"""
    return _preproc(pil_img).unsqueeze(0).to(_device)

@torch.inference_mode()
def predict_topk(pil_img, k: int = 5) -> List[Tuple[str, float]]:
    """Return topK [(label, prob)]"""
    t = preprocess(pil_img)
    logits = _model(t)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    v, idx = probs.topk(k)
    return [(LABELS[i], float(v[j])) for j, i in enumerate(idx.tolist())]

def get_model_and_device():
    return _model, _device
