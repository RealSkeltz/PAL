"""Image embeddings for recognizing an object again: DINOv2 on a crop of it."""
import cv2
import numpy as np

from pal.config import VISION_MODELS
from pal.perception.vision.detect import device

DINO = VISION_MODELS / "dinov2-base"  # facebook/dinov2-base, saved locally so loading needs no network

_DINO = None


def _embedder():
    """Load DINOv2 on first use."""
    global _DINO
    if _DINO is None:
        from transformers import AutoImageProcessor, AutoModel
        _DINO = (AutoImageProcessor.from_pretrained(DINO),
                 AutoModel.from_pretrained(DINO).to(device() or "cpu").eval())
    return _DINO


def embed(crops: list[np.ndarray]) -> np.ndarray:
    """One L2-normalized vector per crop (BGR, as cropped from the frame). The dot product
    of two vectors is their cosine similarity."""
    import torch
    proc, model = _embedder()
    images = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]
    with torch.no_grad():
        out = model(**proc(images=images, return_tensors="pt").to(model.device))
    return torch.nn.functional.normalize(out.last_hidden_state[:, 0], dim=-1).cpu().numpy()


def crop(img, box, pad=0.1):
    """The box's region of `img`, with `pad` (fraction of the box size) around it."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
    x2, y2 = min(w, int(x2 + dx)), min(h, int(y2 + dy))
    return img[y1:y2, x1:x2].copy()
