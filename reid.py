import os

import cv2
import gdown
import numpy as np
import torch
from torchreid.reid.utils import FeatureExtractor

_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{**k, "weights_only": False})

WEIGHTS = {
    "osnet": ("osnet_x0_25", "1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e"),
    "osnet_x0_5": ("osnet_x0_5", "1DHgmb6XV4fwG3n-CnCM0zdL9nMsZ9_RF"),
    "osnet_x1_0": ("osnet_x1_0", "1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x"),
    "osnet_ain": ("osnet_ain_x1_0", "1nIrszJVYSHf3Ej8-j6DTFdWz8EnO42PB"),
}


class OsnetReid:
    def __init__(self, name, model_name, file_id):
        path = os.path.join("resources/networks", name + ".pth")
        if not os.path.exists(path):
            os.makedirs("resources/networks", exist_ok=True)
            gdown.download(id=file_id, output=path, quiet=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = FeatureExtractor(model_name=model_name, model_path=path, device=device)

    def extract(self, image, boxes, masks=None):
        if len(boxes) == 0:
            return np.empty((0, 512), dtype=np.float32)
        crops = []
        for i, (x, y, w, h) in enumerate(boxes[:, :4]):
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = int(x + w), int(y + h)
            crop = image[y1:y2, x1:x2]
            if masks is not None:
                crop = crop * masks[i][y1:y2, x1:x2, None]
            if crop.size == 0:
                crop = np.zeros((1, 1, 3), dtype=np.uint8)
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return self.extractor(crops).cpu().numpy()


def build_reid(name):
    model_name, file_id = WEIGHTS[name]
    return OsnetReid(name, model_name, file_id)
