import os

import cv2
import gdown
import numpy as np
import torch
from torchreid.reid.utils import FeatureExtractor

WEIGHTS = {
    "osnet": ("osnet_x0_25", "1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e"),
    "osnet_x0_5": ("osnet_x0_5", "1DHgmb6XV4fwG3n-CnCM0zdL9nMsZ9_RF"),
}


class OsnetReid:
    def __init__(self, model_name, file_id):
        path = os.path.join("resources/networks", model_name + "_msmt17.pth")
        if not os.path.exists(path):
            os.makedirs("resources/networks", exist_ok=True)
            gdown.download(id=file_id, output=path, quiet=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = FeatureExtractor(model_name=model_name, model_path=path, device=device)

    def extract(self, image, boxes):
        if len(boxes) == 0:
            return np.empty((0, 512), dtype=np.float32)
        crops = []
        for x, y, w, h in boxes[:, :4]:
            x1, y1 = max(0, int(x)), max(0, int(y))
            crop = image[y1:int(y + h), x1:int(x + w)]
            if crop.size == 0:
                crop = np.zeros((1, 1, 3), dtype=np.uint8)
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return self.extractor(crops).cpu().numpy()


def build_reid(name):
    model_name, file_id = WEIGHTS[name]
    return OsnetReid(model_name, file_id)
