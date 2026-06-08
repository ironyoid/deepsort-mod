import numpy as np
import torch
import torchvision
from ultralytics import RTDETR, YOLO


class UltralyticsDetector:
    def __init__(self, model, conf):
        self.model = model
        self.conf = conf

    def detect(self, image):
        result = self.model.predict(image, classes=[0], conf=self.conf, verbose=False)[0]
        boxes = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append([x1, y1, x2 - x1, y2 - y1, float(box.conf[0])])
        return np.array(boxes, dtype=np.float32).reshape(-1, 5)


class TorchvisionDetector:
    def __init__(self, conf):
        self.conf = conf
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights="DEFAULT").eval().to(self.device)

    def detect(self, image):
        rgb = image[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device)
        with torch.no_grad():
            out = self.model([tensor])[0]
        boxes = []
        for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
            if int(label) == 1 and float(score) >= self.conf:
                x1, y1, x2, y2 = box.tolist()
                boxes.append([x1, y1, x2 - x1, y2 - y1, float(score)])
        return np.array(boxes, dtype=np.float32).reshape(-1, 5)


DETECTORS = {
    "yolov8n": lambda conf: UltralyticsDetector(YOLO("yolov8n.pt"), conf),
    "yolov8s": lambda conf: UltralyticsDetector(YOLO("yolov8s.pt"), conf),
    "rtdetr": lambda conf: UltralyticsDetector(RTDETR("rtdetr-l.pt"), conf),
    "fasterrcnn": lambda conf: TorchvisionDetector(conf),
}


def build_detector(name, conf):
    return DETECTORS[name](conf)
