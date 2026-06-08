import numpy as np
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, weights, conf):
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, image):
        result = self.model.predict(image, classes=[0], conf=self.conf, verbose=False)[0]
        boxes = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append([x1, y1, x2 - x1, y2 - y1, float(box.conf[0])])
        return np.array(boxes, dtype=np.float32).reshape(-1, 5)


DETECTORS = {
    "yolov8n": lambda conf: YoloDetector("yolov8n.pt", conf),
    "yolov8s": lambda conf: YoloDetector("yolov8s.pt", conf),
}


def build_detector(name, conf):
    return DETECTORS[name](conf)
