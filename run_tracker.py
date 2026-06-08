import argparse
import glob
import os
import time

import cv2
import numpy as np

from application_util import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from detectors import build_detector
from reid import build_reid
from eval_hota import evaluate


def run(sequence_dir, output_file, detector, reid, nms_max_overlap,
        max_cosine_distance, nn_budget):
    images = sorted(glob.glob(os.path.join(sequence_dir, "img1", "*.jpg")))
    frames = {int(os.path.splitext(os.path.basename(p))[0]): p for p in images}
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []
    elapsed = 0.0
    for frame_idx in sorted(frames):
        image = cv2.imread(frames[frame_idx])
        start = time.time()
        boxes = detector.detect(image)
        features = reid.extract(image, boxes)
        detections = [Detection(boxes[i, :4], boxes[i, 4], features[i]) for i in range(len(boxes))]
        bboxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(bboxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        tracker.predict()
        tracker.update(detections)
        elapsed += time.time() - start
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x, y, w, h = track.to_tlwh()
            results.append([frame_idx, track.track_id, x, y, w, h])
    with open(output_file, "w") as f:
        for row in results:
            f.write("%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n" % tuple(row))
    return len(frames) / elapsed if elapsed else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mot_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--reid", required=True)
    parser.add_argument("--min_confidence", type=float, default=0.3)
    parser.add_argument("--nms_max_overlap", type=float, default=1.0)
    parser.add_argument("--max_cosine_distance", type=float, default=0.2)
    parser.add_argument("--nn_budget", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    detector = build_detector(args.detector, args.min_confidence)
    reid = build_reid(args.reid)
    for sequence in sorted(os.listdir(args.mot_dir)):
        sequence_dir = os.path.join(args.mot_dir, sequence)
        output_file = os.path.join(args.output_dir, sequence + ".txt")
        fps = run(sequence_dir, output_file, detector, reid, args.nms_max_overlap,
                  args.max_cosine_distance, args.nn_budget)
        print("%s: %.1f FPS" % (sequence, fps))

    print(evaluate(args.output_dir, args.mot_dir))


if __name__ == "__main__":
    main()
