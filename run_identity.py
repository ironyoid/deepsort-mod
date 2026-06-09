import argparse
import glob
import os
from collections import Counter

import cv2
import numpy as np

from application_util import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from detectors import build_detector
from eval_hota import evaluate
from identity import IdentityDatabase
from reid import build_reid


def vote(entries, frame_idx, window):
    recent = [ident for frame, ident in entries if frame > frame_idx - window]
    return Counter(recent).most_common(1)[0][0] if recent else None


def run(sequence_dir, output_file, detector, reid, db, window,
        max_cosine_distance, nn_budget, max_age, n_init, nms_max_overlap):
    images = sorted(glob.glob(os.path.join(sequence_dir, "img1", "*.jpg")))
    frames = {int(os.path.splitext(os.path.basename(p))[0]): p for p in images}
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=max_age, n_init=n_init)
    history = {}
    results = []
    for frame_idx in sorted(frames):
        image = cv2.imread(frames[frame_idx])
        boxes = detector.detect(image)
        features = reid.extract(image, boxes)
        detections = [Detection(boxes[i, :4], boxes[i, 4], features[i]) for i in range(len(boxes))]
        bboxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(bboxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        tracker.predict()
        tracker.update(detections)

        active = [t for t in tracker.tracks if t.is_confirmed() and t.time_since_update == 0]
        active = [t for t in active if t.track_id in metric.samples]
        for track in active:
            ident = db.query(metric.samples[track.track_id][-1])
            history.setdefault(track.track_id, []).append((frame_idx, ident))

        resolved = {track.track_id: vote(history[track.track_id], frame_idx, window) for track in active}

        owners = {}
        for tid, ident in resolved.items():
            owners.setdefault(ident, []).append(tid)
        for ident, tids in owners.items():
            if len(tids) > 1:
                for tid in tids:
                    history[tid] = []
                    resolved[tid] = None

        for track in active:
            ident = resolved[track.track_id]
            if ident is None:
                continue
            x, y, w, h = track.to_tlwh()
            results.append([frame_idx, ident, x, y, w, h])

    with open(output_file, "w") as f:
        for row in results:
            f.write("%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n" % tuple(row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mot_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--reid", required=True)
    parser.add_argument("--min_confidence", type=float, default=0.5)
    parser.add_argument("--nms_max_overlap", type=float, default=1.0)
    parser.add_argument("--max_cosine_distance", type=float, default=0.3)
    parser.add_argument("--nn_budget", type=int, default=100)
    parser.add_argument("--max_age", type=int, default=60)
    parser.add_argument("--n_init", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--window", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    detector = build_detector(args.detector, args.min_confidence)
    reid = build_reid(args.reid)
    for sequence in sorted(os.listdir(args.mot_dir)):
        db = IdentityDatabase(args.threshold)
        output_file = os.path.join(args.output_dir, sequence + ".txt")
        run(os.path.join(args.mot_dir, sequence), output_file, detector, reid, db, args.window,
            args.max_cosine_distance, args.nn_budget, args.max_age, args.n_init, args.nms_max_overlap)
        print("done", sequence)

    print(evaluate(args.output_dir, args.mot_dir))


if __name__ == "__main__":
    main()
