import argparse
import glob
import os

import cv2
import numpy as np

from deep_sort.iou_matching import iou
from detectors import build_detector

MOT16 = {"MOT16-09", "MOT16-11"}


def load_gt(seq_dir, name):
    gt = np.loadtxt(os.path.join(seq_dir, "gt", "gt.txt"), delimiter=",")
    if name in MOT16:
        gt = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1)]
    frames = {}
    for row in gt:
        frames.setdefault(int(row[0]), []).append(row[2:6])
    return {f: np.array(v, dtype=np.float32) for f, v in frames.items()}


def count_sequence(seq_dir, name, detector, iou_threshold):
    gt = load_gt(seq_dir, name)
    tp = fp = fn = 0
    for path in sorted(glob.glob(os.path.join(seq_dir, "img1", "*.jpg"))):
        frame_idx = int(os.path.splitext(os.path.basename(path))[0])
        dets = detector.detect(cv2.imread(path))[:, :4]
        gts = gt.get(frame_idx, np.empty((0, 4), dtype=np.float32))
        matched = set()
        for d in dets:
            best, bj = 0.0, -1
            for j in range(len(gts)):
                if j in matched:
                    continue
                value = iou(d, gts[j:j + 1])[0]
                if value > best:
                    best, bj = value, j
            if bj >= 0 and best >= iou_threshold:
                tp += 1
                matched.add(bj)
            else:
                fp += 1
        fn += len(gts) - len(matched)
    return tp, fp, fn


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mot_dir", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--min_confidence", type=float, default=0.3)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    args = parser.parse_args()

    detector = build_detector(args.detector, args.min_confidence)
    total = [0, 0, 0]
    print("%-16s %9s %9s %9s" % ("sequence", "Precision", "Recall", "F1"))
    for name in sorted(os.listdir(args.mot_dir)):
        tp, fp, fn = count_sequence(os.path.join(args.mot_dir, name), name, detector, args.iou_threshold)
        total = [total[0] + tp, total[1] + fp, total[2] + fn]
        p, r, f1 = prf(tp, fp, fn)
        print("%-16s %9.4f %9.4f %9.4f" % (name, p, r, f1))
    p, r, f1 = prf(*total)
    print("%-16s %9.4f %9.4f %9.4f" % ("OVERALL", p, r, f1))


if __name__ == "__main__":
    main()
