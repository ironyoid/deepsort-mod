import argparse
import os

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, fowlkes_mallows_score, silhouette_score
from sklearn.preprocessing import normalize

from reid import build_reid

MOT16 = {"MOT16-09", "MOT16-11"}


def sequence_features(seq_dir, name, reid):
    gt = np.loadtxt(os.path.join(seq_dir, "gt", "gt.txt"), delimiter=",")
    if name in MOT16:
        gt = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1)]
    by_frame = {}
    for row in gt:
        by_frame.setdefault(int(row[0]), []).append((int(row[1]), row[2:6]))
    features, labels = [], []
    for frame_idx, items in by_frame.items():
        image = cv2.imread(os.path.join(seq_dir, "img1", "%06d.jpg" % frame_idx))
        boxes = np.array([b for _, b in items], dtype=np.float32)
        features.append(reid.extract(image, boxes))
        labels.extend(tid for tid, _ in items)
    return np.concatenate(features), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mot_dir", required=True)
    parser.add_argument("--reid", required=True)
    args = parser.parse_args()

    reid = build_reid(args.reid)
    sils, chs, fms = [], [], []
    print("%-16s %12s %14s %14s" % ("sequence", "Silhouette", "Calinski-H", "Fowlkes-M"))
    for name in sorted(os.listdir(args.mot_dir)):
        features, labels = sequence_features(os.path.join(args.mot_dir, name), name, reid)
        sil = silhouette_score(features, labels, metric="cosine")
        ch = calinski_harabasz_score(features, labels)
        clusters = KMeans(n_clusters=len(set(labels)), n_init=3, random_state=0).fit_predict(normalize(features))
        fm = fowlkes_mallows_score(labels, clusters)
        sils.append(sil)
        chs.append(ch)
        fms.append(fm)
        print("%-16s %12.4f %14.1f %14.4f" % (name, sil, ch, fm))
    print("%-16s %12.4f %14.1f %14.4f" % (
        "AVERAGE", sum(sils) / len(sils), sum(chs) / len(chs), sum(fms) / len(fms)))


if __name__ == "__main__":
    main()
