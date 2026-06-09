import argparse
import contextlib
import io
import itertools
import os

from detectors import build_detector
from eval_hota import compute
from reid import build_reid
from run_tracker import run


def floats(text):
    return [float(x) for x in text.split(",")]


def ints(text):
    return [int(x) for x in text.split(",")]


def run_all(mot_dir, out_dir, detector, reid, mcd, max_age, n_init, nn_budget, nms):
    os.makedirs(out_dir, exist_ok=True)
    for seq in sorted(os.listdir(mot_dir)):
        run(os.path.join(mot_dir, seq), os.path.join(out_dir, seq + ".txt"),
            detector, reid, nms, mcd, nn_budget, max_age, n_init)
    with contextlib.redirect_stdout(io.StringIO()):
        return compute(out_dir, mot_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mot_dir", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--reid", required=True)
    parser.add_argument("--min_confidence", type=floats, default=[0.5])
    parser.add_argument("--max_cosine_distance", type=floats, default=[0.3])
    parser.add_argument("--max_age", type=ints, default=[60])
    parser.add_argument("--n_init", type=int, default=3)
    parser.add_argument("--nn_budget", type=int, default=100)
    parser.add_argument("--nms_max_overlap", type=float, default=1.0)
    parser.add_argument("--output", default="results/tune.csv")
    args = parser.parse_args()

    reid = build_reid(args.reid)
    sequences = sorted(os.listdir(args.mot_dir))
    rows = []
    best = ("", -1.0)
    per_video = {seq: ("", -1.0) for seq in sequences}

    for conf in args.min_confidence:
        detector = build_detector(args.detector, conf)
        for mcd, max_age in itertools.product(args.max_cosine_distance, args.max_age):
            scores = run_all(args.mot_dir, "results/_tune_tmp", detector, reid,
                             mcd, max_age, args.n_init, args.nn_budget, args.nms_max_overlap)
            label = "conf=%.2f mcd=%.2f age=%d" % (conf, mcd, max_age)
            rows.append((label, scores))
            print("%-26s HOTA %.4f" % (label, scores["AVERAGE"]))
            if scores["AVERAGE"] > best[1]:
                best = (label, scores["AVERAGE"])
            for seq in sequences:
                if scores[seq] > per_video[seq][1]:
                    per_video[seq] = (label, scores[seq])

    print("\nBEST GLOBAL: %s -> %.4f" % best)
    print("PER-VIDEO BEST:")
    pv_avg = sum(per_video[seq][1] for seq in sequences) / len(sequences)
    for seq in sequences:
        print("  %-16s %.4f (%s)" % (seq, per_video[seq][1], per_video[seq][0]))
    print("  PER-VIDEO AVERAGE %.4f" % pv_avg)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("config," + ",".join(sequences) + ",AVERAGE\n")
        for label, scores in rows:
            f.write(label + "," + ",".join("%.4f" % scores[seq] for seq in sequences))
            f.write(",%.4f\n" % scores["AVERAGE"])


if __name__ == "__main__":
    main()
