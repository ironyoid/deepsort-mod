import argparse
import os

import numpy as np
import trackeval

np.float = float
np.int = int

GROUPS = {
    "MOT15": ["TUD-Campus", "TUD-Stadtmitte", "KITTI-17", "PETS09-S2L1"],
    "MOT16": ["MOT16-09", "MOT16-11"],
}

EVAL_CONFIG = {
    "USE_PARALLEL": False,
    "PRINT_RESULTS": False,
    "PRINT_CONFIG": False,
    "TIME_PROGRESS": False,
    "OUTPUT_SUMMARY": False,
    "OUTPUT_DETAILED": False,
    "PLOT_CURVES": False,
}


def evaluate_group(benchmark, seqs, gt_dir, trackers_dir, tracker):
    config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    config.update({
        "GT_FOLDER": gt_dir,
        "TRACKERS_FOLDER": trackers_dir,
        "TRACKER_SUB_FOLDER": "",
        "BENCHMARK": benchmark,
        "SKIP_SPLIT_FOL": True,
        "TRACKERS_TO_EVAL": [tracker],
        "SEQ_INFO": {seq: None for seq in seqs},
        "PRINT_CONFIG": False,
    })
    dataset = trackeval.datasets.MotChallenge2DBox(config)
    res, _ = trackeval.Evaluator(EVAL_CONFIG).evaluate([dataset], [trackeval.metrics.HOTA()])
    data = res["MotChallenge2DBox"][tracker]
    return {seq: float(np.mean(data[seq]["pedestrian"]["HOTA"]["HOTA"])) for seq in seqs}


def evaluate(results_dir, gt_dir):
    trackers_dir = os.path.dirname(results_dir)
    tracker = os.path.basename(results_dir)

    scores = {}
    for benchmark, seqs in GROUPS.items():
        scores.update(evaluate_group(benchmark, seqs, gt_dir, trackers_dir, tracker))
    scores["AVERAGE"] = sum(scores.values()) / len(scores)

    report = "\n".join("%-16s %.4f" % (seq, hota) for seq, hota in scores.items())
    with open(os.path.join(results_dir, "hota.txt"), "w") as f:
        f.write(report + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--gt_dir", default="data/sequences")
    args = parser.parse_args()
    print(evaluate(args.results_dir, args.gt_dir))


if __name__ == "__main__":
    main()
