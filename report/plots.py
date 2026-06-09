import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

SEQS = ["TUD-Campus", "TUD-Stadtmitte", "KITTI-17", "PETS09-S2L1", "MOT16-09", "MOT16-11"]


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=150)
    plt.close()


def progression():
    labels = ["Baseline", "Modular\n(default)", "Tuned\n(global)", "Per-video"]
    values = [0.4017, 0.5397, 0.5712, 0.5796]
    plt.figure(figsize=(5, 3.2))
    bars = plt.bar(labels, values, color=["#999999", "#4c9be8", "#2f7ed8", "#1a4f8a"])
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.005, "%.4f" % value, ha="center", fontsize=9)
    plt.ylabel("HOTA (avg)")
    plt.ylim(0, 0.65)
    plt.title("Average HOTA progression")
    save("progression.png")


def detectors():
    names = ["yolov8n", "rtdetr", "fasterrcnn"]
    f1 = [0.7835, 0.7076, 0.6119]
    hota = [0.5335, 0.4890, 0.4315]
    x = range(len(names))
    plt.figure(figsize=(5, 3.2))
    plt.bar([i - 0.2 for i in x], f1, 0.4, label="Detection F1", color="#7bb6f0")
    plt.bar([i + 0.2 for i in x], hota, 0.4, label="HOTA (+osnet)", color="#1a4f8a")
    plt.xticks(list(x), names)
    plt.ylim(0, 0.9)
    plt.legend()
    plt.title("Detectors: detection F1 vs tracking HOTA (conf 0.3)")
    save("detectors.png")


def reid():
    names = ["osnet", "osnet_x0_5", "osnet_x1_0", "osnet_ain"]
    hota = [0.5335, 0.5397, 0.5323, 0.5289]
    sil = [0.2594, 0.2744, 0.2821, 0.2796]
    fig, ax1 = plt.subplots(figsize=(5.2, 3.2))
    ax1.bar(names, hota, color="#1a4f8a")
    ax1.set_ylabel("HOTA")
    ax1.set_ylim(0.5, 0.55)
    ax2 = ax1.twinx()
    ax2.plot(names, sil, "o-", color="#e8731a")
    ax2.set_ylabel("Silhouette")
    ax1.set_title("REID: tracking HOTA (bars) vs clustering Silhouette (line)")
    plt.setp(ax1.get_xticklabels(), rotation=15)
    save("reid.png")


def params():
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].plot([0.2, 0.3, 0.4, 0.5], [0.5207, 0.5397, 0.5317, 0.5410], "o-")
    axes[0].set_title("min_confidence")
    axes[1].plot([0.1, 0.2, 0.3, 0.4], [0.5050, 0.5397, 0.5375, 0.5164], "o-")
    axes[1].set_title("max_cosine_distance")
    axes[2].plot([15, 30, 60, 90], [0.5334, 0.5397, 0.5421, 0.5398], "o-")
    axes[2].set_title("max_age")
    for ax in axes:
        ax.set_ylabel("HOTA")
        ax.grid(alpha=0.3)
    fig.suptitle("Parameter evolution (yolov8n + osnet_x0_5)")
    save("params.png")


def identity():
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3))
    axes[0].plot([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
                 [0.4675, 0.5155, 0.5220, 0.5238, 0.5298, 0.4739, 0.4298, 0.3795], "o-")
    axes[0].set_title("DB threshold (window 30)")
    axes[0].set_xlabel("threshold")
    axes[1].plot([15, 30, 60, 120], [0.5070, 0.5155, 0.5245, 0.5311], "o-")
    axes[1].set_title("DB window (threshold 0.2)")
    axes[1].set_xlabel("window (frames)")
    for ax in axes:
        ax.set_ylabel("identity HOTA")
        ax.grid(alpha=0.3)
    fig.suptitle("Identity database parameter evolution")
    save("identity.png")


def per_video():
    baseline = [0.3986, 0.3675, 0.4341, 0.4484, 0.3624, 0.3995]
    best = [0.5205, 0.6353, 0.6037, 0.6099, 0.5519, 0.5565]
    x = range(len(SEQS))
    plt.figure(figsize=(7, 3.4))
    plt.bar([i - 0.2 for i in x], baseline, 0.4, label="Baseline", color="#999999")
    plt.bar([i + 0.2 for i in x], best, 0.4, label="Best (per-video)", color="#1a4f8a")
    plt.xticks(list(x), SEQS, rotation=20, ha="right")
    plt.ylabel("HOTA")
    plt.ylim(0, 0.7)
    plt.legend()
    plt.title("Per-video HOTA: baseline vs best")
    save("per_video.png")


progression()
detectors()
reid()
params()
identity()
per_video()
print("figures written to", OUT)
