# DeepSORT Modification

Original Deep SORT with swappable, modern detectors and re-ID models. Beats the original on the MOT test videos.

## The two archives

- `deepsort-mod.zip` — the code (this repository, zipped). Upload it in Colab.
- `sequences.zip` — the six test videos in MOT format (frames + ground truth), ready to use. Keep it in Google Drive.

## Run on Colab

1. Put `sequences.zip` in your Google Drive (`My Drive`).
2. Open `notebooks/baseline.ipynb` in Colab and set the runtime to **GPU**.
3. **Run all**. Upload `deepsort-mod.zip` when the first cell asks.

The notebook runs the baseline, the best setup, the identity database and segmentation, and prints HOTA for each.

## Run locally

Setup once:
```
python -m venv .venv && . .venv/bin/activate
pip install numpy scipy opencv-python scikit-learn gdown tensorflow torch torchvision ultralytics torchreid
pip install git+https://github.com/JonathonLuiten/TrackEval.git
```

Put the six sequences under `data/sequences/<name>/` (or unzip `sequences.zip` into `data/`), then:
```
# best setup
python run_tracker.py --mot_dir=data/sequences --output_dir=results/best \
    --detector=yolov8s --reid=osnet_x1_0 --min_confidence=0.5 --max_cosine_distance=0.3 --max_age=60

# identity database
python run_identity.py --mot_dir=data/sequences --output_dir=results/identity \
    --detector=yolov8s --reid=osnet_x1_0 --threshold=0.35 --window=180

# overlay videos
python generate_videos.py --mot_dir=data/sequences --result_dir=results/best --output_dir=results/videos/best
```

Detectors: `yolov8n`, `yolov8s`, `rtdetr`, `fasterrcnn`, `yolov8s-seg`.
Re-ID: `osnet`, `osnet_x0_5`, `osnet_x1_0`, `osnet_ain`.

