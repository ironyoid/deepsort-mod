#set page(numbering: "1", margin: 2.2cm)
#set text(size: 11pt)
#set heading(numbering: "1.")
#set par(justify: true)

#align(center)[
  #text(16pt, weight: "bold")[Final Project. Modernizing Deep SORT.]
  #v(4pt)
  #text(11pt)[Deep learning for computer vision]
  #v(2pt)
  #text(10pt)[HSE, 2025/2026]
]

#v(8pt)

= What I did

For this project I took the original Deep SORT tracker and swapped its two main parts — the person
detector and the appearance model — for newer ones. I kept the original tracker code itself
unchanged and just built around it, so I could plug different detectors and re-ID models in and out
and compare them. After that I tuned the parameters, added a separate identity database (the extra
task), and tried a segmentation model.

I measure how good the tracking is with HOTA, averaged over the six test videos, using TrackEval (the
standard MOT evaluation tool). The original Deep SORT gets 0.40. My best setup gets 0.57, and it beats
the original on every single video.

#figure(image("figures/progression.png", width: 68%), caption: [How the average HOTA went up as I improved the system.])

= Datasets

I use six MOTChallenge videos: TUD-Campus, TUD-Stadtmitte, KITTI-17, PETS09-S2L1, MOT16-09 and MOT16-11.

I use three kinds of measurement:
- For the whole tracker: HOTA (averaged over the videos). HOTA mixes two things — how good the boxes
  are (DetA) and how consistent the IDs are (AssA). I evaluate MOT15 and MOT16 with their own standard
  settings and average the per-video HOTA.
- For detectors on their own: Precision, Recall and F1 against the ground-truth boxes.
- For re-ID models on their own: clustering scores on ground-truth crops (Silhouette,
  Calinski-Harabasz, Fowlkes-Mallows), and HOTA using ground-truth boxes so the detector doesn't get
  in the way.

= How the system is put together

Every frame goes through three steps: a detector finds the people (boxes), a re-ID model turns each
box into a 512-number "fingerprint", and the original Deep SORT tracker assigns the IDs. I pick the detector and re-ID model by name before running, so I
can test any combination.

= Baseline

First I measured the original with it's `mars-small128` appearance model. This is what I have to beat.

#figure(
  table(
    columns: 7,
    align: center,
    table.header([], [TUD-Campus], [TUD-Stadt.], [KITTI-17], [PETS09], [MOT16-09], [MOT16-11]),
    [HOTA], [0.3986], [0.3675], [0.4341], [0.4484], [0.3624], [0.3995],
  ),
  caption: [Original Deep SORT, HOTA per video. Average = 0.4017.],
)

= Choosing a detector

I tried three detectors: YOLOv8n and RT-DETR (from Ultralytics) and Faster R-CNN (from torchvision).
That's three models from two different libraries. YOLO came out best, both on detection quality and on
the final HOTA.

#figure(
  table(
    columns: 5,
    align: center,
    table.header([Detector], [Library], [Precision], [Recall], [F1]),
    [yolov8n], [Ultralytics], [0.849], [0.728], [0.784],
    [rtdetr], [Ultralytics], [0.645], [0.784], [0.708],
    [fasterrcnn], [torchvision], [0.510], [0.765], [0.612],
  ),
  caption: [Detector quality against the ground truth (confidence 0.3).],
)

#figure(
  image("figures/detectors.png", width: 68%),
  caption: [Detection F1 and the final HOTA for each detector (with osnet_x0_5, confidence 0.3).],
)

RT-DETR and Faster R-CNN look weak here, but that's mostly because at confidence 0.3 they produce a lot
of false boxes. If I raise the confidence to 0.5, RT-DETR jumps from 0.475 to 0.540 HOTA (about the same
as YOLO) and Faster R-CNN goes to 0.466. So the confidence threshold really matters and is worth
tuning per detector.

= Choosing a re-ID model

I tried four OSNet models from Torchreid (`osnet_x0_25`, `osnet_x0_5`, `osnet_x1_0`,
`osnet_ain_x1_0`). They are trained on large person datasets, which helps them work on videos they
haven't seen.

#figure(
  table(
    columns: 5,
    align: center,
    table.header([Re-ID model], [HOTA], [Silhouette], [Calinski-H], [Fowlkes-M]),
    [osnet_x0_25], [0.5335], [0.2594], [124.6], [0.6122],
    [osnet_x0_5], [0.5397], [0.2744], [123.8], [0.6406],
    [osnet_x1_0], [0.5323], [0.2821], [130.8], [0.6228],
    [osnet_ain], [0.5289], [0.2796], [126.8], [0.6380],
  ),
  caption: [Re-ID models: HOTA (yolov8n, confidence 0.3) and clustering scores on ground-truth crops.],
)

#figure(image("figures/reid.png", width: 62%), caption: [Re-ID HOTA (bars) next to the Silhouette score (line).])

One thing surprised me: the metrics don't agree. `osnet_x1_0` has the most separated fingerprints
(best Silhouette) and wins when I give the tracker perfect boxes, but `osnet_x0_5` wins in the real
pipeline with noisy boxes. When I feed in ground-truth boxes (so only the re-ID matters), HOTA is
around 0.85 for all four models — much higher than the ~0.55 of the full system. So the detector, not
the re-ID model, is what's really holding the system back.

= Tuning the parameters

I swept the three parameters that matter most. Each one has a sweet spot: if the appearance matching
is too strict the IDs break apart, if it's too loose different people get merged, and a larger
`max_age` helps a track survive being hidden, but only up to a point.

#figure(image("figures/params.png", width: 95%), caption: [Changing one parameter at a time (yolov8n + osnet_x0_5).])

Since the detector is the main bottleneck, I switched to the bigger `yolov8s` and `osnet_x1_0` and ran
a grid search. The best single setting was confidence 0.5, max-cosine-distance 0.3, max-age 60, giving
HOTA 0.5712. The task also allows different parameters per video, which pushes the average to 0.5796 —
busy, long videos like MOT16-11 prefer keeping more detections (lower confidence), while calmer videos
do better with a higher confidence.

#figure(
  table(
    columns: 4,
    align: center,
    table.header([Video], [Baseline], [Best (per-video)], [Settings]),
    [TUD-Campus], [0.3986], [0.5205], [conf 0.5 / mcd 0.4],
    [TUD-Stadtmitte], [0.3675], [0.6353], [conf 0.5 / mcd 0.3 / age 60],
    [KITTI-17], [0.4341], [0.6037], [conf 0.5 / mcd 0.4 / age 30],
    [PETS09-S2L1], [0.4484], [0.6099], [conf 0.5 / mcd 0.2 / age 30],
    [MOT16-09], [0.3624], [0.5519], [conf 0.5 / mcd 0.3 / age 90],
    [MOT16-11], [0.3995], [0.5565], [conf 0.3 / mcd 0.2 / age 60],
  ),
  caption: [Per-video baseline vs my best settings. Averages: 0.4017 vs 0.5796.],
)

#figure(image("figures/per_video.png", width: 78%), caption: [HOTA for each video: baseline vs best.])

= Extra task: an identity database

For the extra task I added a separate "identity" layer on top of the tracker. The idea is simple: a
track ID is short-lived (if someone disappears for a while and comes back, the tracker gives them a new
ID), but an identity is meant to be permanent. I keep a small database of identities so that a person
who leaves and returns gets recognized as the same person again.

Each frame I detect, get the fingerprints, run the normal tracking step, and then for every active
track I look up its fingerprint in the database with nearest-centroid kNN (cosine distance with a
threshold). If it's close enough to a known identity I use that one, otherwise I create a new identity.
Each track keeps a short history of identity guesses, and over a time window I take the majority vote.
If two tracks end up claiming the same identity at the same time (which can't be right) I reset them. I
reuse the tracker's OSNet fingerprints and store each identity as the running average of its
fingerprints.

#figure(image("figures/identity.png", width: 88%), caption: [Tuning the database threshold and the voting window.])

Same story as before: if the threshold is too strict I get too many identities, too loose and I merge
people, and a longer voting window is more stable. The best setting (threshold 0.35, window 180) gives
an identity HOTA of 0.5348, and the number of identities per video looks reasonable (about 10–12 for
the TUD videos, 19 for PETS09). This is a bit lower than the track-ID HOTA, which makes sense — keeping
a global identity is harder than keeping a short-term track ID — but the database does something the
plain tracker can't: it recognizes people again after they've been gone.

= Segmentation

I also added a segmentation model (YOLOv8-seg) that I can switch in instead of a normal detector. It
gives a mask for each person; I turn the mask into a box for the tracker, and I can also use the mask
to black out the background of each crop before the re-ID step.

#figure(
  table(
    columns: 4,
    align: center,
    table.header([Setup], [Detection F1], [HOTA], [Note]),
    [yolov8s + osnet_x1_0], [0.78], [0.5712], [reference (boxes)],
    [yolov8s-seg + osnet_x1_0], [0.80], [0.5703], [boxes from masks],
    [yolov8s-seg + masked re-ID], [0.80], [0.5548], [background removed],
  ),
  caption: [Segmentation results.],
)

The segmentation model works about as well as the normal detector (HOTA 0.5703, slightly higher
detection F1). But masking the background for re-ID actually made things a little worse. I think this
is because OSNet was trained on normal crops that include the background, so giving it a black
background confuses it more than it helps. Not the result I expected, but a useful one.

= Conclusion

Overall, swapping in modern models and tuning them raised the average HOTA from 0.4017 to 0.5712 with
one setting, or 0.5796 with per-video settings — and it beats the original on every video while still
running in real time.

A few takeaways:
- My best setup is yolov8s + osnet_x1_0 with confidence 0.5, max-cosine-distance 0.3, max-age 60.
- The detector matters most. With perfect boxes the HOTA is around 0.85, so the detector is the main
  limit, not the re-ID model.
- The clustering scores and the actual HOTA don't always agree, so it's worth checking both before
  picking a re-ID model.
- The identity database (extra task) adds long-term recognition, and the segmentation model also works
  in the pipeline.

= How to reproduce

The six videos go under `data/sequences/`; the model weights download automatically the first time.
Main commands:

```
# baseline
python tools/generate_detections.py --model=resources/networks/mars-small128.pb \
    --mot_dir=data/sequences --output_dir=data/detections
python evaluate_motchallenge.py --mot_dir=data/sequences --detection_dir=data/detections \
    --output_dir=results/baseline --min_confidence=0.3 --nn_budget=100
python eval_hota.py --results_dir=results/baseline

# best setup
python run_tracker.py --mot_dir=data/sequences --output_dir=results/best \
    --detector=yolov8s --reid=osnet_x1_0 --min_confidence=0.5 --max_cosine_distance=0.3 --max_age=60

# evaluating parts, tuning, identity database, segmentation
python eval_detector.py --mot_dir=data/sequences --detector=yolov8n --min_confidence=0.3
python eval_reid.py --mot_dir=data/sequences --reid=osnet_x1_0
python tune.py --mot_dir=data/sequences --detector=yolov8s --reid=osnet_x1_0 \
    --min_confidence=0.3,0.4,0.5 --max_cosine_distance=0.2,0.3,0.4 --max_age=30,60,90
python run_identity.py --mot_dir=data/sequences --output_dir=results/identity \
    --detector=yolov8s --reid=osnet_x1_0 --threshold=0.35 --window=180
python run_tracker.py --mot_dir=data/sequences --output_dir=results/seg \
    --detector=yolov8s-seg --reid=osnet_x1_0 --mask
```
