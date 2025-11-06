# Fall Detection Starter (MMPose + LSTM/InceptionTime/tsai, PyTorch)

This repo provides a ready-to-run pipeline for fall detection from videos by fusing:
- 2D human pose estimation (MMPose) to extract keypoints over time
- A time-series classifier (LSTM / InceptionTime / tsai variants) to detect falls from keypoint sequences

Features:
- Video -> keypoints (npz) via MMPoseInferencer
- Train/val/test on keypoint sequences with sliding windows
- Inference on a new video with on-the-fly pose extraction and smoothed predictions
- Extensible feature engineering (normalization, velocities, confidences)
- Dataset helpers: URFD / Le2i download & structuring scripts

## 1) Environment

Python 3.9+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you're on CUDA, install a matching torch build:
https://pytorch.org/get-started/locally/

MMPose will pull the required MMEngine/MMCV/Mmdet dependencies.

## 2) Datasets

CSV format for splits:

- datasets/train.csv
- datasets/val.csv
- datasets/test.csv

Each CSV row: `video_path,label`, where `label` is 1 for fall, 0 for non-fall.

Generate CSVs from a root folder containing (falls/ non_falls):

```bash
python scripts/make_video_csv.py --root /path/to/root --out datasets --val-ratio 0.15 --test-ratio 0.15
```

### 2.1 URFD / Le2i quick setup

These helper scripts assist downloading (if public links are available) and structuring into:

```
data/raw/<dataset>/{falls,non_falls}/*.mp4
```

Usage:

```bash
# URFD
python scripts/download_urfd.py --out data/raw/urfd
# Le2i
python scripts/download_le2i.py --out data/raw/le2i
```

Notes:
- Some datasets require manual agreement; if a direct public link is unavailable, the script will print instructions and exit gracefully. You can pass `--url` with a direct zip link or `--zip` pointing to a local archive to extract and structure.
- “Fall vs Non-fall” is inferred by file/folder names (contains "fall" -> fall; "adl"/"non"/"no_fall" -> non-fall). Please verify and adjust if needed.

Then create CSVs (example for URFD):

```bash
python scripts/make_video_csv.py --root data/raw/urfd --out datasets --val-ratio 0.15 --test-ratio 0.15
```

## 3) Extract 2D Keypoints

Run MMPose on the CSV-listed videos to produce per-video keypoints (npz):

```bash
python tools/pose_extract.py --csv datasets/train.csv --out-root data/processed --pose-model rtmpose-s
python tools/pose_extract.py --csv datasets/val.csv   --out-root data/processed --pose-model rtmpose-s
python tools/pose_extract.py --csv datasets/test.csv  --out-root data/processed --pose-model rtmpose-s
```

Outputs per video:
- `keypoints`: float32 (T, K, 3) with x, y, score
- `fps`: float
- `meta`: dict with `video_path`, `format` (e.g., "coco-17")

Default: single person per frame (largest bbox). For multi-person scenes, integrate a tracker and persist per-person tracks.

## 4) Train (LSTM / InceptionTime / tsai)

LSTM baseline:

```bash
python -m src.train \
  --csv-train datasets/train.csv \
  --csv-val datasets/val.csv \
  --npz-root data/processed \
  --model lstm \
  --seq-len 64 --seq-step 16 \
  --batch-size 64 --epochs 30 \
  --lr 1e-3 --hidden 128 --layers 2 --dropout 0.2 \
  --use-vel --use-conf --norm shoulder \
  --class-weight 1.0
```

InceptionTime baseline (stronger on many TS tasks):

```bash
python -m src.train \
  --csv-train datasets/train.csv \
  --csv-val datasets/val.csv \
  --npz-root data/processed \
  --model inception \
  --seq-len 128 --seq-step 16 \
  --batch-size 64 --epochs 40 \
  --lr 1e-3 --it-filters 32 --it-depth 6 \
  --use-vel --use-conf --norm shoulder \
  --class-weight 1.0
```

tsai baselines (LSTMPlus, InceptionTime):

```bash
# LSTMPlus (tsai)
python -m src.train \
  --csv-train datasets/train.csv \
  --csv-val datasets/val.csv \
  --npz-root data/processed \
  --model tsai_lstm \
  --seq-len 64 --seq-step 16 \
  --batch-size 64 --epochs 30 \
  --lr 1e-3 \
  --use-vel --use-conf --norm shoulder \
  --class-weight 1.0

# InceptionTime (tsai)
python -m src.train \
  --csv-train datasets/train.csv \
  --csv-val datasets/val.csv \
  --npz-root data/processed \
  --model tsai_inception \
  --seq-len 128 --seq-step 16 \
  --batch-size 64 --epochs 40 \
  --lr 1e-3 \
  --use-vel --use-conf --norm shoulder \
  --class-weight 1.0
```

Checkpoints and logs go to `runs/<timestamp>/`. The checkpoint stores `model_type` and hyperparameters for easy eval/inference.

## 5) Evaluate

```bash
python -m src.eval \
  --csv-test datasets/test.csv \
  --npz-root data/processed \
  --checkpoint runs/<timestamp>/best.pt \
  --seq-len 64 --seq-step 16 \
  --use-vel --use-conf --norm shoulder
```

## 6) Inference on a new video

```bash
python -m src.infer_video \
  --video /path/to/video.mp4 \
  --checkpoint runs/<timestamp>/best.pt \
  --pose-model rtmpose-s \
  --seq-len 64 --seq-step 8 \
  --use-vel --use-conf --norm shoulder \
  --out-vis outputs/result.mp4 \
  --smooth-k 5 --alarm-th 0.6
```

It will:
- Run MMPose on-the-fly
- Compute sliding-window predictions
- Smooth outputs, overlay probability and alarm on frames
- Save the visualization

## 7) Notes and Tips

- Normalization is critical: use shoulder or hip width; fallback to bbox height.
- Handle missing keypoints: include `score` as features; you can add masking/interpolation if needed.
- Class imbalance: tune `--class-weight`, or implement focal loss/oversampling.
- Subject splits (if available) yield better generalization.
- Multi-person: add a tracker (ByteTrack/DeepSORT) to export per-track sequences.

## 8) References & Attribution

- MMPose (OpenMMLab): https://github.com/open-mmlab/mmpose
- RTMPose (MMPose project): https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
- MMDetection / RTMDet: https://github.com/open-mmlab/mmdetection
- tsai (timeseriesAI): https://github.com/timeseriesAI/tsai
- LSTM: Hochreiter & Schmidhuber, "Long Short-Term Memory," 1997
- InceptionTime: H. I. Fawaz et al., "InceptionTime: Finding AlexNet for Time Series Classification," 2019
