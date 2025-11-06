Goal
Create an initial fall-detection starter that fuses MMPose 2D pose with time-series models (LSTM, InceptionTime) for fall detection from videos. Provide dataset helpers (URFD/Le2i), keypoint extraction, training/eval, and video inference with visualization. License: MIT.

What’s included
- README with full setup and usage
- MIT LICENSE, .gitignore, requirements.txt
- Scripts
  - scripts/make_video_csv.py: build train/val/test CSVs from falls/non_falls folders
  - scripts/download_urfd.py: URFD download/structure helper (supports --url/--zip)
  - scripts/download_le2i.py: Le2i download/structure helper (supports --url/--zip)
- Tools
  - tools/pose_extract.py: run MMPose to export per-video keypoints (npz)
- Source
  - src/utils/feature_engineering.py: normalization, velocity, feature builder
  - src/datasets/fall_dataset.py: sliding-window sequence dataset
  - src/models/lstm_fall.py: LSTM baseline
  - src/models/inception_time.py: InceptionTime baseline
  - src/train.py: train LSTM/InceptionTime, save best checkpoint with config
  - src/eval.py: evaluate checkpoint (report + AUC)
  - src/infer_video.py: on-the-fly pose + sliding prediction + smoothing + video overlay

Quickstart
1) Environment
   - python -m venv .venv && source .venv/bin/activate
   - pip install -r requirements.txt

2) Datasets
   - Prepare videos under data/raw/<dataset>/{falls,non_falls}
   - Optional helpers:
     - python scripts/download_urfd.py --out data/raw/urfd [--url <zip> | --zip <local.zip>]
     - python scripts/download_le2i.py --out data/raw/le2i [--url <zip> | --zip <local.zip>]
   - Build CSVs:
     - python scripts/make_video_csv.py --root data/raw/urfd --out datasets --val-ratio 0.15 --test-ratio 0.15

3) Keypoints (MMPose)
   - python tools/pose_extract.py --csv datasets/train.csv --out-root data/processed --pose-model rtmpose-s
   - Repeat for val/test

4) Train
   - LSTM:
     python -m src.train --csv-train datasets/train.csv --csv-val datasets/val.csv --npz-root data/processed --model lstm --seq-len 64 --seq-step 16 --batch-size 64 --epochs 30 --lr 1e-3 --hidden 128 --layers 2 --dropout 0.2 --use-vel --use-conf --norm shoulder --class-weight 1.0
   - InceptionTime:
     python -m src.train --csv-train datasets/train.csv --csv-val datasets/val.csv --npz-root data/processed --model inception --seq-len 128 --seq-step 16 --batch-size 64 --epochs 40 --lr 1e-3 --it-filters 32 --it-depth 6 --use-vel --use-conf --norm shoulder --class-weight 1.0

5) Evaluate
   - python -m src.eval --csv-test datasets/test.csv --npz-root data/processed --checkpoint runs/<ts>/best.pt --seq-len 64 --seq-step 16 --use-vel --use-conf --norm shoulder

6) Inference
   - python -m src.infer_video --video /path/to/video.mp4 --checkpoint runs/<ts>/best.pt --pose-model rtmpose-s --seq-len 64 --seq-step 8 --use-vel --use-conf --norm shoulder --out-vis outputs/result.mp4 --smooth-k 5 --alarm-th 0.6

Notes
- Normalization matters (shoulder/hip/bbox). Include score features and velocities for robustness.
- If dataset links require license agreement, use the scripts with --zip after manual download.
- For multi-person scenes, add a tracker to get per-track sequences.

Follow-ups (optional in next PRs)
- Minimal CI: dependency install + import self-check
- Multi-person tracking (ByteTrack/DeepSORT) integration
- More baselines (TCN/Transformer), focal loss/oversampling, subject-wise splits