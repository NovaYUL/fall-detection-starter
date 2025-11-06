#!/usr/bin/env bash
set -euo pipefail

# Ensure we are in repo root
if [ ! -d .git ]; then
  echo "Please run in the root of your cloned repository."
  exit 1
fi

branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$branch" != "feature/fall-starter" ]; then
  echo "Current branch is $branch, expected feature/fall-starter. Continue anyway."
fi

mkdir -p datasets data/processed outputs runs
mkdir -p scripts tools src/{datasets,models,utils}

# Files
cat > .gitignore <<'FILE'
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
ENV/
.DS_Store
Thumbs.db
runs/
outputs/
.data/
cache/
data/processed/
*.egg-info/
.eggs/
datasets/*.csv
FILE

cat > requirements.txt <<'FILE'
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.23.0
pandas>=2.0.0
opencv-python>=4.8.0
tqdm>=4.66.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
PyYAML>=6.0
requests>=2.31.0
# MMPose and dependencies
mmpose>=1.3.0
mmcv>=2.0.1
mmdet>=3.2.0
mmengine>=0.10.0
FILE

cat > LICENSE <<'FILE'
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
FILE

cat > scripts/make_video_csv.py <<'FILE'
import argparse
import csv
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg")

def collect_videos(root: str):
    root = Path(root)
    falls = []
    non_falls = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            rel = str(p)
            lower = str(p.parent).lower()
            if "fall" in lower and "non" not in lower and "no_fall" not in lower:
                falls.append(rel)
            elif "non_fall" in lower or "nonfall" in lower or "no_fall" in lower:
                non_falls.append(rel)
            elif "non" in lower and "fall" in lower:
                non_falls.append(rel)
            else:
                non_falls.append(rel)
    return sorted(falls), sorted(non_falls)

def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_path", "label"])
        for vp, lb in rows:
            w.writerow([vp, lb])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    falls, non_falls = collect_videos(args.root)
    data = [(p, 1) for p in falls] + [(p, 0) for p in non_falls]
    if not data:
        raise SystemExit("No videos found. Please check the root and supported extensions.")

    X = [p for p, _ in data]
    y = [l for _, l in data]

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=(args.val_ratio + args.test_ratio), stratify=y, random_state=args.seed)
    val_size = args.val_ratio / (args.val_ratio + args.test_ratio) if (args.val_ratio + args.test_ratio) > 0 else 0.0
    if X_tmp:
        X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=(1 - val_size), stratify=y_tmp, random_state=args.seed)
    else:
        X_val, y_val, X_test, y_test = [], [], [], []

    out_dir = Path(args.out)
    write_csv(out_dir / "train.csv", list(zip(X_train, y_train)))
    write_csv(out_dir / "val.csv", list(zip(X_val, y_val)))
    write_csv(out_dir / "test.csv", list(zip(X_test, y_test)))
    print(f"Wrote {out_dir/'train.csv'}, {out_dir/'val.csv'}, {out_dir/'test.csv'}")

if __name__ == "__main__":
    main()
FILE

cat > scripts/download_urfd.py <<'FILE'
import argparse
import shutil
import zipfile
from pathlib import Path
import requests

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg")

HELP = """
URFD download helper

This script tries to:
1) Download a URFD archive if a direct public URL is provided, OR
2) Use a local --zip you already downloaded, then
3) Extract and structure files into:
   <out>/falls/*.mp4 and <out>/non_falls/*.mp4

Usage examples:
  python scripts/download_urfd.py --out data/raw/urfd --url https://<public>/URFD.zip
  python scripts/download_urfd.py --out data/raw/urfd --zip /path/to/URFD.zip

If no URL is public (license required), manually download from the official page,
then pass --zip to structure them.
"""

def download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def extract(zip_path: Path, tmp_dir: Path):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(tmp_dir)

def classify_and_copy(src_root: Path, out_root: Path):
    falls_dir = out_root / "falls"
    non_dir = out_root / "non_falls"
    falls_dir.mkdir(parents=True, exist_ok=True)
    non_dir.mkdir(parents=True, exist_ok=True)

    def is_fall(p: Path):
        s = p.stem.lower()
        sp = str(p.parent).lower()
        return ("fall" in s or "fall" in sp) and not ("non" in s or "adl" in s or "no_fall" in s or "nonfall" in s)

    cnt_f, cnt_n = 0, 0
    for p in src_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            target = falls_dir / p.name if is_fall(p) else non_dir / p.name
            if not target.exists():
                shutil.copy2(p, target)
            if is_fall(p):
                cnt_f += 1
            else:
                cnt_n += 1
    return cnt_f, cnt_n

def main():
    ap = argparse.ArgumentParser(description="URFD downloader/structurer")
    ap.add_argument("--out", required=True, help="Output directory (will create falls/ and non_falls/)")
    ap.add_argument("--url", default="", help="Direct public URL to URFD zip (if available)")
    ap.add_argument("--zip", default="", help="Path to a locally downloaded zip")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.url and not args.zip:
        print(HELP)
        return

    tmp = out_root / "_tmp"
    zip_path = Path(args.zip) if args.zip else out_root / "_tmp" / "urfd.zip"
    try:
        if args.url:
            print(f"Downloading: {args.url}")
            download(args.url, zip_path)
        print(f"Extracting: {zip_path}")
        extract(zip_path, tmp)
        f, n = classify_and_copy(tmp, out_root)
        print(f"Structured URFD into {out_root}")
        print(f"Falls: {f}, Non-falls: {n}")
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()
FILE

cat > scripts/download_le2i.py <<'FILE'
import argparse
import shutil
import zipfile
from pathlib import Path
import requests

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg")

HELP = """
Le2i download helper

This script tries to:
1) Download a Le2i archive if a direct public URL is provided, OR
2) Use a local --zip you already downloaded, then
3) Extract and structure files into:
   <out>/falls/*.mp4 and <out>/non_falls/*.mp4

Usage examples:
  python scripts/download_le2i.py --out data/raw/le2i --url https://<public>/le2i.zip
  python scripts/download_le2i.py --out data/raw/le2i --zip /path/to/le2i.zip

If no URL is public (license required), manually download from the official page,
then pass --zip to structure them.

Heuristics: file/folder names containing 'fall' => fall, containing 'adl'/'non' => non-fall.
Please verify output and adjust if necessary.
"""

def download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def extract(zip_path: Path, tmp_dir: Path):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(tmp_dir)

def classify_and_copy(src_root: Path, out_root: Path):
    falls_dir = out_root / "falls"
    non_dir = out_root / "non_falls"
    falls_dir.mkdir(parents=True, exist_ok=True)
    non_dir.mkdir(parents=True, exist_ok=True)

    def is_fall(p: Path):
        s = p.stem.lower()
        sp = str(p.parent).lower()
        return ("fall" in s or "fall" in sp) and not ("non" in s or "adl" in s or "no_fall" in s or "nonfall" in s)

    cnt_f, cnt_n = 0, 0
    for p in src_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            target = falls_dir / p.name if is_fall(p) else non_dir / p.name
            if not target.exists():
                shutil.copy2(p, target)
            if is_fall(p):
                cnt_f += 1
            else:
                cnt_n += 1
    return cnt_f, cnt_n

def main():
    ap = argparse.ArgumentParser(description="Le2i downloader/structurer")
    ap.add_argument("--out", required=True, help="Output directory (will create falls/ and non_falls/)")
    ap.add_argument("--url", default="", help="Direct public URL to Le2i zip (if available)")
    ap.add_argument("--zip", default="", help="Path to a locally downloaded zip")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.url and not args.zip:
        print(HELP)
        return

    tmp = out_root / "_tmp"
    zip_path = Path(args.zip) if args.zip else out_root / "_tmp" / "le2i.zip"
    try:
        if args.url:
            print(f"Downloading: {args.url}")
            download(args.url, zip_path)
        print(f"Extracting: {zip_path}")
        extract(zip_path, tmp)
        f, n = classify_and_copy(tmp, out_root)
        print(f"Structured Le2i into {out_root}")
        print(f"Falls: {f}, Non-falls: {n}")
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()
FILE

cat > tools/pose_extract.py <<'FILE'
import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 30.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps) if fps > 0 else 30.0

def select_person(instances):
    if not instances:
        return None
    best = None
    best_area = -1
    for inst in instances:
        bbox = inst.get("bbox", None)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best = inst
    return best or instances[0]

def extract_keypoints_from_video(video_path, pose_model="rtmpose-s", det_model="rtmdet_tiny"):
    try:
        from mmpose.apis import MMPoseInferencer
    except Exception as e:
        raise RuntimeError("Please ensure mmpose is installed: pip install mmpose") from e

    inferencer = MMPoseInferencer(pose2d=pose_model, det_model=det_model, device=None)
    gen = inferencer(video_path, show=False)
    frames_kpts = []
    k_format = "coco-17"
    for res in gen:
        preds = res.get("predictions", None)
        if preds is None or len(preds) == 0:
            frames_kpts.append(None)
            continue
        pred = preds[0]
        instances = []
        kp = pred.get("keypoints", None)
        ks = pred.get("keypoint_scores", None)
        bboxes = pred.get("bboxes", None)
        if kp is None:
            frames_kpts.append(None)
            continue
        N, K, _ = kp.shape
        if ks is None:
            ks = np.ones((N, K), dtype=np.float32)
        if bboxes is None:
            bboxes = np.zeros((N, 4), dtype=np.float32)
        for i in range(N):
            instances.append({
                "bbox": bboxes[i].tolist(),
                "keypoints": kp[i].astype(np.float32),
                "keypoint_scores": ks[i].astype(np.float32),
            })
        sel = select_person(instances)
        if sel is None:
            frames_kpts.append(None)
            continue
        kpts = sel["keypoints"]
        scrs = sel["keypoint_scores"]
        if kpts.shape[0] == 17:
            k_format = "coco-17"
        frames_kpts.append(np.concatenate([kpts, scrs[:, None]], axis=1))

    if len(frames_kpts) == 0:
        raise RuntimeError(f"No frames processed for {video_path}")
    K = frames_kpts[0].shape[0] if frames_kpts[0] is not None else 17
    arr = np.full((len(frames_kpts), K, 3), np.nan, dtype=np.float32)
    for t, k in enumerate(frames_kpts):
        if k is not None:
            arr[t] = k
    for j in range(K):
        for c in range(3):
            v = arr[:, j, c]
            last = np.nan
            for t in range(len(v)):
                if not np.isnan(v[t]):
                    last = v[t]
                elif not np.isnan(last):
                    v[t] = last
            last = np.nan
            for t in range(len(v) - 1, -1, -1):
                if not np.isnan(v[t]):
                    last = v[t]
                elif not np.isnan(last):
                    v[t] = last
            arr[:, j, c] = v

    fps = get_fps(video_path)
    meta = {"video_path": str(video_path), "format": k_format}
    return arr, fps, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns: video_path,label")
    ap.add_argument("--out-root", required=True, help="Root dir to save npz files (will mirror input dirs)")
    ap.add_argument("--pose-model", default="rtmpose-s", help="MMPose pose2d model name")
    ap.add_argument("--det-model", default="rtmdet_tiny", help="Detector model for MMPose")
    args = ap.parse_args()

    import csv
    items = []
    with open(args.csv, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            vp = row["video_path"]
            lb = int(row["label"])
            items.append((vp, lb))
    print(f"Loaded {len(items)} videos from {args.csv}")

    for video_path, label in tqdm(items):
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"[WARN] Not found: {video_path}")
            continue
        rel = video_path.with_suffix(".npz")
        out_path = Path(args.out_root) / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            kpts, fps, meta = extract_keypoints_from_video(str(video_path), args.pose_model, args.det_model)
        except Exception as e:
            print(f"[ERROR] Failed on {video_path}: {e}")
            continue
        np.savez_compressed(
            out_path,
            keypoints=kpts.astype(np.float32),
            fps=np.array([fps], dtype=np.float32),
            meta=np.array([meta], dtype=object),
            label=np.array([label], dtype=np.int64),
        )

if __name__ == "__main__":
    main()
FILE

# packages
printf "%s\n" > tools/__init__.py ""
printf "%s\n" > src/__init__.py ""
printf "%s\n" > src/models/__init__.py ""
printf "%s\n" > src/datasets/__init__.py ""
printf "%s\n" > src/utils/__init__.py ""

cat > src/utils/feature_engineering.py <<'FILE'
import numpy as np

NOSE=0; L_EYE=1; R_EYE=2; L_EAR=3; R_EAR=4; L_SHOULDER=5; R_SHOULDER=6
L_ELBOW=7; R_ELBOW=8; L_WRIST=9; R_WRIST=10; L_HIP=11; R_HIP=12; L_KNEE=13; R_KNEE=14; L_ANKLE=15; R_ANKLE=16

def _pair_dist(p1, p2):
    return np.linalg.norm(p1 - p2, axis=-1)

def normalize_keypoints(kpts, method="shoulder"):
    xy = kpts[..., :2].copy()
    scores = kpts[..., 2:3]
    mid_shoulder = (xy[:, L_SHOULDER, :] + xy[:, R_SHOULDER, :]) / 2
    mid_hip = (xy[:, L_HIP, :] + xy[:, R_HIP, :]) / 2
    use_sh = np.isfinite(mid_shoulder).all(axis=1, keepdims=True)
    center = np.where(use_sh, mid_shoulder, mid_hip)
    xy_centered = xy - center[:, None, :]

    if method == "shoulder":
        scale = _pair_dist(xy[:, L_SHOULDER, :], xy[:, R_SHOULDER, :])
    elif method == "hip":
        scale = _pair_dist(xy[:, L_HIP, :], xy[:, R_HIP, :])
    else:
        scale = (xy.max(axis=1)[:, 1] - xy.min(axis=1)[:, 1])

    if np.any(scale <= 1e-6):
        safe = scale[scale > 0]
        fallback = np.median(safe) if safe.size > 0 else 1.0
        scale = np.where(scale <= 1e-6, fallback, scale)

    scale = scale.reshape(-1, 1, 1)
    xy_norm = xy_centered / scale
    out = np.concatenate([xy_norm, scores], axis=-1)
    return out, scale.squeeze()

def build_features(kpts, use_conf=True, add_velocity=True, norm="shoulder"):
    k_norm, _ = normalize_keypoints(kpts, method=norm)
    feats = [k_norm[..., :2].reshape(k_norm.shape[0], -1)]
    if use_conf:
        feats.append(k_norm[..., 2:].reshape(k_norm.shape[0], -1))
    if add_velocity:
        vel = np.diff(k_norm[..., :2], axis=0, prepend=k_norm[[0], ..., :2])
        feats.append(vel.reshape(vel.shape[0], -1))
    X = np.concatenate(feats, axis=1).astype(np.float32)
    return X
FILE

cat > src/datasets/fall_dataset.py <<'FILE'
import csv
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from src.utils.feature_engineering import build_features

class FallSeqDataset(Dataset):
    def __init__(self, csv_path, npz_root, seq_len=64, seq_step=16,
                 use_conf=True, add_velocity=True, norm="shoulder"):
        self.items = []
        self.seq_len = seq_len
        self.use_conf = use_conf
        self.add_velocity = add_velocity
        self.norm = norm
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vp = Path(row["video_path"]).with_suffix(".npz")
                npz_path = Path(npz_root) / vp
                label = int(row["label"])
                self.items.append((npz_path, label))
        self.index = []
        for i, (npz_path, label) in enumerate(self.items):
            if not npz_path.exists():
                continue
            k = np.load(npz_path, allow_pickle=True)
            keypoints = k["keypoints"]
            T = keypoints.shape[0]
            for s in range(0, max(1, T - seq_len + 1), seq_step):
                e = s + seq_len
                if e <= T:
                    self.index.append((i, s, e))
        if len(self.index) == 0:
            raise RuntimeError("No sequences found. Did you run pose extraction?")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, s, e = self.index[idx]
        npz_path, label = self.items[i]
        data = np.load(npz_path, allow_pickle=True)
        kpts = data["keypoints"][s:e]
        X = build_features(kpts, use_conf=self.use_conf, add_velocity=self.add_velocity, norm=self.norm)
        y = np.array(label, dtype=np.float32)
        return X, y
FILE

cat > src/models/lstm_fall.py <<'FILE'
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden=128, layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(out_dim, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.head(last).squeeze(-1)
        return logits
FILE

cat > src/models/inception_time.py <<'FILE'
import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels=32, kernel_sizes=(9,19,39), use_bn=True):
        super().__init__()
        self.use_bottleneck = in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            in_conv = bottleneck_channels
        else:
            in_conv = in_channels

        self.conv_list = nn.ModuleList([
            nn.Conv1d(in_conv, out_channels, kernel_size=k, padding=k//2, bias=False)
            for k in kernel_sizes
        ])
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1)) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_bn = self.bottleneck(x) if self.use_bottleneck else x
        conv_outs = [conv(x_bn) for conv in self.conv_list]
        pool_out = self.conv_pool(self.maxpool(x))
        x = torch.cat(conv_outs + [pool_out], dim=1)
        x = self.bn(x)
        return self.relu(x)

class InceptionTime(nn.Module):
    def __init__(self, in_channels, num_blocks=6, out_channels=32, bottleneck_channels=32, use_residual=True, n_classes=1, dropout=0.2):
        super().__init__()
        blocks = []
        self.use_residual = use_residual
        self.shortcut_layers = nn.ModuleList()
        ch = in_channels
        for i in range(num_blocks):
            block = InceptionBlock(
                in_channels=ch,
                out_channels=out_channels,
                bottleneck_channels=bottleneck_channels
            )
            blocks.append(block)
            out_ch = out_channels * 4
            if use_residual and (i % 3 == 2):
                self.shortcut_layers.append(nn.Sequential(
                    nn.Conv1d(ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_ch)
                ))
                ch = out_ch
            else:
                self.shortcut_layers.append(None)
                ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.final_bn = nn.BatchNorm1d(ch)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(ch, n_classes)

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            out = block(x)
            if self.use_residual and (i % 3 == 2):
                sc = self.shortcut_layers[i](x)
                out = F.relu(out + sc)
            x = out
        x = self.final_bn(x)
        x = x.mean(dim=-1)
        x = self.dropout(x)
        logits = self.head(x).squeeze(-1)
        return logits
FILE

cat > src/train.py <<'FILE'
import argparse
import os
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from src.datasets.fall_dataset import FallSeqDataset
from src.models.lstm_fall import LSTMClassifier
from src.models.inception_time import InceptionTime

def collate(batch):
    Xs, ys = zip(*batch)
    X = torch.tensor(np.stack(Xs), dtype=torch.float32)
    y = torch.tensor(np.array(ys), dtype=torch.float32)
    return X, y

def evaluate(model, loader, device, model_type):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for X, y in loader:
            if model_type == "inception":
                X = X.permute(0, 2, 1)
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            prob = torch.sigmoid(logits)
            ys.append(y.cpu().numpy())
            ps.append(prob.cpu().numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    y_pred = (y_prob >= 0.5).astype(np.int32)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    return {"precision": p, "recall": r, "f1": f1, "auc": auc}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-train", required=True)
    ap.add_argument("--csv-val", required=True)
    ap.add_argument("--npz-root", required=True)
    ap.add_argument("--model", default="lstm", choices=["lstm","inception"])
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--seq-step", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--it-filters", type=int, default=32)
    ap.add_argument("--it-depth", type=int, default=6)
    ap.add_argument("--use-vel", action="store_true")
    ap.add_argument("--use-conf", action="store_true")
    ap.add_argument("--norm", default="shoulder", choices=["shoulder","hip","bbox"])
    ap.add_argument("--class-weight", type=float, default=1.0)
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = FallSeqDataset(args.csv_train, args.npz_root, seq_len=args.seq_len, seq_step=args.seq_step, use_conf=args.use_conf, add_velocity=args.use_vel, norm=args.norm)
    val_ds   = FallSeqDataset(args.csv_val, args.npz_root, seq_len=args.seq_len, seq_step=args.seq_step, use_conf=args.use_conf, add_velocity=args.use_vel, norm=args.norm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate)

    input_dim = train_ds[0][0].shape[-1]

    if args.model == "lstm":
        model = LSTMClassifier(input_dim, hidden=args.hidden, layers=args.layers, dropout=args.dropout, bidirectional=args.bidirectional).to(args.device)
    else:
        model = InceptionTime(in_channels=input_dim, num_blocks=args.it_depth, out_channels=args.it_filters, bottleneck_channels=min(32, max(8, input_dim//8)), n_classes=1, dropout=args.dropout).to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    pos_weight = torch.tensor([args.class_weight], dtype=torch.float32, device=args.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, stamp)
    os.makedirs(run_dir, exist_ok=True)
    best_f1 = -1
    best_path = os.path.join(run_dir, "best.pt")

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            if args.model == "inception":
                X = X.permute(0, 2, 1)
            X = X.to(args.device)
            y = y.to(args.device)
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        metrics = evaluate(model, val_loader, args.device, args.model)
        print(f"[{epoch}/{args.epochs}] train_loss={train_loss:.4f} val_p={metrics['precision']:.3f} val_r={metrics['recall']:.3f} val_f1={metrics['f1']:.3f} val_auc={metrics['auc']:.3f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "model_type": args.model,
                "input_dim": input_dim
            }, best_path)
    print(f"Best F1: {best_f1:.3f} saved at {best_path}")

if __name__ == "__main__":
    main()
FILE

cat > src/eval.py <<'FILE'
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score
from src.datasets.fall_dataset import FallSeqDataset
from src.models.lstm_fall import LSTMClassifier
from src.models.inception_time import InceptionTime

def collate(batch):
    Xs, ys = zip(*batch)
    X = torch.tensor(np.stack(Xs), dtype=torch.float32)
    y = torch.tensor(np.array(ys), dtype=torch.float32)
    return X, y

def build_model(ckpt, input_dim, device):
    args = ckpt.get("args", {})
    model_type = ckpt.get("model_type", args.get("model", "lstm"))
    if model_type == "inception":
        it_filters = args.get("it_filters", 32)
        it_depth = args.get("it_depth", 6)
        dropout = args.get("dropout", 0.2)
        model = InceptionTime(in_channels=input_dim, num_blocks=it_depth, out_channels=it_filters, bottleneck_channels=min(32, max(8, input_dim//8)), n_classes=1, dropout=dropout).to(device)
    else:
        hidden = args.get("hidden", 128)
        layers = args.get("layers", 2)
        dropout = args.get("dropout", 0.2)
        bidirectional = args.get("bidirectional", False)
        model = LSTMClassifier(input_dim, hidden=hidden, layers=layers, dropout=dropout, bidirectional=bidirectional).to(device)
    return model, model_type

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-test", required=True)
    ap.add_argument("--npz-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--seq-step", type=int, default=16)
    ap.add_argument("--use-vel", action="store_true")
    ap.add_argument("--use-conf", action="store_true")
    ap.add_argument("--norm", default="shoulder", choices=["shoulder","hip","bbox"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ds = FallSeqDataset(args.csv_test, args.npz_root, seq_len=args.seq_len, seq_step=args.seq_step, use_conf=args.use_conf, add_velocity=args.use_vel, norm=args.norm)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, collate_fn=collate)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    input_dim = ckpt.get("input_dim", ds[0][0].shape[-1])
    model, model_type = build_model(ckpt, input_dim, args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for X, y in loader:
            if model_type == "inception":
                X = X.permute(0, 2, 1)
            X = X.to(args.device)
            logits = model(X)
            prob = torch.sigmoid(logits).cpu().numpy()
            ys.append(y.numpy())
            ps.append(prob)
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    y_pred = (y_prob >= 0.5).astype(np.int32)

    print(classification_report(y_true, y_pred, digits=3))
    try:
        auc = roc_auc_score(y_true, y_prob)
        print(f"AUC: {auc:.3f}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
FILE

cat > src/infer_video.py <<'FILE'
import argparse
import numpy as np
import torch
import cv2
from src.models.lstm_fall import LSTMClassifier
from src.models.inception_time import InceptionTime
from src.utils.feature_engineering import build_features
from tools.pose_extract import extract_keypoints_from_video

def smooth_preds(probs, k=5):
    if k <= 1:
        return probs
    out = np.copy(probs)
    for i in range(len(probs)):
        s = max(0, i - k + 1)
        out[i] = np.mean(probs[s:i+1])
    return out

def build_model(ckpt, input_dim, device):
    args = ckpt.get("args", {})
    model_type = ckpt.get("model_type", args.get("model", "lstm"))
    if model_type == "inception":
        it_filters = args.get("it_filters", 32)
        it_depth = args.get("it_depth", 6)
        dropout = args.get("dropout", 0.2)
        model = InceptionTime(in_channels=input_dim, num_blocks=it_depth, out_channels=it_filters, bottleneck_channels=min(32, max(8, input_dim//8)), n_classes=1, dropout=dropout).to(device)
    else:
        hidden = args.get("hidden", 128)
        layers = args.get("layers", 2)
        dropout = args.get("dropout", 0.2)
        bidirectional = args.get("bidirectional", False)
        model = LSTMClassifier(input_dim, hidden=hidden, layers=layers, dropout=dropout, bidirectional=bidirectional).to(device)
    return model, model_type

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pose-model", default="rtmpose-s")
    ap.add_argument("--det-model", default="rtmdet_tiny")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--seq-step", type=int, default=8)
    ap.add_argument("--use-vel", action="store_true")
    ap.add_argument("--use-conf", action="store_true")
    ap.add_argument("--norm", default="shoulder", choices=["shoulder","hip","bbox"])
    ap.add_argument("--smooth-k", type=int, default=5)
    ap.add_argument("--alarm-th", type=float, default=0.6)
    ap.add_argument("--out-vis", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print("Extracting keypoints...")
    kpts, fps, meta = extract_keypoints_from_video(args.video, args.pose_model, args.det_model)
    X = build_features(kpts, use_conf=args.use_conf, add_velocity=args.use_vel, norm=args.norm)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    input_dim = ckpt.get("input_dim", X.shape[-1])
    model, model_type = build_model(ckpt, input_dim, args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    probs = []
    idxs = []
    with torch.no_grad():
        for s in range(0, max(1, X.shape[0] - args.seq_len + 1), args.seq_step):
            e = s + args.seq_len
            if e > X.shape[0]:
                break
            xw = X[s:e]
            if model_type == "inception":
                xw = np.transpose(xw, (1, 0))
            xw = torch.tensor(xw[None, ...], dtype=torch.float32, device=args.device)
            p = torch.sigmoid(model(xw)).item()
            probs.append(p)
            idxs.append((s, e))
    probs = np.array(probs, dtype=np.float32)
    probs_sm = smooth_preds(probs, k=args.smooth_k)

    print(f"Mean prob: {probs_sm.mean():.3f}, Max prob: {probs_sm.max():.3f}")
    if args.out_vis:
        cap = cv2.VideoCapture(args.video)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 25
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.out_vis, fourcc, fps_v, (w, h))
        t = 0
        win_map = np.zeros((X.shape[0],), dtype=np.float32)
        for (s, e), p in zip(idxs, probs_sm):
            win_map[s:e] = np.maximum(win_map[s:e], p)
        alarm = win_map >= args.alarm_th
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            p = win_map[t] if t < len(win_map) else 0.0
            is_alarm = alarm[t] if t < len(alarm) else False
            cv2.putText(frame, f"Fall prob: {p:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0) if not is_alarm else (0,0,255), 2)
            if is_alarm:
                cv2.putText(frame, f"ALERT", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
            out.write(frame)
            t += 1
        cap.release()
        out.release()
        print(f"Saved visualization to {args.out_vis}")

if __name__ == "__main__":
    main()
FILE

# README (four-backtick markdown)
cat > README.md <<'FILE'
# Fall Detection Starter (MMPose + LSTM/InceptionTime, PyTorch)

This repo provides a ready-to-run pipeline for fall detection from videos by fusing:
- 2D human pose estimation (MMPose) to extract keypoints over time
- A time-series classifier (LSTM or InceptionTime) to detect falls from keypoint sequences

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

## 4) Train (LSTM or InceptionTime)

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
FILE

# Commit and push
git add -A
git commit -m "feat: fall-detection starter (MMPose + LSTM/InceptionTime)"
git push -u origin "$(git rev-parse --abbrev-ref HEAD)"

# Print PR link
owner_repo=$(git config --get remote.origin.url | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')
echo
echo "Open PR: https://github.com/${owner_repo}/compare/main...$(git rev-parse --abbrev-ref HEAD)?expand=1"