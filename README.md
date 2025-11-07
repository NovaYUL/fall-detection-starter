# 跌倒检测起步项目 (MMPose + LSTM / InceptionTime / tsai, PyTorch)

本仓库提供一个可直接运行的从视频中检测跌倒 (fall) 的端到端示例流程，核心思想：
- 使用 MMPose 进行 2D 人体姿态估计，提取随时间变化的关键点序列
- 使用时间序列分类模型（LSTM / InceptionTime / tsai 版本）对关键点序列进行跌倒事件判别

## 功能特性

- 视频 → 关键点 (npz)：基于 MMPoseInferencer 自动下载并推理 RTMPose + RTMDet
- 关键点序列的滑动窗口切片，构造训练 / 验证 / 测试样本
- 单视频在线推理：边姿态估计边滑窗预测，输出平滑后的跌倒概率并可生成可视化视频
- 特征可扩展：归一化、速度、关键点置信度等
- 数据集辅助脚本：URFD / Le2i 下载与结构整理
- 支持多种时间序列模型：原生 PyTorch LSTM / InceptionTime，以及 tsai 中的 LSTMPlus / InceptionTime 版本

---

## 1) 环境准备

建议使用 Python 3.9+。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

若使用 GPU，请到 PyTorch 官网选择匹配的 CUDA 版本：
https://pytorch.org/get-started/locally/

安装 mmpose 时会自动拉取 mmengine / mmcv / mmdet 等依赖。

---

## 2) 数据集

我们使用三个拆分的 CSV：
- `datasets/train.csv`
- `datasets/val.csv`
- `datasets/test.csv`

CSV 每行格式：`video_path,label`，其中 `label=1` 表示跌倒视频，`label=0` 表示非跌倒。

从包含 `falls/` 与 `non_falls/`（或其他名称，脚本有一定启发式）的根目录生成 CSV：

```bash
python scripts/make_video_csv.py \
  --root /path/to/root \
  --out datasets \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

### 2.1 快速获取 URFD / Le2i

辅助脚本会尝试下载（如果有公开链接）或对本地 zip 进行结构化，目标目录结构：

```
data/raw/<dataset>/{falls,non_falls}/*.mp4
```

示例：

```bash
# URFD
python scripts/download_urfd.py --out data/raw/urfd
# Le2i
python scripts/download_le2i.py --out data/raw/le2i
```

说明：
- 若数据集需手工申请或无直接公开链接，可手动下载后使用 `--zip` 参数让脚本进行解压与分类。
- “fall vs non-fall” 的判断基于文件或父目录名包含关键字（如包含 fall；包含 adl / non / no_fall 则视为非跌倒），请根据实际数据适当核对。

生成 CSV（URFD 示例）：

```bash
python scripts/make_video_csv.py --root data/raw/urfd --out datasets --val-ratio 0.15 --test-ratio 0.15
```

---

## 3) 2D 关键点提取

对 CSV 中视频批量执行姿态估计，生成 npz 文件：

```bash
python tools/pose_extract.py --csv datasets/train.csv --out-root data/processed --pose-model rtmpose-s
python tools/pose_extract.py --csv datasets/val.csv   --out-root data/processed --pose-model rtmpose-s
python tools/pose_extract.py --csv datasets/test.csv  --out-root data/processed --pose-model rtmpose-s
```

输出（每个视频对应一个 npz）包含：
- `keypoints`: (T, K, 3) → x, y, score
- `fps`: 帧率
- `meta`: 包含 `video_path` 与关键点格式标识（例如 `"coco-17"`）

默认选取每帧中 bbox 最大的人作为目标（单人场景）。多人人体检测可后续集成跟踪（ByteTrack / DeepSORT）并按 track 输出序列。

---

## 4) 训练（LSTM / InceptionTime / tsai）

### 4.1 原生 LSTM 基线

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

### 4.2 原生 InceptionTime 基线

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

### 4.3 tsai 模型（LSTMPlus / InceptionTime）

```bash
# tsai LSTMPlus
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

# tsai InceptionTime
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

训练日志与最优权重保存在：`runs/<timestamp>/best.pt`，其中包含 `model_type`、输入特征维度等元数据。

---

## 5) 评估

```bash
python -m src.eval \
  --csv-test datasets/test.csv \
  --npz-root data/processed \
  --checkpoint runs/<timestamp>/best.pt \
  --seq-len 64 --seq-step 16 \
  --use-vel --use-conf --norm shoulder
```

输出包括分类报告与 AUC（如可计算）。

---

## 6) 单视频推理与可视化

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

流程：
1. 在线调用 MMPose 进行关键点估计
2. 构造滑窗序列并计算概率
3. 平滑概率（滑动平均）
4. 在输出视频帧叠加 “Fall prob” 与超过阈值的 “ALERT” 提示

---

## 7) 说明与技巧

- 归一化：肩宽 / 臀宽 / bbox 高度，肩优先；归一化尺度稳定性很关键。
- 缺失关键点：建议使用 `score` 特征，与插值或前后填充处理缺失。
- 样本不平衡：使用 `--class-weight`，或后续加入 focal loss、过采样、加权采样。
- 受试者划分：若数据集提供人员标识，按人员进行训练/验证拆分更利于泛化。
- 多人场景：需要跟踪（ByteTrack / DeepSORT）后按跟踪 ID 生成多份序列。
- tsai 模型输入为通道优先 (B, C, L)，训练脚本已自动在 `--model tsai_*` 时转换。

---

## 8) 目录结构（示例）

```
.
├── datasets/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── data/
│   └── processed/
├── outputs/
├── runs/
├── scripts/
│   ├── make_video_csv.py
│   ├── download_urfd.py
│   ├── download_le2i.py
│   └── check_imports.py        # CI 自检脚本（如已添加）
├── src/
│   ├── datasets/fall_dataset.py
│   ├── models/
│   │   ├── lstm_fall.py
│   │   ├── inception_time.py
│   │   ├── tsai_lstm.py
│   │   └── tsai_inception.py
│   ├── utils/feature_engineering.py
│   ├── train.py
│   ├── eval.py
│   └── infer_video.py
└── tools/
    └── pose_extract.py
```

---

## 9) 参考与来源 (References & Attribution)

### 姿态估计
- MMPose (OpenMMLab): https://github.com/open-mmlab/mmpose  
- RTMPose 项目页: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose  
- MMDetection / RTMDet: https://github.com/open-mmlab/mmdetection  

### 时间序列模型（自定义 + 开源）
- tsai (timeseriesAI): https://github.com/timeseriesAI/tsai  
- LSTM 论文：Sepp Hochreiter & Jürgen Schmidhuber, “Long Short-Term Memory,” Neural Computation, 1997.  
- InceptionTime 论文：Hassan Ismail Fawaz et al., “InceptionTime: Finding AlexNet for Time Series Classification,” arXiv:1909.04939, 2019.  

### 其他
- PyTorch: https://pytorch.org  
- OpenMMLab 文档（生态组件）：https://openmmlab.com  

许可说明：
- 本仓库代码使用 MIT License。
- MMPose / MMDetection 等属于各自的开源许可（Apache-2.0 等），使用时需遵守其条款。
- tsai 为 MIT 许可。

---

## 10) 后续可扩展方向

- 多人场景跟踪与多实例概率融合
- 加入更多时序模型（TCN、Transformer、Temporal Fusion Transformer）
- 增加 focal loss / 加权采样 / 分层评估指标
- 模型配置 YAML 化与实验记录自动化
- 引入推理加速（ONNX / TensorRT）与轻量化（量化 / 剪枝）

---

## 11) 快速试运行（自检建议）

```bash
# 仅做环境搭建与导入检查（不跑训练）
python scripts/check_imports.py
```

若需对 README 进一步精简或加入英文对照，请在 PR 中提出。
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
