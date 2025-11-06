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
