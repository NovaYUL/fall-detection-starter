import argparse
import numpy as np
import torch
import cv2
from src.models.lstm_fall import LSTMClassifier
from src.models.inception_time import InceptionTime
from src.models.tsai_lstm import TsaiLSTMClassifier
from src.models.tsai_inception import TsaiInceptionTimeClassifier
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
    elif model_type == "tsai_lstm":
        model = TsaiLSTMClassifier(input_dim=input_dim).to(device)
    elif model_type == "tsai_inception":
        model = TsaiInceptionTimeClassifier(input_dim=input_dim).to(device)
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
            if model_type in {"inception", "tsai_lstm", "tsai_inception"}:
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
