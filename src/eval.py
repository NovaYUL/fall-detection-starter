import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score
from src.datasets.fall_dataset import FallSeqDataset
from src.models.lstm_fall import LSTMClassifier
from src.models.inception_time import InceptionTime
from src.models.tsai_lstm import TsaiLSTMClassifier
from src.models.tsai_inception import TsaiInceptionTimeClassifier

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
            if model_type in {"inception", "tsai_lstm", "tsai_inception"}:
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
