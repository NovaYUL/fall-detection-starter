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
