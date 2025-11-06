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
