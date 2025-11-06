import os
from pathlib import Path
import argparse
import numpy as np


def make_npz(out_path: Path, label: int, T: int = 160, K: int = 17, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Generate smooth 2D trajectories with noise; add confidence in [0.7,1.0]
    t = np.linspace(0, 4 * np.pi, T, dtype=np.float32)
    base_x = np.sin(t)[..., None]
    base_y = np.cos(t)[..., None]
    xy = np.concatenate([base_x, base_y], axis=1)  # (T,2)
    xy = xy[None, ...] + 0.05 * rng.standard_normal((K, T, 2), dtype=np.float32)
    xy = np.transpose(xy, (1, 0, 2))  # (T,K,2)
    conf = rng.uniform(0.7, 1.0, size=(T, K, 1)).astype(np.float32)
    keypoints = np.concatenate([xy, conf], axis=-1).astype(np.float32)  # (T,K,3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        keypoints=keypoints,
        fps=np.array([25.0], dtype=np.float32),
        meta=np.array([{"video_path": str(out_path.with_suffix('.mp4'))}], dtype=object),
        label=np.array([label], dtype=np.int64),
    )


def write_csv(csv_path: Path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("video_path,label\n")
        for vp, lb in rows:
            f.write(f"{vp},{lb}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="data/processed", help="Where to place dummy npz files")
    ap.add_argument("--datasets", default="datasets", help="Where to write CSV files")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    npz_root = Path(args.out_root)
    ds_root = Path(args.datasets)

    # Define dummy items (relative paths used in CSV)
    items = {
        "train": [("dummy/fall_001.mp4", 1), ("dummy/non_001.mp4", 0), ("dummy/fall_002.mp4", 1), ("dummy/non_002.mp4", 0)],
        "val":   [("dummy/fall_101.mp4", 1), ("dummy/non_101.mp4", 0)],
        "test":  [("dummy/fall_201.mp4", 1), ("dummy/non_201.mp4", 0)],
    }

    # Create npz files under npz_root/<rel with .npz>
    for split, rows in items.items():
        for vp, lb in rows:
            npz_path = npz_root / Path(vp).with_suffix(".npz")
            make_npz(npz_path, label=lb, seed=rng.integers(0, 10_000))

    # Write CSVs that reference the video paths (the dataset loader will map to .npz under npz_root)
    for split, rows in items.items():
        write_csv(ds_root / f"{split}.csv", rows)

    print(f"Wrote dummy CSVs to {ds_root} and npz to {npz_root}/dummy/*.npz")


if __name__ == "__main__":
    main()
