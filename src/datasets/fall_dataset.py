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
