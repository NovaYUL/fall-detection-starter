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
