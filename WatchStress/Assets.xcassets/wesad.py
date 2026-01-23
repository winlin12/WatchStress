#!/usr/bin/env python3
import argparse, json, os, pickle, zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# Optional: sklearn for a clean logistic regression fit
try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# -----------------------------
# Helpers to read E4 CSV format
# -----------------------------
import os
import numpy as np

def read_e4_signal_from_folder(folder: str, filename: str):
    """
    E4 CSV format:
      row0: unix start time
      row1: sample rate (Hz)
      rows2+: samples (1 col for TEMP/BVP/EDA, 3 cols for ACC)
    Returns: (start_unix, fs_hz, samples)
    """
    path = os.path.join(folder, filename)
    arr = np.loadtxt(path, delimiter=",", dtype=float)
    if arr.ndim == 1:
        start = float(arr[0])
        fs = float(arr[1])
        data = arr[2:].reshape(-1, 1)
    else:
        start = float(arr[0, 0])
        fs = float(arr[1, 0])
        data = arr[2:, :]
    return start, fs, data


def read_ibi_from_folder_if_exists(folder: str):
    """
    IBI.csv format:
      row0: unix start time, literal "IBI" marker
      col0: time since start (s)
      col1: ibi duration (s)
    """
    for cand in ("IBI.csv", "ibi.csv"):
        path = os.path.join(folder, cand)
        if os.path.exists(path):
            arr = np.loadtxt(path, delimiter=",", dtype=float, skiprows=1)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 2)
            return arr
    return None



# -----------------------------
# Feature extraction
# -----------------------------
@dataclass
class WindowFeatures:
    # Map these names to your Swift feature names later
    hr_mean_bpm: float
    hrv_sdnn_ms: float
    wrist_temp_c: float
    acc_rms_g: float


def extract_features_for_window(
    t0: float,
    t1: float,
    temp_start: float,
    temp_fs: float,
    temp: np.ndarray,
    hr_start: float,
    hr_fs: float,
    hr: np.ndarray,
    acc_start: float,
    acc_fs: float,
    acc: np.ndarray,
    ibi: Optional[np.ndarray],
) -> Optional[WindowFeatures]:
    """
    Window is [t0, t1] seconds from session start.
    """
    # TEMP: convert window seconds -> index
    temp_i0 = int(max(0, np.floor(t0 * temp_fs)))
    temp_i1 = int(min(len(temp), np.ceil(t1 * temp_fs)))
    if temp_i1 - temp_i0 < max(2, int(0.25 * (t1 - t0) * temp_fs)):
        return None
    temp_mean = float(np.mean(temp[temp_i0:temp_i1, 0]))

    # HR mean (bpm)
    hr_i0 = int(max(0, np.floor(t0 * hr_fs)))
    hr_i1 = int(min(len(hr), np.ceil(t1 * hr_fs)))
    if hr_i1 - hr_i0 < max(2, int(0.25 * (t1 - t0) * hr_fs)):
        return None
    hr_mean = float(np.mean(hr[hr_i0:hr_i1, 0]))

    # ACC RMS magnitude
    acc_i0 = int(max(0, np.floor(t0 * acc_fs)))
    acc_i1 = int(min(len(acc), np.ceil(t1 * acc_fs)))
    if acc_i1 - acc_i0 < max(10, int(0.25 * (t1 - t0) * acc_fs)):
        return None
    seg = acc[acc_i0:acc_i1, :]
    # ACC is in "1/64g" in raw E4 files for WESAD; convert to g.
    seg_g = seg / 64.0
    mag = np.linalg.norm(seg_g, axis=1)
    acc_rms = float(np.sqrt(np.mean(mag ** 2)))

    # IBI -> resting HR + SDNN (if available)
    # SDNN is std dev of NN intervals, typically ms.
    sdnn_ms = np.nan
    if ibi is not None:
        # Filter beats whose timestamps fall within window
        mask = (ibi[:, 0] >= t0) & (ibi[:, 0] < t1)
        ibi_win = ibi[mask, 1]  # seconds
        if len(ibi_win) >= 5:
            ibi_ms = ibi_win * 1000.0
            sdnn_ms = float(np.std(ibi_ms, ddof=1))

    return WindowFeatures(
        hr_mean_bpm=hr_mean,
        hrv_sdnn_ms=float(sdnn_ms),
        wrist_temp_c=temp_mean,
        acc_rms_g=acc_rms,
    )


def robust_mean_std(x: np.ndarray) -> Tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0, 1.0
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else 1.0
    if sd < 1e-6:
        sd = 1.0
    return mu, sd


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wesad_root", required=True, help="Path containing S2/, S3/, ... folders")
    ap.add_argument("--window_s", type=float, default=60.0)
    ap.add_argument("--stride_s", type=float, default=60.0)
    ap.add_argument("--out", default="priors.json")
    ap.add_argument("--subjects", default="", help="Comma-separated subject IDs (e.g., S2,S3). Empty = auto-detect")
    args = ap.parse_args()

    # Label codes from WESAD readme: 1=baseline, 2=stress, 3=amusement, 4=meditation; others ignored.
    BASELINE = 1
    STRESS = 2

    subjects = []
    if args.subjects.strip():
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        subjects = sorted([d for d in os.listdir(args.wesad_root) if d.startswith("S") and os.path.isdir(os.path.join(args.wesad_root, d))])

    X_rows: List[List[float]] = []
    y_rows: List[int] = []

    for sid in subjects:
        subj_dir = os.path.join(args.wesad_root, sid)
        pkl_path = os.path.join(subj_dir, f"{sid}.pkl")
        e4_dir = os.path.join(subj_dir, f"{sid}_E4_Data")
        if not os.path.isdir(e4_dir):
            # some datasets name it "E4_Data" or "E4"
            for alt in ("E4_Data", "E4"):
                alt_path = os.path.join(subj_dir, alt)
                if os.path.isdir(alt_path):
                    e4_dir = alt_path
                    break

        if not os.path.isdir(e4_dir):
            print(f"[skip] {sid}: missing E4 folder (tried {sid}_E4_Data / E4_Data / E4)")
            continue

        if not os.path.exists(pkl_path):
            print(f"[skip] {sid}: missing {sid}.pkl for labels")
            continue
        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f, encoding="latin1")
        labels = pkl_data.get("label")
        if labels is None:
            print(f"[skip] {sid}: missing 'label' in {sid}.pkl")
            continue
        labels = np.asarray(labels).astype(int)

        _, temp_fs, temp = read_e4_signal_from_folder(e4_dir, "TEMP.csv")
        _, hr_fs, hr = read_e4_signal_from_folder(e4_dir, "HR.csv")
        _, acc_fs, acc = read_e4_signal_from_folder(e4_dir, "ACC.csv")
        ibi = read_ibi_from_folder_if_exists(e4_dir)

        duration_s = min(
            len(labels) / 700.0,
            len(temp) / temp_fs,
            len(hr) / hr_fs,
            len(acc) / acc_fs,
        )

        t = 0.0
        while t + args.window_s <= duration_s:
            t0, t1 = t, t + args.window_s

            # Window label = majority label in the interval
            i0 = int(t0 * 700)
            i1 = int(t1 * 700)
            lbl = labels[i0:i1]
            if len(lbl) == 0:
                break
            # drop undefined / ignored labels
            # keep only baseline vs stress for now
            maj = int(np.bincount(lbl).argmax())
            if maj not in (BASELINE, STRESS):
                t += args.stride_s
                continue

            feats = extract_features_for_window(
                t0, t1,
                temp_start=0.0, temp_fs=temp_fs, temp=temp,
                hr_start=0.0, hr_fs=hr_fs, hr=hr,
                acc_start=0.0, acc_fs=acc_fs, acc=acc,
                ibi=ibi,
            )
            if feats is None:
                t += args.stride_s
                continue

            row = [feats.hr_mean_bpm, feats.hrv_sdnn_ms, feats.wrist_temp_c, feats.acc_rms_g]
            if not np.all(np.isfinite(row)):
                # if IBI isn’t available, HR/SDNN may be NaN — you can either skip or impute.
                # For v1: skip rows missing HRV/HR.
                t += args.stride_s
                continue

            X_rows.append(row)
            y_rows.append(1 if maj == STRESS else 0)

            t += args.stride_s

        print(f"[ok] {sid}: collected so far X={len(X_rows)}")

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int)

    if len(X) < 50:
        raise SystemExit(f"Not enough samples extracted ({len(X)}). Check paths / zip contents.")

    feat_names = ["hrMeanBPM", "hrvSDNNms", "wristTempC", "accRMSG"]

    # Compute priors from BASELINE windows only (y==0)
    X_base = X[y == 0]
    priors = {}
    for j, name in enumerate(feat_names):
        mu, sd = robust_mean_std(X_base[:, j])
        priors[name] = {"mean": mu, "std": sd}

    # Standardize using baseline µ/σ
    Xz = np.zeros_like(X)
    for j, name in enumerate(feat_names):
        mu = priors[name]["mean"]
        sd = priors[name]["std"]
        Xz[:, j] = (X[:, j] - mu) / (sd + 1e-6)

    # Fit simple model: stress vs baseline
    if SKLEARN_OK:
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
        clf.fit(Xz, y)
        w = clf.coef_.reshape(-1)
        b = float(clf.intercept_[0])
    else:
        # fallback: ridge-ish linear regression to y
        lam = 1.0
        XtX = Xz.T @ Xz + lam * np.eye(Xz.shape[1])
        Xty = Xz.T @ y.astype(float)
        w = np.linalg.solve(XtX, Xty)
        b = float(np.mean(y) - np.mean(Xz, axis=0) @ w)

    # Your Swift score is “higher is better”.
    # Here y=1 means stress, so higher w·z => more stress => should LOWER the score.
    # So we negate weights to make stress-associated directions reduce score.
    w = -w

    # Scale weights into a sane 0–100 range:
    # Make typical |w·z| around 15 points.
    proj = Xz @ w
    scale = 15.0 / (np.std(proj) + 1e-6)
    w_scaled = w * scale

    out = {
        "meta": {
            "source": "WESAD",
            "labels": {"baseline": 1, "stress": 2},
            "window_s": args.window_s,
            "stride_s": args.stride_s,
            "notes": "Means/stds computed on baseline windows. Weights trained to separate stress vs baseline, then negated so 'higher score = better'."
        },
        "priors": priors,
        "weights": {feat_names[j]: float(w_scaled[j]) for j in range(len(feat_names))},
        "bias": float(b)
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote {args.out}")
    print("Feature priors:", priors)
    print("Weights:", out["weights"])


if __name__ == "__main__":
    main()
