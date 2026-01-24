#!/usr/bin/env python3
import argparse, json, os, pickle
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# Optional: sklearn for a clean logistic regression fit
try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

LABEL_FS_HZ = 700.0
WRIST_FS_HZ = {"ACC": 32.0, "BVP": 64.0, "EDA": 4.0, "TEMP": 4.0}
CHEST_FS_HZ = {"ECG": LABEL_FS_HZ}

def bandpass_fft(x: np.ndarray, fs_hz: float, f_lo: float = 0.7, f_hi: float = 3.0) -> np.ndarray:
    if len(x) == 0:
        return x
    x = x.astype(float) - float(np.mean(x))
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs_hz)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    X_filt = np.zeros_like(X)
    X_filt[mask] = X[mask]
    return np.fft.irfft(X_filt, n=len(x))


def detect_peaks_simple(x: np.ndarray, fs_hz: float, max_bpm: float = 180.0) -> np.ndarray:
    if len(x) < 3:
        return np.array([], dtype=int)
    mid = x[1:-1]
    peaks = np.where((mid > x[:-2]) & (mid >= x[2:]))[0] + 1
    if len(peaks) == 0:
        return np.array([], dtype=int)
    thr = float(np.mean(x) + 0.5 * np.std(x))
    peaks = peaks[x[peaks] > thr]
    if len(peaks) == 0:
        return np.array([], dtype=int)
    min_dist = int(fs_hz * 60.0 / max_bpm)
    keep: List[int] = []
    last = -min_dist
    for p in peaks:
        if p - last >= min_dist:
            keep.append(int(p))
            last = p
    return np.array(keep, dtype=int)


def ecg_peaks_from_signal(ecg: np.ndarray, fs_hz: float) -> np.ndarray:
    if len(ecg) < int(fs_hz * 5):
        return np.array([], dtype=int)
    # Basic ECG processing: bandpass for QRS, rectify + smooth, detect peaks.
    filt = bandpass_fft(ecg, fs_hz, f_lo=5.0, f_hi=15.0)
    rect = np.abs(filt)
    win = max(3, int(0.05 * fs_hz))
    if win > 1:
        kernel = np.ones(win) / win
        smooth = np.convolve(rect, kernel, mode="same")
    else:
        smooth = rect
    return detect_peaks_simple(smooth, fs_hz, max_bpm=200.0)


def ibi_from_peaks(peaks: np.ndarray, fs_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(peaks) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    peak_times = peaks.astype(float) / fs_hz
    ibi_values = np.diff(peak_times)
    ibi_times = peak_times[1:]
    return ibi_times, ibi_values


def hr_hrv_from_ibi_window(
    ibi_times: np.ndarray,
    ibi_values: np.ndarray,
    t0: float,
    t1: float,
) -> Optional[Tuple[float, float]]:
    mask = (ibi_times >= t0) & (ibi_times < t1)
    ibi = ibi_values[mask]
    if len(ibi) < 2:
        return None
    min_bpm, max_bpm = 40.0, 200.0
    ibi = ibi[(ibi >= 60.0 / max_bpm) & (ibi <= 60.0 / min_bpm)]
    if len(ibi) < 2:
        return None
    med = float(np.median(ibi))
    ibi = ibi[(ibi > 0.85 * med) & (ibi < 1.15 * med)]
    if len(ibi) < 2:
        return None
    hr_mean = 60.0 / float(np.mean(ibi))
    sdnn_ms = float(np.std(ibi * 1000.0, ddof=1)) if len(ibi) > 1 else 0.0
    return hr_mean, sdnn_ms


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic_newton(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    for _ in range(max_iter):
        z = X @ w + b
        p = sigmoid(z)
        W = p * (1.0 - p)
        grad_w = X.T @ (p - y) + l2 * w
        grad_b = float(np.sum(p - y))
        Xw = X * W[:, None]
        H = X.T @ Xw + l2 * np.eye(d)
        try:
            step_w = np.linalg.solve(H, grad_w)
        except np.linalg.LinAlgError:
            break
        w -= step_w
        b -= grad_b / (np.sum(W) + 1e-6)
        if np.linalg.norm(step_w) < tol and abs(grad_b) < tol:
            break
    return w, b


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
    temp: np.ndarray,
    temp_fs: float,
    acc: np.ndarray,
    acc_fs: float,
    ibi_times: np.ndarray,
    ibi_values: np.ndarray,
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

    # ACC RMS magnitude
    acc_i0 = int(max(0, np.floor(t0 * acc_fs)))
    acc_i1 = int(min(len(acc), np.ceil(t1 * acc_fs)))
    if acc_i1 - acc_i0 < max(10, int(0.25 * (t1 - t0) * acc_fs)):
        return None
    seg = acc[acc_i0:acc_i1, :]
    # ACC is in "1/64g" in raw E4 files for WESAD; convert to g.
    seg_g = seg / 64.0
    mag = np.linalg.norm(seg_g, axis=1)
    acc_rms = float(np.std(mag, ddof=1))

    # ECG peaks -> HR mean + SDNN
    hr_res = hr_hrv_from_ibi_window(ibi_times, ibi_values, t0, t1)
    if hr_res is None:
        return None
    hr_mean, sdnn_ms = hr_res

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
    ap.add_argument("--weight_mode", choices=["effect_size", "linear"], default="effect_size",
                    help="Weighting strategy: effect_size (default) or linear (logistic/least squares)")
    ap.add_argument("--use_sklearn", action="store_true", help="Use sklearn LogisticRegression if available (linear mode only)")
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

        signals = pkl_data.get("signal") or {}
        wrist = signals.get("wrist") or {}
        chest = signals.get("chest") or {}
        temp = wrist.get("TEMP")
        acc = wrist.get("ACC")
        ecg = chest.get("ECG")
        if temp is None or acc is None or ecg is None:
            print(f"[skip] {sid}: missing wrist/chest signals in {sid}.pkl")
            continue
        temp = np.asarray(temp)
        acc = np.asarray(acc)
        ecg = np.asarray(ecg)
        temp_fs = WRIST_FS_HZ["TEMP"]
        acc_fs = WRIST_FS_HZ["ACC"]
        ecg_fs = CHEST_FS_HZ["ECG"]

        ecg_peaks = ecg_peaks_from_signal(ecg[:, 0] if ecg.ndim > 1 else ecg, ecg_fs)
        ibi_times, ibi_values = ibi_from_peaks(ecg_peaks, ecg_fs)
        if len(ibi_values) < 2:
            print(f"[skip] {sid}: insufficient ECG peaks for HRV")
            continue

        duration_s = min(
            len(labels) / LABEL_FS_HZ,
            len(temp) / temp_fs,
            len(acc) / acc_fs,
            len(ecg) / ecg_fs,
        )

        t = 0.0
        while t + args.window_s <= duration_s:
            t0, t1 = t, t + args.window_s

            # Window label = majority label in the interval
            i0 = int(t0 * LABEL_FS_HZ)
            i1 = int(t1 * LABEL_FS_HZ)
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
                temp=temp, temp_fs=temp_fs,
                acc=acc, acc_fs=acc_fs,
                ibi_times=ibi_times, ibi_values=ibi_values,
            )
            if feats is None:
                t += args.stride_s
                continue

            row = [feats.hr_mean_bpm, feats.hrv_sdnn_ms, feats.wrist_temp_c, feats.acc_rms_g]
            if not np.all(np.isfinite(row)):
                # Skip rows with invalid HR/HRV extraction.
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

    if args.weight_mode == "effect_size":
        # Univariate effect size: stress minus baseline in baseline std units.
        X_stress = X[y == 1]
        w = np.zeros(X.shape[1], dtype=float)
        for j, name in enumerate(feat_names):
            mu_base = priors[name]["mean"]
            sd_base = priors[name]["std"]
            mu_stress = float(np.mean(X_stress[:, j])) if len(X_stress) else mu_base
            w[j] = (mu_stress - mu_base) / (sd_base + 1e-6)
        b = 0.0
    else:
        # Fit simple model: stress vs baseline
        if args.use_sklearn and SKLEARN_AVAILABLE:
            clf = LogisticRegression(max_iter=10000)
            clf.fit(Xz, y)
            w = clf.coef_.reshape(-1)
            b = float(clf.intercept_[0])
        else:
            w, b = fit_logistic_newton(Xz, y, l2=1.0)

    out = {
        "meta": {
            "source": "WESAD",
            "labels": {"baseline": 1, "stress": 2},
            "window_s": args.window_s,
            "stride_s": args.stride_s,
            "notes": "Signals from WESAD .pkl wrist/chest data aligned to labels. HR/HRV estimated from chest ECG via FFT bandpass + peak detection. Weights use effect_size mode unless linear is selected."
        },
        "priors": priors,
        "weights": {feat_names[j]: float(w[j]) for j in range(len(feat_names))},
        "bias": float(b)
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
