#!/usr/bin/env python3
"""
emo_train.py
============
Trains a linear stress-scoring model  z = W·x + b  from the K-EmoPhone (EMO)
dataset and writes a priors.json file compatible with ScoreEngine.swift.

Features (all available in HealthKit):
  HR             – heart rate (bpm)
  HRV            – SDNN of RR-intervals (ms)
  skinTemperature – wrist skin temperature (°C)
  UltraViolet    – UV intensity encoded (NONE=0 … VERY_HIGH=4)
  stepCount      – steps accumulated in the lookback window
  Calorie        – active calories accumulated in the window (kcal)
  Distance       – distance accumulated in the window (m)

Stress labels come from the ESM self-reports (EsmResponse.csv):
  stress > 0  → stressed  (y = 1)
  stress ≤ 0  → baseline  (y = 0)

Split:  first 50 subjects → train
        next  20 subjects → validation
        last  10 subjects → test  (held-out, not used during training)

Training uses PyTorch logistic regression with CUDA if a GPU is available
(e.g. NVIDIA 5070 Ti), otherwise falls back to CPU.

Usage:
    python emo_train.py --emo_root ./emo --out priors.json
    python emo_train.py --emo_root ./emo --window_min 30 --out priors.json
    python emo_train.py --emo_root ./emo --epochs 500 --lr 0.05 --out priors.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PyTorch import (optional — falls back to NumPy Newton-Raphson)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    print("[warn] PyTorch not found — using NumPy Newton-Raphson fallback.", file=sys.stderr)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: List[str] = [
    "HR",
    "HRV",
    "skinTemperature",
    "UltraViolet",
    "stepCount",
    "Calorie",
    "Distance",
]

UV_MAP: Dict[str, float] = {
    "NONE": 0.0,
    "LOW": 1.0,
    "MODERATE": 2.0,
    "HIGH": 3.0,
    "VERY_HIGH": 4.0,
    "EXTREME": 5.0,
}

WINDOW_DEFAULT_MIN = 30   # minutes of sensor data before each ESM response


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    """Load a CSV file, returning None if it doesn't exist or is empty."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception:
        return None


def sdnn_ms(rri_ms: np.ndarray) -> float:
    """Compute SDNN (std of NN intervals) in milliseconds."""
    if len(rri_ms) < 2:
        return float("nan")
    # Basic ectopic-beat filter: keep only IBIs within ±20% of the median
    med = float(np.median(rri_ms))
    rri_clean = rri_ms[(rri_ms > 0.80 * med) & (rri_ms < 1.20 * med)]
    if len(rri_clean) < 2:
        return float("nan")
    return float(np.std(rri_clean, ddof=1))


def cumulative_delta(df: pd.DataFrame, col: str, t0_ms: int, t1_ms: int) -> float:
    """
    For cumulative-today columns (stepsToday, caloriesToday, distanceToday):
    return the increase within [t0_ms, t1_ms].
    """
    mask = (df["timestamp"] >= t0_ms) & (df["timestamp"] <= t1_ms)
    sub = df.loc[mask, col]
    if len(sub) < 2:
        return float("nan")
    return float(sub.iloc[-1] - sub.iloc[0])


def robust_mean_std(x: np.ndarray) -> Tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0, 1.0
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else 1.0
    return mu, max(sd, 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction for a single ESM event
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    subj_dir: str,
    response_ts_ms: int,
    window_ms: int,
) -> Optional[np.ndarray]:
    """
    Return a 7-element feature vector for one ESM event, or None if data
    is insufficient.

    Sensor data window: [response_ts_ms - window_ms, response_ts_ms]
    """
    t0 = response_ts_ms - window_ms
    t1 = response_ts_ms

    # ── HR ────────────────────────────────────────────────────────────────
    hr_df = load_csv_safe(os.path.join(subj_dir, "HR.csv"))
    if hr_df is None:
        return None
    mask_hr = (hr_df["timestamp"] >= t0) & (hr_df["timestamp"] <= t1)
    hr_vals = hr_df.loc[mask_hr, "bpm"].values.astype(float)
    if len(hr_vals) < 3:
        return None
    feat_hr = float(np.mean(hr_vals))

    # ── HRV (from RRI) ────────────────────────────────────────────────────
    rri_df = load_csv_safe(os.path.join(subj_dir, "RRI.csv"))
    feat_hrv = float("nan")
    if rri_df is not None:
        mask_rri = (rri_df["timestamp"] >= t0) & (rri_df["timestamp"] <= t1)
        rri_vals = rri_df.loc[mask_rri, "interval"].values.astype(float)
        feat_hrv = sdnn_ms(rri_vals)

    # ── Skin Temperature ──────────────────────────────────────────────────
    temp_df = load_csv_safe(os.path.join(subj_dir, "SkinTemperature.csv"))
    feat_temp = float("nan")
    if temp_df is not None:
        mask_t = (temp_df["timestamp"] >= t0) & (temp_df["timestamp"] <= t1)
        temp_vals = temp_df.loc[mask_t, "temperature"].values.astype(float)
        if len(temp_vals) >= 1:
            feat_temp = float(np.mean(temp_vals))

    # ── UV ────────────────────────────────────────────────────────────────
    uv_df = load_csv_safe(os.path.join(subj_dir, "UltraViolet.csv"))
    feat_uv = 0.0   # default: NONE (indoors / no reading)
    if uv_df is not None:
        mask_uv = (uv_df["timestamp"] >= t0) & (uv_df["timestamp"] <= t1)
        uv_sub = uv_df.loc[mask_uv]
        if not uv_sub.empty:
            encoded = uv_sub["intensity"].map(
                lambda s: UV_MAP.get(str(s).strip().upper(), 0.0)
            ).values.astype(float)
            feat_uv = float(np.mean(encoded))

    # ── Step Count ────────────────────────────────────────────────────────
    step_df = load_csv_safe(os.path.join(subj_dir, "StepCount.csv"))
    feat_steps = float("nan")
    if step_df is not None:
        feat_steps = cumulative_delta(step_df, "stepsToday", t0, t1)

    # ── Calorie ───────────────────────────────────────────────────────────
    cal_df = load_csv_safe(os.path.join(subj_dir, "Calorie.csv"))
    feat_cal = float("nan")
    if cal_df is not None:
        feat_cal = cumulative_delta(cal_df, "caloriesToday", t0, t1)

    # ── Distance ──────────────────────────────────────────────────────────
    dist_df = load_csv_safe(os.path.join(subj_dir, "Distance.csv"))
    feat_dist = float("nan")
    if dist_df is not None:
        feat_dist = cumulative_delta(dist_df, "distanceToday", t0, t1)

    vec = np.array([
        feat_hr,
        feat_hrv,
        feat_temp,
        feat_uv,
        feat_steps,
        feat_cal,
        feat_dist,
    ], dtype=float)

    # Require at minimum HR to be valid
    if not np.isfinite(vec[0]):
        return None

    return vec


# ─────────────────────────────────────────────────────────────────────────────
# Logistic regression (Newton-Raphson, pure NumPy)
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def fit_logistic_numpy(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> Tuple[np.ndarray, float]:
    """
    Fit  P(y=1) = σ(X w + b)  via Newton-Raphson with L2 regularisation.
    Returns (weights, bias).
    """
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    for _ in range(max_iter):
        z = X @ w + b
        p = sigmoid(z)
        r = p * (1.0 - p)                          # IRLS weights
        grad_w = X.T @ (p - y) + l2 * w
        grad_b = float(np.sum(p - y))
        H = X.T @ (X * r[:, None]) + l2 * np.eye(d)
        try:
            dw = np.linalg.solve(H, grad_w)
        except np.linalg.LinAlgError:
            break
        db = grad_b / (float(np.sum(r)) + 1e-9)
        w -= dw
        b -= db
        if np.linalg.norm(dw) < tol and abs(db) < tol:
            break
    return w, b


def fit_logistic_torch(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
    epochs: int = 2000,
    lr: float = 0.05,
    batch_size: int = 512,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, float]:
    """
    Fit logistic regression via mini-batch SGD (Adam) on GPU/CPU using PyTorch.
    Falls back to CPU if CUDA is not available.
    Returns (weights, bias) as numpy arrays.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [torch] Training on device: {device}")
    if device == "cuda":
        print(f"  [torch] GPU: {torch.cuda.get_device_name(0)}")

    dev = torch.device(device)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)

    n, d = X.shape
    model = nn.Linear(d, 1, bias=True).to(dev)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2 / n)
    bce = nn.BCEWithLogitsLoss()

    dataset = torch.utils.data.TensorDataset(Xt, yt)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    patience, patience_cnt = 30, 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = bce(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= n

        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  [torch] Early stop at epoch {epoch+1}  loss={epoch_loss:.5f}")
                break

        if (epoch + 1) % 200 == 0:
            print(f"  [torch] epoch {epoch+1:4d}  loss={epoch_loss:.5f}")

    w_np = model.weight.detach().cpu().numpy().flatten()
    b_np = float(model.bias.detach().cpu().item())
    return w_np, b_np


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
    use_gpu: bool = True,
    epochs: int = 2000,
    lr: float = 0.05,
) -> Tuple[np.ndarray, float]:
    """
    Dispatch to PyTorch GPU trainer if available, else NumPy Newton-Raphson.
    """
    if _TORCH_AVAILABLE and use_gpu:
        return fit_logistic_torch(X, y, l2=l2, epochs=epochs, lr=lr)
    return fit_logistic_numpy(X, y, l2=l2)


def accuracy(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    preds = (sigmoid(X @ w + b) >= 0.5).astype(int)
    return float(np.mean(preds == y))


# ─────────────────────────────────────────────────────────────────────────────
# Scale raw weights to [0, 100]
# ─────────────────────────────────────────────────────────────────────────────

def scale_weights_0_100(w: np.ndarray) -> np.ndarray:
    """
    Linearly scale weights so the maximum absolute value maps to 100.
    Sign is preserved (positive = feature increases under stress).
    If all weights are zero, return zeros.
    """
    max_abs = float(np.max(np.abs(w)))
    if max_abs < 1e-9:
        return np.zeros_like(w)
    return w * (100.0 / max_abs)


# ─────────────────────────────────────────────────────────────────────────────
# Impute NaN values in the feature matrix (column median of training data)
# ─────────────────────────────────────────────────────────────────────────────

def impute(
    X_train: np.ndarray,
    *extras: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """Replace NaN entries with the column median from X_train."""
    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)

    def _fill(X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for j in range(X.shape[1]):
            nan_mask = ~np.isfinite(X[:, j])
            X[nan_mask, j] = medians[j]
        return X

    return (_fill(X_train),) + tuple(_fill(X) for X in extras)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emo_root", default="./emo",
                    help="Path to the emo/ directory containing P01/, P02/, … and SubjData/")
    ap.add_argument("--window_min", type=float, default=WINDOW_DEFAULT_MIN,
                    help="Lookback window in minutes before each ESM response (default: 30)")
    ap.add_argument("--l2", type=float, default=1.0,
                    help="L2 regularisation strength (default: 1.0)")
    ap.add_argument("--out", default="priors.json",
                    help="Output JSON path (default: priors.json)")
    ap.add_argument("--n_train", type=int, default=50)
    ap.add_argument("--n_val",   type=int, default=20)
    ap.add_argument("--n_test",  type=int, default=10)
    ap.add_argument("--no_gpu",  action="store_true",
                    help="Disable GPU even if CUDA is available (force NumPy path)")
    ap.add_argument("--epochs",  type=int, default=2000,
                    help="Training epochs for PyTorch trainer (default: 2000)")
    ap.add_argument("--lr",      type=float, default=0.05,
                    help="Learning rate for Adam optimiser (default: 0.05)")
    args = ap.parse_args()

    # ── GPU check ─────────────────────────────────────────────────────────
    if _TORCH_AVAILABLE and not args.no_gpu:
        if torch.cuda.is_available():
            print(f"[GPU] CUDA available — {torch.cuda.get_device_name(0)}")
        else:
            print("[GPU] CUDA not available — falling back to NumPy Newton-Raphson.")

    emo_root   = args.emo_root
    window_ms  = int(args.window_min * 60 * 1000)

    # ── Discover subjects ─────────────────────────────────────────────────
    all_subjects = sorted(
        [d for d in os.listdir(emo_root) if d.startswith("P") and
         os.path.isdir(os.path.join(emo_root, d))],
        key=lambda s: int(s[1:])
    )
    total = len(all_subjects)
    needed = args.n_train + args.n_val + args.n_test
    if total < needed:
        print(f"[warn] Only {total} subjects found; need {needed}. "
              f"Using all {total} (train={args.n_train}, val={total-args.n_train-args.n_test}, test={args.n_test}).",
              file=sys.stderr)

    train_subj = all_subjects[:args.n_train]
    val_subj   = all_subjects[args.n_train : args.n_train + args.n_val]
    test_subj  = all_subjects[args.n_train + args.n_val : args.n_train + args.n_val + args.n_test]

    print(f"Split → train={len(train_subj)}, val={len(val_subj)}, test={len(test_subj)}")

    # ── Load ESM labels ───────────────────────────────────────────────────
    esm_path = os.path.join(emo_root, "SubjData", "EsmResponse.csv")
    if not os.path.exists(esm_path):
        sys.exit(f"[error] ESM file not found: {esm_path}")
    esm = pd.read_csv(esm_path)
    # stress column: -3 … +3; positive → stressed
    esm["label"] = (esm["stress"] > 0).astype(int)

    # ── Feature extraction ────────────────────────────────────────────────
    def collect_split(subjects: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        X_rows, y_rows = [], []
        for sid in subjects:
            subj_dir = os.path.join(emo_root, sid)
            esm_sub = esm[esm["pcode"] == sid]
            if esm_sub.empty:
                print(f"  [skip] {sid}: no ESM responses")
                continue
            n_ok = 0
            for _, row in esm_sub.iterrows():
                ts_ms = int(row["responseTime"])
                feat = extract_features(subj_dir, ts_ms, window_ms)
                if feat is None:
                    continue
                X_rows.append(feat)
                y_rows.append(int(row["label"]))
                n_ok += 1
            print(f"  [ok] {sid}: {n_ok}/{len(esm_sub)} windows extracted")
        if not X_rows:
            return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=int)
        return np.array(X_rows, dtype=float), np.array(y_rows, dtype=int)

    print("\n── Training subjects ──")
    X_train, y_train = collect_split(train_subj)
    print("\n── Validation subjects ──")
    X_val, y_val     = collect_split(val_subj)
    print("\n── Test subjects ──")
    X_test, y_test   = collect_split(test_subj)

    if len(X_train) < 10:
        sys.exit(f"[error] Too few training samples: {len(X_train)}. Check --emo_root path.")

    print(f"\nSamples → train={len(X_train)} (stress={int(y_train.sum())}), "
          f"val={len(X_val)} (stress={int(y_val.sum())}), "
          f"test={len(X_test)} (stress={int(y_test.sum())})")

    # ── Impute missing values ─────────────────────────────────────────────
    X_train, X_val, X_test = impute(X_train, X_val, X_test)

    # ── Compute baseline (non-stressed) priors ────────────────────────────
    X_base = X_train[y_train == 0]
    priors: Dict[str, Dict[str, float]] = {}
    for j, name in enumerate(FEATURE_NAMES):
        mu, sd = robust_mean_std(X_base[:, j])
        priors[name] = {"mean": round(mu, 6), "std": round(sd, 6)}

    # ── Standardise using baseline µ/σ ────────────────────────────────────
    def standardise(X: np.ndarray) -> np.ndarray:
        Xz = np.zeros_like(X)
        for j, name in enumerate(FEATURE_NAMES):
            mu = priors[name]["mean"]
            sd = priors[name]["std"]
            Xz[:, j] = (X[:, j] - mu) / (sd + 1e-9)
        return Xz

    Xz_train = standardise(X_train)
    Xz_val   = standardise(X_val)
    Xz_test  = standardise(X_test)

    # ── Fit logistic regression ───────────────────────────────────────────
    use_gpu = not args.no_gpu
    if _TORCH_AVAILABLE and use_gpu:
        print(f"\nFitting logistic regression (PyTorch, epochs={args.epochs}, lr={args.lr}) …")
    else:
        print("\nFitting logistic regression (NumPy Newton-Raphson) …")
    w_raw, b_raw = fit_logistic(Xz_train, y_train, l2=args.l2, use_gpu=use_gpu,
                                epochs=args.epochs, lr=args.lr)

    acc_train = accuracy(Xz_train, y_train, w_raw, b_raw)
    acc_val   = accuracy(Xz_val,   y_val,   w_raw, b_raw) if len(X_val) > 0 else float("nan")
    acc_test  = accuracy(Xz_test,  y_test,  w_raw, b_raw) if len(X_test) > 0 else float("nan")

    print(f"  Accuracy → train={acc_train:.3f}  val={acc_val:.3f}  test={acc_test:.3f}")

    # ── Scale weights to [0, 100] ─────────────────────────────────────────
    w_scaled = scale_weights_0_100(w_raw)

    print("\n── Learned feature weights (0-100 scale) ──")
    for name, ws, wr in zip(FEATURE_NAMES, w_scaled, w_raw):
        direction = "↑ stress" if wr > 0 else "↓ stress"
        print(f"  {name:<20s}  {ws:+7.2f}  (raw: {wr:+.4f})  [{direction}]")
    print(f"  {'bias':<20s}  {b_raw:+.4f}")

    # ── Build output JSON (compatible with ScoreEngine.swift) ─────────────
    out = {
        "meta": {
            "source": "K-EmoPhone (EMO)",
            "split": {
                "train": len(train_subj),
                "val": len(val_subj),
                "test": len(test_subj),
            },
            "n_samples": {
                "train": len(X_train),
                "val": len(X_val),
                "test": len(X_test),
            },
            "accuracy": {
                "train": round(acc_train, 4),
                "val": round(float(acc_val), 4),
                "test": round(float(acc_test), 4),
            },
            "window_min": args.window_min,
            "l2": args.l2,
            "labels": {
                "stressed": "ESM stress > 0",
                "baseline": "ESM stress <= 0",
            },
            "features": FEATURE_NAMES,
            "notes": (
                "weights contains raw logistic regression coefficients (not rescaled). "
                "weights_display contains the same values linearly scaled to [-100, +100] "
                "for human-readable interpretation only. "
                "Score = sigmoid(b + sum(w_i * z_i)) * 100 where z_i = clip((x_i - mu_i)/sigma_i, -3, 3)."
            ),
        },
        "priors": priors,
        # Raw logistic coefficients — used by ScoreEngine for Wx+b
        "weights": {name: round(float(w_raw[j]), 6) for j, name in enumerate(FEATURE_NAMES)},
        # Display-only scaled version (max|w|=100) — for UI labels
        "weights_display": {name: round(float(w_scaled[j]), 4) for j, name in enumerate(FEATURE_NAMES)},
        "bias": round(float(b_raw), 6),
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n✓ Wrote {args.out}")
    print("\nNext steps:")
    print("  1. Add priors.json to your Xcode target (Copy Bundle Resources).")
    print("  2. Update ScoreEngine.swift Feature enum with the 7 new feature keys.")
    print("  3. Run validation / test splits independently before shipping.")


if __name__ == "__main__":
    main()
