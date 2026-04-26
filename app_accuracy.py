#!/usr/bin/env python3
"""
app_accuracy.py
===============
Full training → validation → testing pipeline that also generates priors.json.

Pipeline
--------
  PART 1 (train)   47 subjects  →  fit logistic regression on 7 base features
                                   compute priors (μ/σ) from baseline samples
                                   write priors.json  (ScoreEngine.swift compatible)
  PART 2 (compare) optional     →  train RF or XGBoost on engineered features
                                   for side-by-side accuracy comparison
  PART 3 (test)    10 subjects  →  score with app logistic model, per-subject report
                                   this is the *theoretical app accuracy*

Scoring (mirrors ScoreEngine.swift exactly):
  z_i   = clip((x_i - μ_i) / σ_i, -3, +3)
  score = sigmoid(b + Σ w_i·z_i) × 100
  label = stressed  if score < 50  else  calm

Usage
-----
    python app_accuracy.py --emo_root ./emo
    python app_accuracy.py --emo_root ./emo --out WatchStress/priors.json
    python app_accuracy.py --emo_root ./emo --model xgb --out priors.json
    python app_accuracy.py --emo_root ./emo --priors old_priors.json  # compare
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.inspection import permutation_importance as sklearn_perm_importance
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    print("[warn] scikit-learn not found — Random Forest unavailable.", file=sys.stderr)

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    print("[warn] xgboost not found — XGBoost unavailable.  pip install xgboost", file=sys.stderr)

try:
    from scipy.stats import mannwhitneyu
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    print("[warn] scipy not found — Mann-Whitney unavailable.  pip install scipy", file=sys.stderr)

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    print("[warn] shap not found — SHAP analysis unavailable.  pip install shap", file=sys.stderr)

# ─────────────────────────────────────────────────────────────────────────────
# Constants (must match ScoreEngine.swift and emo_train.py)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: List[str] = [
    "HR",
    "HRV",
    "UltraViolet",
    "Calorie",
]

UV_MAP: Dict[str, float] = {
    "NONE": 0.0,
    "LOW": 1.0,
    "MODERATE": 2.0,
    "HIGH": 3.0,
    "VERY_HIGH": 4.0,
    "EXTREME": 5.0,
}

WINDOW_DEFAULT_MIN = 30
Z_CLAMP = 3.0   # ScoreEngine clamps z-scores to ±3


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — load each CSV once per subject, then vectorize windows
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception:
        return None


def rmssd_ms(rri_ms: np.ndarray) -> float:
    """RMSSD — root mean square of successive RR differences.
    More sensitive to parasympathetic (HF) activity than SDNN;
    drops under stress as vagal tone withdraws."""
    if len(rri_ms) < 3:
        return float("nan")
    med = float(np.median(rri_ms))
    rri_clean = rri_ms[(rri_ms > 0.80 * med) & (rri_ms < 1.20 * med)]
    if len(rri_clean) < 3:
        return float("nan")
    diffs = np.diff(rri_clean)
    return float(np.sqrt(np.mean(diffs ** 2)))


def load_subject_data(subj_dir: str) -> Dict[str, np.ndarray]:
    """
    Load and sort all sensor CSVs for one subject into plain NumPy arrays.
    Called once per subject; all ESM windows are then computed without any
    further disk I/O.
    """
    d: Dict[str, np.ndarray] = {}

    def _load_sorted(fname: str, *cols: str):
        df = load_csv_safe(os.path.join(subj_dir, fname))
        if df is None:
            return
        df = df.sort_values("timestamp")
        d[fname + "_ts"] = df["timestamp"].values.astype(np.int64)
        for col in cols:
            if col in df.columns:
                d[fname + "_" + col] = df[col].values.astype(float)

    _load_sorted("HR.csv", "bpm")
    _load_sorted("RRI.csv", "interval")
    _load_sorted("Calorie.csv", "caloriesToday")

    # UV needs string-to-float mapping before storing
    uv_df = load_csv_safe(os.path.join(subj_dir, "UltraViolet.csv"))
    if uv_df is not None:
        uv_df = uv_df.sort_values("timestamp")
        d["UltraViolet.csv_ts"] = uv_df["timestamp"].values.astype(np.int64)
        d["UltraViolet.csv_intensity"] = (
            uv_df["intensity"]
            .map(lambda s: UV_MAP.get(str(s).strip().upper(), 0.0))
            .values.astype(float)
        )

    return d


def _window_mean(ts: np.ndarray, vals: np.ndarray, t0s: np.ndarray, t1s: np.ndarray,
                 min_count: int = 1) -> np.ndarray:
    """Mean of vals in [t0, t1] for each row. Returns nan where count < min_count."""
    out = np.full(len(t0s), np.nan)
    for i in range(len(t0s)):
        lo = np.searchsorted(ts, t0s[i])
        hi = np.searchsorted(ts, t1s[i], side="right")
        if hi - lo >= min_count:
            out[i] = vals[lo:hi].mean()
    return out


def _window_rmssd(ts: np.ndarray, vals: np.ndarray, t0s: np.ndarray, t1s: np.ndarray) -> np.ndarray:
    out = np.full(len(t0s), np.nan)
    for i in range(len(t0s)):
        lo = np.searchsorted(ts, t0s[i])
        hi = np.searchsorted(ts, t1s[i], side="right")
        out[i] = rmssd_ms(vals[lo:hi])
    return out


def _window_cumulative_delta(ts: np.ndarray, vals: np.ndarray,
                              t0s: np.ndarray, t1s: np.ndarray) -> np.ndarray:
    """vals[-1] - vals[0] within [t0, t1]. Returns nan where fewer than 2 points."""
    out = np.full(len(t0s), np.nan)
    for i in range(len(t0s)):
        lo = np.searchsorted(ts, t0s[i])
        hi = np.searchsorted(ts, t1s[i], side="right")
        if hi - lo >= 2:
            out[i] = vals[hi - 1] - vals[lo]
    return out


def extract_features_batch(
    data: Dict[str, np.ndarray],
    timestamps_ms: np.ndarray,
    window_ms: int,
) -> np.ndarray:
    """Return (N, 4) feature matrix for all N ESM timestamps at once.
    Features: HR, HRV (RMSSD), UltraViolet, Calorie."""
    t0s = timestamps_ms - window_ms
    t1s = timestamps_ms
    n   = len(timestamps_ms)
    X   = np.full((n, len(FEATURE_NAMES)), np.nan)

    if "HR.csv_ts" in data:
        X[:, 0] = _window_mean(data["HR.csv_ts"], data["HR.csv_bpm"], t0s, t1s, min_count=3)
    if "RRI.csv_ts" in data:
        X[:, 1] = _window_rmssd(data["RRI.csv_ts"], data["RRI.csv_interval"], t0s, t1s)
    # UV defaults to 0.0 (NONE) when no reading in window
    X[:, 2] = 0.0
    if "UltraViolet.csv_ts" in data:
        uv = _window_mean(data["UltraViolet.csv_ts"], data["UltraViolet.csv_intensity"], t0s, t1s)
        X[:, 2] = np.where(np.isfinite(uv), uv, 0.0)
    if "Calorie.csv_ts" in data:
        X[:, 3] = _window_cumulative_delta(
            data["Calorie.csv_ts"], data["Calorie.csv_caloriesToday"], t0s, t1s
        )

    return X


def _process_subject(
    sid: str,
    subj_dir: str,
    esm_sub: pd.DataFrame,
    window_ms: int,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Load one subject's data and extract all features. Runs in a thread."""
    sensor_data = load_subject_data(subj_dir)
    if "HR.csv_ts" not in sensor_data:
        return sid, np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=int)

    timestamps = esm_sub["responseTime"].values.astype(np.int64)
    labels     = esm_sub["label"].values.astype(int)

    X_batch = extract_features_batch(sensor_data, timestamps, window_ms)

    # Drop rows without valid HR (same requirement as original code)
    valid = np.isfinite(X_batch[:, 0])
    return sid, X_batch[valid], labels[valid]


def collect_split(
    subjects: List[str],
    emo_root: str,
    esm: pd.DataFrame,
    window_ms: int,
    n_workers: int = 8,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract features for all subjects in parallel (ThreadPoolExecutor).
    Each subject's CSVs are loaded exactly once; windows are computed with
    vectorized NumPy searchsorted rather than per-row Python loops.
    """
    tasks = {
        sid: esm[esm["pcode"] == sid]
        for sid in subjects
        if not esm[esm["pcode"] == sid].empty
    }
    skipped = [s for s in subjects if s not in tasks]
    for s in skipped:
        print(f"  [skip] {s}: no ESM responses")

    X_list, y_list, sid_list = [], [], []
    results = {}

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _process_subject,
                sid,
                os.path.join(emo_root, sid),
                esm_sub,
                window_ms,
            ): sid
            for sid, esm_sub in tasks.items()
        }
        for fut in as_completed(futures):
            sid, X_batch, y_batch = fut.result()
            results[sid] = (X_batch, y_batch)

    # Print in subject order for readable output
    for sid in subjects:
        if sid not in results:
            continue
        X_batch, y_batch = results[sid]
        n_total = len(esm[esm["pcode"] == sid])
        print(f"  [ok] {sid}: {len(X_batch)}/{n_total} windows extracted")
        if len(X_batch) > 0:
            X_list.append(X_batch)
            y_list.extend(y_batch.tolist())
            sid_list.extend([sid] * len(X_batch))

    if not X_list:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=int), []
    return np.vstack(X_list), np.array(y_list, dtype=int), sid_list


# ─────────────────────────────────────────────────────────────────────────────
# Imputation and standardisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def impute(X_train: np.ndarray, *extras: np.ndarray) -> Tuple[np.ndarray, ...]:
    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)

    def _fill(X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for j in range(X.shape[1]):
            nan_mask = ~np.isfinite(X[:, j])
            X[nan_mask, j] = medians[j]
        return X

    return (_fill(X_train),) + tuple(_fill(X) for X in extras)


def standardise(X: np.ndarray, priors: Dict) -> np.ndarray:
    """
    Standardise X using the priors dict loaded from priors.json.
    Mirrors ScoreEngine.swift: z = clip((x - mu) / sigma, -3, +3)
    """
    Xz = np.zeros_like(X)
    for j, name in enumerate(FEATURE_NAMES):
        mu = priors[name]["mean"]
        sd = priors[name]["std"]
        Xz[:, j] = np.clip((X[:, j] - mu) / (sd + 1e-9), -Z_CLAMP, Z_CLAMP)
    return Xz


# ─────────────────────────────────────────────────────────────────────────────
# Logistic regression trainers (from emo_train.py, unchanged)
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
    n, d = X.shape
    n_pos = int(y.sum())
    n_neg = n - n_pos
    # Class-balanced sample weights: each class contributes equally regardless of size
    sw = np.where(y == 1, n / (2.0 * max(n_pos, 1)), n / (2.0 * max(n_neg, 1)))
    w = np.zeros(d, dtype=float)
    b = 0.0
    for _ in range(max_iter):
        z = X @ w + b
        p = sigmoid(z)
        r = sw * p * (1.0 - p)
        residuals = sw * (p - y)
        grad_w = X.T @ residuals + l2 * w
        grad_b = float(np.sum(residuals))
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
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [torch] Training on device: {device}")
    dev = torch.device(device)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    n, d = X.shape
    model = nn.Linear(d, 1, bias=True).to(dev)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2 / n)
    n_pos = int((y == 1).sum())
    n_neg = n - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dataset = torch.utils.data.TensorDataset(Xt, yt)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    best_loss, patience, patience_cnt = float("inf"), 30, 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = bce(model(xb).squeeze(1), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= n
        if epoch_loss < best_loss - 1e-6:
            best_loss, patience_cnt = epoch_loss, 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  [torch] Early stop at epoch {epoch+1}  loss={epoch_loss:.5f}")
                break
        if (epoch + 1) % 200 == 0:
            print(f"  [torch] epoch {epoch+1:4d}  loss={epoch_loss:.5f}")
    return (
        model.weight.detach().cpu().numpy().flatten(),
        float(model.bias.detach().cpu().item()),
    )


def fit_logistic(X, y, l2=1.0, use_gpu=True, epochs=2000, lr=0.05, device="cpu"):
    if _TORCH_AVAILABLE and use_gpu:
        return fit_logistic_torch(X, y, l2=l2, epochs=epochs, lr=lr, device=device)
    return fit_logistic_numpy(X, y, l2=l2)


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest (scikit-learn)
# ─────────────────────────────────────────────────────────────────────────────

def fit_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 5,
) -> "RandomForestClassifier":
    if not _SKLEARN_AVAILABLE:
        sys.exit("[error] scikit-learn is required for --model rf.  pip install scikit-learn")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",   # handles stressed/calm imbalance
        n_jobs=-1,                 # use all CPU cores
        random_state=42,
    )
    clf.fit(X, y)
    return clf


def rf_accuracy(clf: "RandomForestClassifier", X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(clf.predict(X) == y))


def rf_per_subject_accuracy(
    clf: "RandomForestClassifier",
    X: np.ndarray,
    y: np.ndarray,
    sids: List[str],
) -> Dict[str, Dict]:
    preds = clf.predict(X)
    results = {}
    for sid in sorted(set(sids)):
        idx = [i for i, s in enumerate(sids) if s == sid]
        ys, ps = y[idx], preds[idx]
        n_stressed = int(ys.sum())
        results[sid] = {
            "n_samples": len(ys),
            "n_stressed": n_stressed,
            "n_calm": len(ys) - n_stressed,
            "accuracy": float(np.mean(ps == ys)),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

# Indices into the raw 7-feature vector
_HR, _HRV, _UV, _CAL = range(4)

ENGINEERED_NAMES: List[str] = [
    "hr_hrv_ratio",   # high HR + low HRV = classic stress marker
    "hrv_hr_product", # joint autonomic balance signal
]


def engineer_features(X: np.ndarray) -> np.ndarray:
    """
    Append 2 derived features to the raw 3-column matrix.
    NaN-safe: division where denominator is 0 stays NaN.
    Returns shape (N, 5).
    """
    n = X.shape[0]
    feats = np.full((n, len(ENGINEERED_NAMES)), np.nan)

    hr  = X[:, _HR]
    hrv = X[:, _HRV]

    with np.errstate(divide="ignore", invalid="ignore"):
        feats[:, 0] = np.where(hrv > 0, hr / hrv, np.nan)  # hr_hrv_ratio
        feats[:, 1] = hr * hrv                               # hrv_hr_product

    return np.hstack([X, feats])


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def fit_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int = 600,
    max_depth: int = 4,
    lr: float = 0.05,
    device: str = "cpu",
) -> "XGBClassifier":
    if not _XGB_AVAILABLE:
        sys.exit("[error] xgboost is required for --model xgb.  pip install xgboost")

    # Balance classes
    n_calm    = int((y_train == 0).sum())
    n_stressed = int((y_train == 1).sum())
    scale_pos = n_calm / max(n_stressed, 1)

    xgb_device = "cuda" if device == "cuda" else "cpu"

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        early_stopping_rounds=30,
        device=xgb_device,
        random_state=42,
        verbosity=0,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"  [xgb] Best iteration: {clf.best_iteration}  "
          f"val-logloss: {clf.best_score:.4f}  device: {xgb_device}")
    return clf


def scale_weights_0_100(w: np.ndarray) -> np.ndarray:
    """Linearly scale weights so the maximum absolute value maps to 100."""
    max_abs = float(np.max(np.abs(w)))
    if max_abs < 1e-9:
        return np.zeros_like(w)
    return w * (100.0 / max_abs)


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance analysis — 7 complementary methods
# ─────────────────────────────────────────────────────────────────────────────

def _cohens_d(s: np.ndarray, c: np.ndarray) -> float:
    """Cohen's d: (stressed_mean − calm_mean) / pooled σ."""
    if len(s) < 2 or len(c) < 2:
        return float("nan")
    pooled = np.sqrt(
        ((len(s) - 1) * s.std(ddof=1) ** 2 + (len(c) - 1) * c.std(ddof=1) ** 2)
        / (len(s) + len(c) - 2)
    )
    return float((s.mean() - c.mean()) / (pooled + 1e-12))


def compute_feature_analysis(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sids_train: List[str],
    w_logistic: np.ndarray,
    n_rf_trees: int = 200,
) -> Dict[str, Dict[str, float]]:
    """
    Run 7 feature importance / stress-correlation methods on the 7 base features.

    Methods
    -------
    logistic_w    – raw logistic regression weight (signed)
    cohen_d       – Cohen's d: (stressed−calm) / pooled σ  (signed)
    mannwhit_p    – Mann-Whitney U p-value (unsigned; lower = more significant)
    mutual_info   – mutual information with label (unsigned)
    within_subj_r – mean within-subject Pearson r (signed)
    perm_imp      – permutation importance on val set using a dedicated RF
    shap_mean     – mean SHAP value on val set using the same RF (signed)

    Returns a nested dict: {feature_name: {method_name: float}}
    """
    results: Dict[str, Dict[str, float]] = {name: {} for name in FEATURE_NAMES}

    # NaN-safe fill with column medians (needed for model-based methods)
    def _fill(X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for j in range(X.shape[1]):
            mask = ~np.isfinite(X[:, j])
            if mask.any():
                X[mask, j] = float(np.nanmedian(X[:, j]))
        return X

    Xf_train = _fill(X_train)
    Xf_val   = _fill(X_val) if len(X_val) > 0 else Xf_train

    calm     = Xf_train[y_train == 0]
    stressed = Xf_train[y_train == 1]
    y_val_eff = y_val if len(X_val) > 0 else y_train[:len(Xf_val)]

    # 1. Logistic regression weights (already trained in PART 1)
    for j, name in enumerate(FEATURE_NAMES):
        results[name]["logistic_w"] = float(w_logistic[j])

    # 2. Cohen's d
    for j, name in enumerate(FEATURE_NAMES):
        results[name]["cohen_d"] = _cohens_d(stressed[:, j], calm[:, j])

    # 3. Mann-Whitney U p-value
    if _SCIPY_AVAILABLE:
        for j, name in enumerate(FEATURE_NAMES):
            try:
                _, p = mannwhitneyu(stressed[:, j], calm[:, j], alternative="two-sided")
                results[name]["mannwhit_p"] = float(p)
            except Exception:
                results[name]["mannwhit_p"] = float("nan")
    else:
        print("  [skip] Mann-Whitney — scipy not installed")

    # 4. Mutual information
    if _SKLEARN_AVAILABLE:
        mi = mutual_info_classif(Xf_train, y_train, random_state=42)
        for j, name in enumerate(FEATURE_NAMES):
            results[name]["mutual_info"] = float(mi[j])
    else:
        print("  [skip] Mutual information — scikit-learn not installed")

    # 5. Within-subject Pearson r
    for j, name in enumerate(FEATURE_NAMES):
        corrs = []
        for sid in sorted(set(sids_train)):
            idx = [i for i, s in enumerate(sids_train) if s == sid]
            if len(idx) < 5:
                continue
            r = np.corrcoef(Xf_train[idx, j], y_train[idx].astype(float))[0, 1]
            if np.isfinite(r):
                corrs.append(r)
        results[name]["within_subj_r"] = float(np.mean(corrs)) if corrs else float("nan")

    # 6 & 7. Permutation importance + SHAP — use a dedicated RF on 7 base features
    # (keeps all methods on the same feature space)
    if _SKLEARN_AVAILABLE:
        print(f"  [analysis] Training RF ({n_rf_trees} trees) for permutation importance & SHAP …")
        clf_rf = RandomForestClassifier(
            n_estimators=n_rf_trees,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        clf_rf.fit(Xf_train, y_train)

        perm = sklearn_perm_importance(
            clf_rf, Xf_val, y_val_eff, n_repeats=30, random_state=42
        )
        for j, name in enumerate(FEATURE_NAMES):
            results[name]["perm_imp"] = float(perm.importances_mean[j])

        if _SHAP_AVAILABLE:
            try:
                explainer = _shap.TreeExplainer(clf_rf)
                shap_vals = explainer.shap_values(Xf_val)
                # sklearn RF returns list [class_0_shap, class_1_shap]
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                for j, name in enumerate(FEATURE_NAMES):
                    results[name]["shap_mean"] = float(np.mean(shap_vals[:, j]))
            except Exception as e:
                print(f"  [warn] SHAP failed: {e}")
        else:
            print("  [skip] SHAP — shap not installed  (pip install shap)")
    else:
        print("  [skip] Permutation importance & SHAP — scikit-learn not installed")

    return results


def print_feature_table(analysis: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted value table and rank table for all analysis methods."""

    # Ordered metadata: display name, is_signed, description
    METHOD_META: Dict[str, Tuple[str, bool, str]] = {
        "logistic_w":    ("LogReg(w)",  True,  "Raw logistic weight — + = ↑ stress"),
        "cohen_d":       ("Cohen-d",    True,  "Effect size (stressed−calm)/pooled σ — + = ↑ stress"),
        "mannwhit_p":    ("MW-p",       False, "Mann-Whitney p-value — LOWER is more significant"),
        "mutual_info":   ("MutInfo",    False, "Mutual information with label — higher = stronger link"),
        "within_subj_r": ("Within-r",   True,  "Mean within-subject Pearson r — + = ↑ stress"),
        "perm_imp":      ("PermImp",    False, "Permutation importance on val set (RF) — higher = more important"),
        "shap_mean":     ("SHAP",       True,  "Mean SHAP value (RF, stressed class) — + = pushes toward stressed"),
    }

    # Only show methods that were actually computed
    sample = next(iter(analysis.values()))
    present = [m for m in METHOD_META if m in sample]

    def _rank(method: str) -> Dict[str, int]:
        vals = {n: analysis[n].get(method, float("nan")) for n in FEATURE_NAMES}
        if method == "mannwhit_p":
            ordered = sorted(FEATURE_NAMES, key=lambda n: vals[n] if np.isfinite(vals[n]) else 1.0)
        else:
            ordered = sorted(FEATURE_NAMES, key=lambda n: abs(vals[n]) if np.isfinite(vals[n]) else -1.0, reverse=True)
        return {name: i + 1 for i, name in enumerate(ordered)}

    all_ranks = {m: _rank(m) for m in present}

    # Consensus = average rank (1 = most stress-correlated)
    consensus: Dict[str, float] = {}
    for name in FEATURE_NAMES:
        rs = [all_ranks[m][name] for m in present if np.isfinite(analysis[name].get(m, float("nan")))]
        consensus[name] = float(np.mean(rs)) if rs else float("nan")

    sorted_feats = sorted(FEATURE_NAMES, key=lambda n: consensus.get(n, 99.0))

    W_FEAT, W_COL = 20, 10

    def _header(last_col: str) -> str:
        h = f"{'Feature':<{W_FEAT}}"
        for m in present:
            h += f"  {METHOD_META[m][0]:>{W_COL}}"
        h += f"  {last_col:>{W_COL}}"
        return h

    sep = "─" * len(_header("Consensus"))

    # ── Value table ───────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Raw Values  (sorted by consensus rank, 1 = strongest stress signal)")
    print(f"{'─'*60}")
    print(_header("Consensus"))
    print(sep)
    for name in sorted_feats:
        row = f"{name:<{W_FEAT}}"
        for m in present:
            v = analysis[name].get(m, float("nan"))
            if not np.isfinite(v):
                row += f"  {'n/a':>{W_COL}}"
            elif METHOD_META[m][1]:  # signed
                row += f"  {v:>+{W_COL}.4f}"
            elif m == "mannwhit_p":
                row += f"  {v:{W_COL}.2e}" if v < 1e-4 else f"  {v:>{W_COL}.4f}"
            else:
                row += f"  {v:>{W_COL}.4f}"
        row += f"  {consensus[name]:>{W_COL}.2f}"
        print(row)

    # ── Rank table ────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Ranks  (1 = strongest stress signal per method)")
    print(f"{'─'*60}")
    print(_header(" Avg.Rank"))
    print(sep)
    for name in sorted_feats:
        row = f"{name:<{W_FEAT}}"
        for m in present:
            row += f"  {all_ranks[m][name]:>{W_COL}}"
        row += f"  {consensus[name]:>{W_COL}.1f}"
        print(row)

    # ── Legend ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Legend")
    print(f"{'─'*60}")
    for m in present:
        print(f"  {METHOD_META[m][0]:<12s}  {METHOD_META[m][2]}")
    print()





def get_device(no_gpu: bool = False) -> str:
    if not no_gpu and _TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _infer(X: np.ndarray, w: np.ndarray, b: float, device: str) -> np.ndarray:
    """Run sigmoid(Xw + b) on the given device. Returns a NumPy array of probabilities."""
    if _TORCH_AVAILABLE:
        dev = torch.device(device)
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32, device=dev)
            wt = torch.tensor(w, dtype=torch.float32, device=dev)
            bt = torch.tensor(b, dtype=torch.float32, device=dev)
            probs = torch.sigmoid(Xt @ wt + bt)
        return probs.cpu().numpy()
    return sigmoid(X @ w + b)


def compute_accuracy(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, device: str = "cpu"
) -> float:
    probs = _infer(X, w, b, device)
    preds = (probs >= 0.5).astype(int)
    return float(np.mean(preds == y))


def calibrate_bias(
    Xz_val: np.ndarray,
    y_val: np.ndarray,
    w: np.ndarray,
    b: float,
    device: str = "cpu",
    search_range: float = 5.0,
    n_steps: int = 201,
) -> Tuple[float, float]:
    """
    Grid-search a bias offset that maximises val-set accuracy.

    After class-balanced training the raw bias may not reflect the actual
    class distribution.  This function finds the scalar δ that maximises
    accuracy(sigmoid(Xz·w + b + δ) ≥ 0.5) on the val set.

    Returns (calibrated_bias, val_accuracy_at_calibration).
    If val set is empty, returns (b, nan) unchanged.
    """
    if len(Xz_val) == 0 or len(y_val) == 0:
        return b, float("nan")

    logits = Xz_val @ w  # (N,)   — computed once, offset added per δ
    deltas = np.linspace(-search_range, search_range, n_steps)
    best_delta, best_acc = 0.0, -1.0
    for delta in deltas:
        preds = ((logits + b + delta) >= 0.0).astype(int)
        acc = float(np.mean(preds == y_val))
        if acc > best_acc:
            best_acc, best_delta = acc, float(delta)
    return b + best_delta, best_acc


def per_subject_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    sids: List[str],
    w: np.ndarray,
    b: float,
    device: str = "cpu",
) -> Dict[str, Dict]:
    """Return accuracy and sample counts broken down by subject ID."""
    probs = _infer(X, w, b, device)
    preds = (probs >= 0.5).astype(int)
    results = {}
    for sid in sorted(set(sids)):
        idx = [i for i, s in enumerate(sids) if s == sid]
        ys = y[idx]
        ps = preds[idx]
        n_stressed = int(ys.sum())
        results[sid] = {
            "n_samples": len(ys),
            "n_stressed": n_stressed,
            "n_calm": len(ys) - n_stressed,
            "accuracy": float(np.mean(ps == ys)),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--emo_root", default="./emo",
                    help="Path to the emo/ directory (default: ./emo)")
    ap.add_argument("--out", default="priors.json",
                    help="Write generated priors.json to this path (default: priors.json). "
                         "Pass '' to skip writing.")
    ap.add_argument("--priors", default=None,
                    help="Optional path to an external priors.json to compare against "
                         "(e.g. a previously deployed model). If omitted, the model "
                         "trained in this run is the reference.")
    ap.add_argument("--window_min", type=float, default=WINDOW_DEFAULT_MIN,
                    help="Lookback window in minutes before each ESM response (default: 30)")
    ap.add_argument("--l2", type=float, default=1.0,
                    help="L2 regularisation for logistic regression (default: 1.0)")
    ap.add_argument("--n_train", type=int, default=47)
    ap.add_argument("--n_val",   type=int, default=20)
    ap.add_argument("--n_test",  type=int, default=10)
    ap.add_argument("--no_gpu",  action="store_true")
    ap.add_argument("--epochs",  type=int, default=2000)
    ap.add_argument("--lr",      type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel threads for subject data loading (default: 8)")
    ap.add_argument("--model", choices=["logistic", "rf", "xgb"], default="xgb",
                    help="Additional comparison model: logistic, rf, or xgb (default: xgb). "
                         "The app logistic model (for priors.json) is always trained.")
    ap.add_argument("--n_estimators", type=int, default=600,
                    help="Number of trees for RF/XGBoost (default: 600)")
    ap.add_argument("--max_depth", type=int, default=None,
                    help="Max tree depth for RF/XGBoost (default: None for RF, 4 for XGBoost)")
    ap.add_argument("--engineer", action="store_true", default=True,
                    help="Append engineered features for the comparison model — on by default")
    ap.add_argument("--no_engineer", dest="engineer", action="store_false",
                    help="Disable engineered features for the comparison model")
    args = ap.parse_args()

    # ── Device selection ─────────────────────────────────────────────────────
    device = get_device(args.no_gpu)
    if device == "cuda":
        print(f"[GPU] CUDA available — {torch.cuda.get_device_name(0)}")
    else:
        reason = "disabled via --no_gpu" if args.no_gpu else (
            "PyTorch not installed" if not _TORCH_AVAILABLE else "CUDA not available"
        )
        print(f"[CPU] Running on CPU ({reason})")

    # ── Discover subjects ─────────────────────────────────────────────────────
    emo_root = args.emo_root
    all_subjects = sorted(
        [d for d in os.listdir(emo_root)
         if d.startswith("P") and os.path.isdir(os.path.join(emo_root, d))],
        key=lambda s: int(s[1:]),
    )
    total = len(all_subjects)
    needed = args.n_train + args.n_val + args.n_test
    if total < needed:
        sys.exit(
            f"[error] Only {total} subjects found but {needed} needed "
            f"({args.n_train} train + {args.n_val} val + {args.n_test} test). "
            f"Lower the split counts or point --emo_root at the correct directory."
        )

    train_subj = all_subjects[:args.n_train]
    val_subj   = all_subjects[args.n_train : args.n_train + args.n_val]
    test_subj  = all_subjects[args.n_train + args.n_val : args.n_train + args.n_val + args.n_test]

    print(f"\nSubject split → train={len(train_subj)}, val={len(val_subj)}, test={len(test_subj)}")
    print(f"  Test subjects: {test_subj}")

    # ── Load ESM labels ───────────────────────────────────────────────────────
    esm_path = os.path.join(emo_root, "SubjData", "EsmResponse.csv")
    if not os.path.exists(esm_path):
        sys.exit(f"[error] ESM file not found: {esm_path}")
    esm = pd.read_csv(esm_path)
    esm["label"] = (esm["stress"] > 0).astype(int)

    window_ms = int(args.window_min * 60 * 1000)

    # ── Extract features ──────────────────────────────────────────────────────
    print(f"\n── Training subjects ({len(train_subj)}) ──")
    X_train, y_train, sids_train = collect_split(train_subj, emo_root, esm, window_ms, args.workers)
    print(f"\n── Validation subjects ({len(val_subj)}) ──")
    X_val,   y_val,   sids_val   = collect_split(val_subj,   emo_root, esm, window_ms, args.workers)
    print(f"\n── Test subjects ({len(test_subj)}) ──")
    X_test,  y_test,  sids_test  = collect_split(test_subj,  emo_root, esm, window_ms, args.workers)

    if len(X_train) < 10:
        sys.exit(f"[error] Too few training samples: {len(X_train)}. Check --emo_root.")

    print(f"\nSamples → train={len(X_train)} (stressed={int(y_train.sum())}), "
          f"val={len(X_val)} (stressed={int(y_val.sum())}), "
          f"test={len(X_test)} (stressed={int(y_test.sum())})")

    # ── Impute NaNs (medians from training data only) ─────────────────────────
    X_train, X_val, X_test = impute(X_train, X_val, X_test)

    # ─────────────────────────────────────────────────────────────────────────
    # PART 1: Train the app logistic model (7 base features → priors.json)
    # Always runs regardless of --model, because ScoreEngine.swift uses
    # logistic regression with z-score normalisation.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 1 — App logistic model  (7 base features → priors.json)")
    print("=" * 60)

    # Compute priors from baseline (non-stressed) training samples
    X_base = X_train[y_train == 0]
    app_priors: Dict[str, Dict] = {}
    for j, name in enumerate(FEATURE_NAMES):
        vals = X_base[:, j]
        vals = vals[np.isfinite(vals)]
        mu = float(np.mean(vals)) if len(vals) > 0 else 0.0
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0
        app_priors[name] = {"mean": round(mu, 6), "std": round(max(sd, 1e-6), 6)}

    # Standardise using app priors (mirrors ScoreEngine.swift)
    Xz_train_app = standardise(X_train, app_priors)
    Xz_val_app   = standardise(X_val,   app_priors)
    Xz_test_app  = standardise(X_test,  app_priors)

    use_gpu = device == "cuda"
    if _TORCH_AVAILABLE and use_gpu:
        print(f"Fitting logistic regression (PyTorch, epochs={args.epochs}, lr={args.lr}) …")
    else:
        print("Fitting logistic regression (NumPy Newton-Raphson) …")

    w_app, b_app = fit_logistic(
        Xz_train_app, y_train, l2=args.l2, use_gpu=use_gpu,
        epochs=args.epochs, lr=args.lr, device=device,
    )

    # Calibrate decision threshold on val set: balanced training may push the
    # raw bias away from the actual class distribution.  We grid-search a bias
    # offset δ that maximises val-set accuracy, then store b_app + δ in
    # priors.json.  Training weights are unchanged.
    if len(X_val) > 0:
        b_cal, acc_cal = calibrate_bias(Xz_val_app, y_val, w_app, b_app, device)
        print(f"  [calibrate] bias {b_app:+.4f} → {b_cal:+.4f}  "
              f"(val acc before: {compute_accuracy(Xz_val_app, y_val, w_app, b_app, device):.1%} "
              f"→ after: {acc_cal:.1%})")
    else:
        b_cal = b_app

    acc_train_app = compute_accuracy(Xz_train_app, y_train, w_app, b_cal, device)
    acc_val_app   = compute_accuracy(Xz_val_app,   y_val,   w_app, b_cal, device) if len(X_val)  > 0 else float("nan")
    acc_test_app  = compute_accuracy(Xz_test_app,  y_test,  w_app, b_cal, device) if len(X_test) > 0 else float("nan")

    print(f"\nLogistic (app model) accuracy:")
    print(f"  Train  ({len(train_subj):2d} subj) : {acc_train_app:.1%}")
    print(f"  Val    ({len(val_subj):2d} subj) : {acc_val_app:.1%}")
    print(f"  Test   ({len(test_subj):2d} subj) : {acc_test_app:.1%}")

    print("\n── Learned feature weights (0-100 scale) ──")
    w_scaled = scale_weights_0_100(w_app)
    for name, ws, wr in zip(FEATURE_NAMES, w_scaled, w_app):
        direction = "↑ stress" if wr > 0 else "↓ stress"
        print(f"  {name:<20s}  {ws:+7.2f}  (raw: {wr:+.4f})  [{direction}]")
    print(f"  {'bias':<20s}  {b_cal:+.4f}  (calibrated)")

    # ── Write priors.json ─────────────────────────────────────────────────────
    if args.out:
        out_json = {
            "meta": {
                "source": "K-EmoPhone (EMO)",
                "split": {
                    "train": len(train_subj),
                    "val":   len(val_subj),
                    "test":  len(test_subj),
                },
                "n_samples": {
                    "train": len(X_train),
                    "val":   len(X_val),
                    "test":  len(X_test),
                },
                "accuracy": {
                    "train": round(acc_train_app, 4),
                    "val":   round(float(acc_val_app),  4),
                    "test":  round(float(acc_test_app), 4),
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
                    "bias is threshold-calibrated on the val set to maximise accuracy. "
                    "Score = sigmoid(b + sum(w_i * z_i)) * 100 where "
                    "z_i = clip((x_i - mu_i)/sigma_i, -3, 3)."
                ),
            },
            "priors": app_priors,
            "weights": {
                name: round(float(w_app[j]), 6)
                for j, name in enumerate(FEATURE_NAMES)
            },
            "weights_display": {
                name: round(float(w_scaled[j]), 4)
                for j, name in enumerate(FEATURE_NAMES)
            },
            "bias": round(float(b_cal), 6),
        }
        with open(args.out, "w") as f:
            json.dump(out_json, f, indent=2)
        print(f"\n✓ Wrote {args.out}")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 2: Optional comparison model (RF or XGBoost, with engineered features)
    # ─────────────────────────────────────────────────────────────────────────
    acc_train_cmp = acc_val_cmp = acc_test_cmp = float("nan")

    if args.model != "logistic":
        # Build engineered feature matrices for the comparison model
        if args.engineer:
            X_train_m = engineer_features(X_train)
            X_val_m   = engineer_features(X_val)
            X_test_m  = engineer_features(X_test)
            X_train_m, X_val_m, X_test_m = impute(X_train_m, X_val_m, X_test_m)
            feat_names = FEATURE_NAMES + ENGINEERED_NAMES
        else:
            X_train_m, X_val_m, X_test_m = X_train, X_val, X_test
            feat_names = FEATURE_NAMES

        print("\n" + "=" * 60)
        print(f"PART 2 — {args.model.upper()} comparison model  "
              f"({len(feat_names)} features, {'engineered' if args.engineer else 'base only'})")
        print("=" * 60)

        if args.model == "rf":
            print(f"Fitting Random Forest (n_estimators={args.n_estimators}, "
                  f"max_depth={args.max_depth}, n_jobs=-1) …")
            clf = fit_random_forest(
                X_train_m, y_train,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
            )
            acc_train_cmp = rf_accuracy(clf, X_train_m, y_train)
            acc_val_cmp   = rf_accuracy(clf, X_val_m,   y_val)   if len(X_val)  > 0 else float("nan")
            acc_test_cmp  = rf_accuracy(clf, X_test_m,  y_test)  if len(X_test) > 0 else float("nan")

            print(f"\nRandom Forest accuracy:")
            print(f"  Train  ({len(train_subj):2d} subj) : {acc_train_cmp:.1%}")
            print(f"  Val    ({len(val_subj):2d} subj) : {acc_val_cmp:.1%}")
            print(f"  Test   ({len(test_subj):2d} subj) : {acc_test_cmp:.1%}")

            importances = clf.feature_importances_
            print("\n── Feature importances (RF) ──")
            for name, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
                bar = "█" * int(imp * 40)
                print(f"  {name:<22s}  {imp:.4f}  {bar}")

            print(f"\n── Classification report (val set) ──")
            print(classification_report(y_val, clf.predict(X_val_m),
                                        target_names=["calm", "stressed"], digits=3))

        elif args.model == "xgb":
            max_d = args.max_depth if args.max_depth is not None else 4
            print(f"Fitting XGBoost (n_estimators={args.n_estimators}, max_depth={max_d}, "
                  f"lr={args.lr}, device={device}) …")
            clf = fit_xgboost(
                X_train_m, y_train,
                X_val_m,   y_val,
                n_estimators=args.n_estimators,
                max_depth=max_d,
                lr=args.lr,
                device=device,
            )
            acc_train_cmp = rf_accuracy(clf, X_train_m, y_train)
            acc_val_cmp   = rf_accuracy(clf, X_val_m,   y_val)   if len(X_val)  > 0 else float("nan")
            acc_test_cmp  = rf_accuracy(clf, X_test_m,  y_test)  if len(X_test) > 0 else float("nan")

            print(f"\nXGBoost accuracy:")
            print(f"  Train  ({len(train_subj):2d} subj) : {acc_train_cmp:.1%}")
            print(f"  Val    ({len(val_subj):2d} subj) : {acc_val_cmp:.1%}")
            print(f"  Test   ({len(test_subj):2d} subj) : {acc_test_cmp:.1%}")

            importances = clf.feature_importances_
            print("\n── Feature importances (XGBoost) ──")
            for name, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
                bar = "█" * int(imp * 40)
                print(f"  {name:<22s}  {imp:.4f}  {bar}")

            print(f"\n── Classification report (val set) ──")
            print(classification_report(y_val, clf.predict(X_val_m),
                                        target_names=["calm", "stressed"], digits=3))

    # ─────────────────────────────────────────────────────────────────────────
    # PART 3: Theoretical app accuracy — per-subject test breakdown
    # Uses the logistic model trained in PART 1 (the model written to priors.json)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 3 — Theoretical app accuracy (test subjects, app logistic model)")
    print("=" * 60)
    print("z_i = clip((x_i − μ_i) / σ_i, −3, +3)  [app priors]")
    print("score = sigmoid(b + Σ w_i·z_i) × 100    [app weights]")
    print("prediction = stressed if score < 50 else calm\n")

    per_subj = per_subject_accuracy(Xz_test_app, y_test, sids_test, w_app, b_cal, device)

    print(f"{'Subject':<10} {'Samples':>8} {'Stressed':>9} {'Calm':>6} {'Accuracy':>10}")
    print("-" * 46)
    for sid, stats in per_subj.items():
        print(f"  {sid:<8} {stats['n_samples']:>8} {stats['n_stressed']:>9} "
              f"{stats['n_calm']:>6} {stats['accuracy']:>9.1%}")
    print("-" * 46)
    print(f"  {'OVERALL':<8} {len(y_test):>8} {int(y_test.sum()):>9} "
          f"{len(y_test)-int(y_test.sum()):>6} {acc_test_app:>9.1%}")

    majority_acc = max(float(y_test.mean()), 1.0 - float(y_test.mean()))
    lift = acc_test_app - majority_acc
    print(f"\n  Majority-class baseline  : {majority_acc:.1%}")
    print(f"  App logistic model       : {acc_test_app:.1%}")
    print(f"  Lift over baseline       : {lift:+.1%}")

    # ── Optional: compare against an externally supplied priors.json ──────────
    if args.priors:
        if not os.path.exists(args.priors):
            print(f"\n[warn] --priors file not found: {args.priors}. Skipping comparison.")
        else:
            with open(args.priors) as f:
                ext_model = json.load(f)
            ext_priors  = ext_model["priors"]
            ext_weights = np.array([ext_model["weights"][n] for n in FEATURE_NAMES], dtype=float)
            ext_bias    = float(ext_model["bias"])
            Xz_test_ext = standardise(X_test, ext_priors)
            ext_acc = compute_accuracy(Xz_test_ext, y_test, ext_weights, ext_bias, device)
            ext_lift = ext_acc - majority_acc
            print(f"\n── External model ({args.priors}) ──")
            print(f"  Accuracy on test set     : {ext_acc:.1%}")
            print(f"  Lift over baseline       : {ext_lift:+.1%}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Training subjects             : {len(train_subj)}")
    print(f"  Validation subjects           : {len(val_subj)}")
    print(f"  Test subjects                 : {len(test_subj)}")
    print()
    print(f"  App logistic — train acc      : {acc_train_app:.1%}")
    print(f"  App logistic — val acc        : {acc_val_app:.1%}")
    print(f"  App logistic — test acc       : {acc_test_app:.1%}  ← THEORETICAL APP ACCURACY")
    if args.model != "logistic" and not (
        acc_train_cmp != acc_train_cmp  # nan check
    ):
        print(f"\n  {args.model.upper():<10} comparison — train : {acc_train_cmp:.1%}")
        print(f"  {args.model.upper():<10} comparison — val   : {acc_val_cmp:.1%}")
        print(f"  {args.model.upper():<10} comparison — test  : {acc_test_cmp:.1%}")
    print()
    print(f"  Majority-class baseline       : {majority_acc:.1%}")
    print(f"  Lift over baseline            : {lift:+.1%}")
    if args.out:
        print(f"\n  priors.json written to        : {args.out}")
    print("=" * 60)

    # ─────────────────────────────────────────────────────────────────────────
    # PART 4 — Feature importance analysis (all 7 methods)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 4 — Feature Importance Analysis")
    print("=" * 60)
    print("Comparing 7 methods to identify which features most reliably")
    print("correlate with stress — model-free and model-based.\n")

    analysis = compute_feature_analysis(
        X_train, y_train, X_val, y_val, sids_train, w_app,
    )
    print_feature_table(analysis)


if __name__ == "__main__":
    main()
