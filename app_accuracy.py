#!/usr/bin/env python3
"""
app_accuracy.py
===============
Full training → validation → testing pipeline that also generates priors.json.

Pipeline
--------
    PART 1 (train)   47 subjects  →  fit logistic/linear regression on base features
                                   compute priors (μ/σ) from baseline samples
                                   write priors.json  (ScoreEngine.swift compatible)
  PART 2 (compare) optional     →  train RF or XGBoost on engineered features
                                   for side-by-side accuracy comparison
    PART 3 (test)    10 subjects  →  score with app regression model, per-subject report
                                   % calm in test set (66.2 % of people are calm)

Scoring (for app model):
    z_i   = clip((x_i - μ_i) / σ_i, -3, +3)
    logistic: score = sigmoid(b + Σ w_i·z_i) × 100
    linear:   score = clip(b + Σ w_i·z_i, 0, 1) × 100
    label = stressed if score >= 50 else calm

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
    from sklearn.svm import SVC
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    print("[warn] scikit-learn not found — Random Forest unavailable.", file=sys.stderr)

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
    "HR_mean_30",   # mean HR over 30-min window
    "HR_std_30",    # std dev of HR over 30-min window
    "HR_slope_30",  # linear slope of HR over 30-min window (bpm/min)
    "HRV_30",       # RMSSD over 30-min window
    "HR_mean_5",    # mean HR over 5-min window (closer to ESM response)
    "HRV_5",        # RMSSD over 5-min window
]

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


def _window_std(ts, vals, t0s, t1s, min_count=3):
    """Std dev of vals in [t0, t1]. Returns nan where count < min_count."""
    out = np.full(len(t0s), np.nan)
    for i in range(len(t0s)):
        lo = np.searchsorted(ts, t0s[i])
        hi = np.searchsorted(ts, t1s[i], side="right")
        if hi - lo >= min_count:
            out[i] = vals[lo:hi].std()
    return out


def _window_slope(ts, vals, t0s, t1s, min_count=5):
    """Linear slope (units/ms) of vals vs time in [t0, t1].
    Returns nan where count < min_count."""
    out = np.full(len(t0s), np.nan)
    for i in range(len(t0s)):
        lo = np.searchsorted(ts, t0s[i])
        hi = np.searchsorted(ts, t1s[i], side="right")
        if hi - lo >= min_count:
            t = ts[lo:hi].astype(float)
            v = vals[lo:hi]
            t_c = t - t.mean()
            denom = float(np.dot(t_c, t_c))
            if denom > 0:
                out[i] = float(np.dot(t_c, v) / denom) * 60000  # convert to per-minute
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
    window_ms: int,   # kept for API compat, not used (we use fixed windows below)
) -> np.ndarray:
    """Return (N, 6) feature matrix. Two windows: 30-min and 5-min."""
    W30 = 30 * 60 * 1000
    W5  =  5 * 60 * 1000
    n   = len(timestamps_ms)
    X   = np.full((n, len(FEATURE_NAMES)), np.nan)

    if "HR.csv_ts" in data:
        hr_ts  = data["HR.csv_ts"]
        hr_bpm = data["HR.csv_bpm"]
        t0_30 = timestamps_ms - W30
        t0_5  = timestamps_ms - W5

        X[:, 0] = _window_mean(hr_ts, hr_bpm, t0_30, timestamps_ms, min_count=3)
        X[:, 1] = _window_std(hr_ts, hr_bpm, t0_30, timestamps_ms, min_count=3)
        X[:, 2] = _window_slope(hr_ts, hr_bpm, t0_30, timestamps_ms, min_count=5)
        X[:, 4] = _window_mean(hr_ts, hr_bpm, t0_5,  timestamps_ms, min_count=2)

    if "RRI.csv_ts" in data:
        rri_ts  = data["RRI.csv_ts"]
        rri_val = data["RRI.csv_interval"]
        t0_30 = timestamps_ms - W30
        t0_5  = timestamps_ms - W5

        X[:, 3] = _window_rmssd(rri_ts, rri_val, t0_30, timestamps_ms)
        X[:, 5] = _window_rmssd(rri_ts, rri_val, t0_5,  timestamps_ms)

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
    """Standardise X using a priors dict: z = clip((x - mu) / sigma, -3, +3)."""
    Xz = np.zeros_like(X)
    for j, name in enumerate(FEATURE_NAMES):
        mu = priors[name]["mean"]
        sd = priors[name]["std"]
        Xz[:, j] = np.clip((X[:, j] - mu) / (sd + 1e-9), -Z_CLAMP, Z_CLAMP)
    return Xz


def compute_personal_priors(
    X: np.ndarray,
    y: np.ndarray,
    sids: List[str],
    min_calm: int = 3,
) -> Dict[str, Dict]:
    """Compute per-subject priors from each subject's own calm (label=0) samples.
    Subjects with fewer than min_calm calm samples are skipped (caller falls
    back to population priors for those subjects)."""
    personal: Dict[str, Dict] = {}
    for sid in sorted(set(sids)):
        idx = [i for i, s in enumerate(sids) if s == sid and y[i] == 0]
        if len(idx) < min_calm:
            continue
        subj_priors: Dict = {}
        for j, name in enumerate(FEATURE_NAMES):
            vals = X[idx, j]
            vals = vals[np.isfinite(vals)]
            mu = float(np.mean(vals)) if len(vals) > 0 else 0.0
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0
            subj_priors[name] = {"mean": round(mu, 6), "std": round(max(sd, 1e-6), 6)}
        personal[sid] = subj_priors
    return personal


def standardise_personal(
    X: np.ndarray,
    sids: List[str],
    personal_priors: Dict[str, Dict],
    fallback_priors: Dict,
) -> np.ndarray:
    """Per-row standardisation: each row uses that subject's personal priors,
    falling back to population priors for subjects without enough calm data."""
    Xz = np.zeros_like(X)
    for i, sid in enumerate(sids):
        priors = personal_priors.get(sid, fallback_priors)
        for j, name in enumerate(FEATURE_NAMES):
            mu = priors[name]["mean"]
            sd = priors[name]["std"]
            Xz[i, j] = np.clip((X[i, j] - mu) / (sd + 1e-9), -Z_CLAMP, Z_CLAMP)
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

def fit_linear_ridge(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Simple ridge linear regression: y_hat = Xw + b."""
    n, d = X.shape
    Xa = np.hstack([X, np.ones((n, 1), dtype=float)])
    reg = np.eye(d + 1, dtype=float)
    reg[-1, -1] = 0.0  # do not regularise bias
    A = Xa.T @ Xa + l2 * reg
    rhs = Xa.T @ y.astype(float)
    beta = np.linalg.solve(A, rhs)
    w = beta[:-1]
    b = float(beta[-1])
    return w, b


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


def _predict_score(
    X: np.ndarray,
    w: np.ndarray,
    b: float,
) -> np.ndarray:
    """Return raw linear scores clipped to [-3, 3].
    Positive = stressed, negative = calm. Threshold at 0."""
    return np.clip(X @ w + b, -Z_CLAMP, Z_CLAMP)


def compute_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
) -> float:
    scores = _predict_score(X, w, b)
    preds = (scores > 0.0).astype(int)
    return float(np.mean(preds == y))


def calibrate_bias(
    Xz_val: np.ndarray,
    y_val: np.ndarray,
    w: np.ndarray,
    b: float,
    search_range: float = 5.0,
    n_steps: int = 201,
) -> Tuple[float, float]:
    """Grid-search a bias offset that maximises Matthews Correlation Coefficient.
    MCC = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    MCC = 0 when all samples are predicted as one class, so it cannot be gamed
    by a degenerate threshold the way accuracy or F-beta can.
    Returns (calibrated_bias, MCC_at_calibration)."""
    if len(Xz_val) == 0 or len(y_val) == 0:
        return b, float("nan")

    linear = Xz_val @ w
    deltas = np.linspace(-search_range, search_range, n_steps)
    best_delta, best_mcc = 0.0, -2.0
    for delta in deltas:
        preds = (np.clip(linear + b + delta, -Z_CLAMP, Z_CLAMP) > 0.0).astype(int)
        tp = float(np.sum((y_val == 1) & (preds == 1)))
        tn = float(np.sum((y_val == 0) & (preds == 0)))
        fp = float(np.sum((y_val == 0) & (preds == 1)))
        fn = float(np.sum((y_val == 1) & (preds == 0)))
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / (denom + 1e-9)
        if mcc > best_mcc:
            best_mcc, best_delta = mcc, float(delta)
    return b + best_delta, best_mcc


def per_subject_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    sids: List[str],
    w: np.ndarray,
    b: float,
) -> Dict[str, Dict]:
    """Return accuracy and sample counts broken down by subject ID."""
    scores = _predict_score(X, w, b)
    preds = (scores > 0.0).astype(int)

    def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        return {
            "true_stress": tp,
            "true_calm": tn,
            "false_stress": fp,
            "false_calm": fn,
        }

    results = {}
    for sid in sorted(set(sids)):
        idx = [i for i, s in enumerate(sids) if s == sid]
        ys = y[idx]
        ps = preds[idx]
        n_stressed = int(ys.sum())
        c = _confusion_counts(ys, ps)
        results[sid] = {
            "n_samples": len(ys),
            "n_stressed": n_stressed,
            "n_calm": len(ys) - n_stressed,
            "accuracy": float(np.mean(ps == ys)),
            **c,
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
                    help="L2 regularisation strength for logistic regression (default: 1.0)")
    ap.add_argument("--n_train", type=int, default=47)
    ap.add_argument("--n_val",   type=int, default=20)
    ap.add_argument("--n_test",  type=int, default=10)
    ap.add_argument("--no_gpu",  action="store_true")
    ap.add_argument("--epochs",  type=int, default=2000)
    ap.add_argument("--lr",      type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel threads for subject data loading (default: 8)")
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

    # ── Population priors (calm training samples) — saved to priors.json ──────
    # Used as cold-start fallback in the app before personal baseline accumulates.
    X_base = X_train[y_train == 0]
    app_priors: Dict[str, Dict] = {}
    for j, name in enumerate(FEATURE_NAMES):
        vals = X_base[:, j]
        vals = vals[np.isfinite(vals)]
        mu = float(np.mean(vals)) if len(vals) > 0 else 0.0
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0
        app_priors[name] = {"mean": round(mu, 6), "std": round(max(sd, 1e-6), 6)}

    # ── Per-person priors — used for all model training and evaluation ─────────
    # Each subject's calm windows define their own baseline mu/sigma.
    # Subjects with < 3 calm samples fall back to population priors.
    pp_train = compute_personal_priors(X_train, y_train, sids_train)
    pp_val   = compute_personal_priors(X_val,   y_val,   sids_val)
    pp_test  = compute_personal_priors(X_test,  y_test,  sids_test)

    n_personal = sum(1 for sids in [sids_train, sids_val, sids_test]
                     for s in set(sids)
                     if s in {**pp_train, **pp_val, **pp_test})
    n_total_subj = len(set(sids_train) | set(sids_val) | set(sids_test))
    print(f"\n  Personal priors: {n_personal}/{n_total_subj} subjects have ≥3 calm samples "
          f"(rest fall back to population)")

    Xz_train = standardise_personal(X_train, sids_train, pp_train, app_priors)
    Xz_val   = standardise_personal(X_val,   sids_val,   pp_val,   app_priors)
    Xz_test  = standardise_personal(X_test,  sids_test,  pp_test,  app_priors)

    # Standardise using app priors (mirrors ScoreEngine.swift)
    # ─────────────────────────────────────────────────────────────────────────
    # PART 1: Train the logistic regression model (personal normalization)
    # priors.json gets population priors (cold-start). Training uses personal.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PART 1 — App logistic model  ({len(FEATURE_NAMES)} features, per-person z-scores)")
    print("=" * 60)

    use_gpu = device == "cuda"
    if _TORCH_AVAILABLE and use_gpu:
        print(f"Fitting logistic regression (PyTorch, epochs={args.epochs}, lr={args.lr}) …")
    else:
        print("Fitting logistic regression (NumPy Newton-Raphson) …")
    w_app, b_app = fit_logistic(
        Xz_train, y_train, l2=args.l2, use_gpu=use_gpu,
        epochs=args.epochs, lr=args.lr, device=device,
    )
    b_cal = b_app

    acc_train_app = compute_accuracy(Xz_train, y_train, w_app, b_cal)
    acc_val_app   = compute_accuracy(Xz_val,   y_val,   w_app, b_cal) if len(X_val)  > 0 else float("nan")
    acc_test_app  = compute_accuracy(Xz_test,  y_test,  w_app, b_cal) if len(X_test) > 0 else float("nan")

    scores_train = np.clip(Xz_train @ w_app + b_cal, -Z_CLAMP, Z_CLAMP)
    print(f"\n── Score diagnostics (train, personal z-scores) ──")
    print(f"  Weights : {dict(zip(FEATURE_NAMES, [f'{v:+.4f}' for v in w_app]))}")
    print(f"  Bias    : {b_cal:+.4f}")
    print(f"  Score std : {scores_train.std():.4f}  (near-zero = no signal)")
    print(f"  Predicted stressed % : {float(np.mean(scores_train > 0)):.1%}  "
          f"(actual: {y_train.mean():.1%})")

    print(f"\nLogistic accuracy (per-person z-scores):")
    print(f"  Train  ({len(train_subj):2d} subj) : {acc_train_app:.1%}")
    print(f"  Val    ({len(val_subj):2d} subj) : {acc_val_app:.1%}")
    print(f"  Test   ({len(test_subj):2d} subj) : {acc_test_app:.1%}")

    print("\n── Learned feature weights (0-100 scale) ──")
    w_scaled = scale_weights_0_100(w_app)
    for name, ws, wr in zip(FEATURE_NAMES, w_scaled, w_app):
        direction = "↑ stress" if wr > 0 else "↓ stress"
        print(f"  {name:<20s}  {ws:+7.2f}  (raw: {wr:+.4f})  [{direction}]")
    print(f"  {'bias':<20s}  {b_cal:+.4f}")

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
                    "Weights are raw logistic regression coefficients. "
                    "weights_display is linearly scaled to [-100, +100] for readability. "
                    "bias is threshold-calibrated on the val set. "
                    "Score = clip(b + sum(w_i * z_i), -3, +3)  where z_i = clip((x_i - mu_i)/sigma_i, -3, 3). "
                    "Positive score = stressed, negative = calm, threshold at 0."
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
    # PART 2: Comparison models — SVM, Random Forest, XGBoost, MLP
    # Trained on train+val combined (no held-out val needed for calibration).
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 2 — Comparison models  (SVM / RF / XGBoost / MLP)")
    print("=" * 60)

    # Combine train + val for comparison models
    clf_rf2 = None   # kept in scope so PART 3 / SUMMARY can reference it
    Xz_trainval = np.vstack([Xz_train, Xz_val]) if len(Xz_val) > 0 else Xz_train
    y_trainval  = np.concatenate([y_train, y_val])       if len(y_val)       > 0 else y_train

    def _confusion_report(name: str, y_true: np.ndarray, y_pred: np.ndarray,
                          acc_train: float, acc_test: float) -> None:
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        print(f"\n  {name}")
        print(f"    Train acc : {acc_train:.1%}   Test acc : {acc_test:.1%}")
        print(f"    TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    # ── SVM ──────────────────────────────────────────────────────────────────
    try:
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")
        print("\n  [SVM] Training …")
        clf_svm = SVC(kernel="rbf", class_weight="balanced", C=1.0, gamma="scale", random_state=42)
        clf_svm.fit(Xz_trainval, y_trainval)
        svm_train_acc = float(np.mean(clf_svm.predict(Xz_trainval) == y_trainval))
        svm_preds     = clf_svm.predict(Xz_test)
        svm_test_acc  = float(np.mean(svm_preds == y_test))
        _confusion_report("SVM (RBF, C=1)", y_test, svm_preds, svm_train_acc, svm_test_acc)
    except ImportError as e:
        print(f"\n  [warn] SVM skipped: {e}")

    # ── Random Forest ─────────────────────────────────────────────────────────
    try:
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")
        print("\n  [RF] Training …")
        clf_rf2 = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )
        clf_rf2.fit(Xz_trainval, y_trainval)
        rf_train_acc = float(np.mean(clf_rf2.predict(Xz_trainval) == y_trainval))
        rf_preds     = clf_rf2.predict(Xz_test)
        rf_test_acc  = float(np.mean(rf_preds == y_test))
        _confusion_report("Random Forest (300 trees)", y_test, rf_preds, rf_train_acc, rf_test_acc)
    except ImportError as e:
        print(f"\n  [warn] Random Forest skipped: {e}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    try:
        from xgboost import XGBClassifier
        print("\n  [XGB] Training …")
        n_pos_tv = int(y_trainval.sum())
        n_neg_tv = len(y_trainval) - n_pos_tv
        scale_pos = n_neg_tv / max(n_pos_tv, 1)
        clf_xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pos,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        clf_xgb.fit(Xz_trainval, y_trainval)
        xgb_train_acc = float(np.mean(clf_xgb.predict(Xz_trainval) == y_trainval))
        xgb_preds     = clf_xgb.predict(Xz_test)
        xgb_test_acc  = float(np.mean(xgb_preds == y_test))
        _confusion_report("XGBoost (300 trees, depth=4)", y_test, xgb_preds, xgb_train_acc, xgb_test_acc)
    except ImportError:
        print("\n  [warn] XGBoost skipped — not installed.  pip install xgboost")

    # ── MLP (PyTorch) ─────────────────────────────────────────────────────────
    try:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")
        print("\n  [MLP] Training (PyTorch, 500 epochs, Adam lr=0.01) …")
        _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Xt_tv  = torch.tensor(Xz_trainval, dtype=torch.float32, device=_dev)
        yt_tv  = torch.tensor(y_trainval,  dtype=torch.float32, device=_dev)
        Xt_te  = torch.tensor(Xz_test, dtype=torch.float32, device=_dev)

        n_in = Xz_trainval.shape[1]
        mlp = nn.Sequential(
            nn.Linear(n_in, 32), nn.ReLU(),
            nn.Linear(32, 16),   nn.ReLU(),
            nn.Linear(16, 1),
        ).to(_dev)

        n_pos_tv = int((y_trainval == 1).sum())
        n_neg_tv = len(y_trainval) - n_pos_tv
        pos_w_tv = torch.tensor([n_neg_tv / max(n_pos_tv, 1)], dtype=torch.float32, device=_dev)
        bce_mlp  = nn.BCEWithLogitsLoss(pos_weight=pos_w_tv)
        opt_mlp  = torch.optim.Adam(mlp.parameters(), lr=0.01)

        mlp.train()
        for _ep in range(500):
            opt_mlp.zero_grad()
            loss_mlp = bce_mlp(mlp(Xt_tv).squeeze(1), yt_tv)
            loss_mlp.backward()
            opt_mlp.step()

        mlp.eval()
        with torch.no_grad():
            mlp_train_preds = (torch.sigmoid(mlp(Xt_tv).squeeze(1)) >= 0.5).cpu().numpy().astype(int)
            mlp_test_preds  = (torch.sigmoid(mlp(Xt_te).squeeze(1)) >= 0.5).cpu().numpy().astype(int)
        mlp_train_acc = float(np.mean(mlp_train_preds == y_trainval))
        mlp_test_acc  = float(np.mean(mlp_test_preds  == y_test))
        _confusion_report("MLP (32→16→1, ReLU, BCE, 500 ep)", y_test, mlp_test_preds,
                          mlp_train_acc, mlp_test_acc)
    except ImportError as e:
        print(f"\n  [warn] MLP skipped: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 3: Per-subject score breakdown on test set
    # Score = clip(b + Σ w_i·z_i, -3, +3). Positive = stressed, threshold at 0.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 3 — Per-subject test breakdown  (Random Forest)")
    print("=" * 60)

    if clf_rf2 is not None:
        rf_preds_all = clf_rf2.predict(Xz_test)
        primary_preds = rf_preds_all
        primary_acc   = float(np.mean(primary_preds == y_test))
        model_label   = "RF"
    else:
        scores_test   = _predict_score(Xz_test, w_app, b_cal)
        primary_preds = (scores_test > 0.0).astype(int)
        primary_acc   = acc_test_app
        model_label   = "Logistic"

    # Per-subject breakdown
    sids_arr = np.array(sids_test)
    print(f"{'Subject':<10} {'Samples':>8} {'Stressed':>9} {'Calm':>6} {'Accuracy':>10}")
    print("-" * 46)
    for sid in sorted(set(sids_test)):
        mask = sids_arr == sid
        y_s, p_s = y_test[mask], primary_preds[mask]
        print(f"  {sid:<8} {mask.sum():>8} {int(y_s.sum()):>9} "
              f"{int((y_s==0).sum()):>6} {float(np.mean(p_s==y_s)):>9.1%}")
    print("-" * 46)
    print(f"  {'OVERALL':<8} {len(y_test):>8} {int(y_test.sum()):>9} "
          f"{len(y_test)-int(y_test.sum()):>6} {primary_acc:>9.1%}")

    tp = int(np.sum((y_test == 1) & (primary_preds == 1)))
    tn = int(np.sum((y_test == 0) & (primary_preds == 0)))
    fp = int(np.sum((y_test == 0) & (primary_preds == 1)))
    fn = int(np.sum((y_test == 1) & (primary_preds == 0)))
    print(f"\nConfusion counts ({model_label}):")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    majority_acc = max(float(y_test.mean()), 1.0 - float(y_test.mean()))
    lift = primary_acc - majority_acc
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
            ext_acc = compute_accuracy(Xz_test_ext, y_test, ext_weights, ext_bias)
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
    print(f"  Primary model ({model_label}) — test acc   : {primary_acc:.1%}")
    print(f"  Logistic (app model) — test acc      : {acc_test_app:.1%}")
    print()
    print(f"  Majority-class baseline              : {majority_acc:.1%}")
    print(f"  Lift over baseline ({model_label})         : {lift:+.1%}")
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

    # ─────────────────────────────────────────────────────────────────────────
    # PART 5 — Online Huber learning simulation
    # Simulates what happens in the app: start from population weights, receive
    # one ESM-labeled sample at a time, do a Huber SGD step, track accuracy.
    # Target encoding: stressed → +1, calm → -1  (matches [-3,+3] score range)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 5 — Online Huber Learning Simulation")
    print("=" * 60)
    print("Each test subject starts from population weights.")
    print("Accuracy is measured BEFORE each update (predict-then-learn).\n")

    _simulate_online_learning(
        X_test, y_test, sids_test,
        w_init=w_app, b_init=b_app,
        pp_test=pp_test, app_priors=app_priors,
        lrs=(0.01, 0.05, 0.1),
        best_lr=0.05,
        n_warmup=30,
    )


def _simulate_online_learning(
    X_test: np.ndarray,
    y_test: np.ndarray,
    sids_test: List[str],
    w_init: np.ndarray,
    b_init: float,
    pp_test: Dict,
    app_priors: Dict,
    lrs: Tuple = (0.01, 0.05, 0.1),
    best_lr: float = 0.05,
    huber_delta: float = 1.0,
    weight_decay: float = 0.01,
    n_warmup: int = 30,
    report_at: Tuple = (0, 1, 3, 5, 10, 20, 30, 50),
) -> None:
    """Simulate online Huber SGD for each lr; print accuracy-vs-N table."""
    subjects = [s for s in sorted(set(sids_test))
                if sids_test.count(s) >= 5]

    if not subjects:
        print("  [warn] No test subjects with ≥5 samples — skipping simulation.")
        return

    def _run_lr(lr: float) -> Dict[str, List[int]]:
        """Returns {sid: [correct_before_update_0, correct_before_update_1, ...]}"""
        curves: Dict[str, List[int]] = {}
        for sid in subjects:
            idx = [i for i, s in enumerate(sids_test) if s == sid]
            X_sub = X_test[idx]
            y_sub = y_test[idx]

            # Per-person z-scores (mirrors app runtime behaviour)
            priors = pp_test.get(sid, app_priors)
            Xz = np.zeros_like(X_sub, dtype=float)
            for j, name in enumerate(FEATURE_NAMES):
                mu, sd = priors[name]["mean"], priors[name]["std"]
                Xz[:, j] = np.clip((X_sub[:, j] - mu) / (sd + 1e-9), -Z_CLAMP, Z_CLAMP)

            w = w_init.copy().astype(float)
            b = float(b_init)
            hits: List[int] = []

            for i in range(len(Xz)):
                z = Xz[i]
                target = 1.0 if y_sub[i] == 1 else -1.0

                # Predict BEFORE update
                score = float(np.dot(w, z) + b)
                hits.append(int((score > 0) == (y_sub[i] == 1)))

                # Huber gradient  (quadratic core, linear tails)
                residual = score - target
                grad = residual if abs(residual) <= huber_delta else huber_delta * np.sign(residual)

                # SGD step with L2 weight decay (keeps weights near population)
                w = w * (1.0 - weight_decay * lr) - lr * grad * z
                b = b - lr * grad

            curves[sid] = hits
        return curves

    # ── Run all learning rates ────────────────────────────────────────────────
    results: Dict[float, Dict] = {lr: _run_lr(lr) for lr in lrs}

    # ── Aggregate: accuracy at each checkpoint N ──────────────────────────────
    # For a subject with fewer than N samples, use their full curve.
    def _acc_at(curves: Dict[str, List[int]], n: int) -> float:
        hits, total = 0, 0
        for hits_list in curves.values():
            window = hits_list[:n] if n > 0 else []
            hits  += sum(window)
            total += len(window)
        return hits / total if total > 0 else float("nan")

    def _acc_after(curves: Dict[str, List[int]], n: int) -> float:
        """Accuracy on samples index >= n (performance after n warm-up steps)."""
        hits, total = 0, 0
        for hits_list in curves.values():
            window = hits_list[n:]
            hits  += sum(window)
            total += len(window)
        return hits / total if total > 0 else float("nan")

    # Population baseline: accuracy with w_init, b_init, personal z-scores, no updates
    pop_curves = _run_lr(0.0)   # lr=0 → no weight change
    pop_acc = _acc_after(pop_curves, 0)

    print(f"  Population baseline (no adaptation) : {pop_acc:.1%}")
    print(f"  Subjects simulated  : {len(subjects)}")
    print(f"  Huber δ             : {huber_delta}   weight_decay : {weight_decay}\n")

    # Header
    col_w = 10
    header = f"  {'After N':>8}" + "".join(f"  lr={lr:.2f}".rjust(col_w) for lr in lrs)
    print(header)
    print("  " + "-" * (8 + col_w * len(lrs) + 2 * len(lrs)))

    checkpoints = [n for n in report_at]
    for n in checkpoints:
        row = f"  {n:>7} →"
        for lr in lrs:
            acc = _acc_after(results[lr], n)
            marker = " ←best" if acc == max(_acc_after(results[l], n) for l in lrs) else ""
            row += f"  {acc:>7.1%}{'':<3}"
        print(row)

    # ── Per-subject breakdown at best lr ─────────────────────────────────────
    # Find lr that maximises accuracy after n_warmup samples
    best_lr = best_lr if best_lr in lrs else max(lrs, key=lambda lr: _acc_after(results[lr], n_warmup))
    best_curves = results[best_lr]

    print(f"\n  Per-subject breakdown (lr={best_lr}, after {n_warmup} warm-up samples):")
    print(f"  {'Subject':<10} {'N':>4}  {'Pop acc':>8}  {'Adapted':>8}  {'Δ':>6}")
    print("  " + "-" * 42)
    for sid in subjects:
        n_samples = len(best_curves[sid])
        pop_s  = float(np.mean(pop_curves[sid]))
        ada_s  = float(np.mean(best_curves[sid][n_warmup:])) if n_samples > n_warmup else float("nan")
        delta  = ada_s - pop_s if not np.isnan(ada_s) else float("nan")
        d_str  = f"{delta:+.1%}" if not np.isnan(delta) else "   n/a"
        print(f"  {sid:<10} {n_samples:>4}  {pop_s:>8.1%}  "
              f"{ada_s if not np.isnan(ada_s) else 0:>8.1%}  {d_str:>6}")


if __name__ == "__main__":
    main()
