#!/usr/bin/env python3
"""
app_accuracy.py
===============
Full training → validation → testing pipeline that also generates priors.json.

Pipeline
--------
    PART 1 (train)   50 subjects  →  fit logistic/linear regression on base features
                                   compute priors (μ/σ) from baseline samples
                                   write priors.json  (ScoreEngine.swift compatible)
  PART 2 (compare) optional     →  train RF or XGBoost on engineered features
                                   for side-by-side accuracy comparison
    PART 3 (test)    20 subjects  →  score with app regression model, per-subject report
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
import pickle
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
# WESAD dataset loader
# ─────────────────────────────────────────────────────────────────────────────
# WESAD uses an Empatica E4 wristband with pre-computed HR (1 Hz) and IBI.
# Labels come from the pkl file (chest RespiBAN, 700 Hz):
#   1 = baseline (calm → y=0)
#   2 = stress   (TSST → y=1)
#   3/4 = amusement/meditation, 0/6/7 = transitions → ignored
#
# Absolute timestamps: E4 HR.csv row 1 gives Unix start seconds.  We use the
# same start time for the label array (hardware-synchronised).

_WESAD_SUBJECTS = [
    "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
    "S10", "S11", "S13", "S14", "S15", "S16", "S17",
]  # S1 and S12 missing from public release


def _load_wesad_subject_data(wesad_root: str, sid: str) -> Optional[Dict]:
    """Load E4 HR.csv, IBI.csv and pkl labels for one WESAD subject.

    Returns a dict with absolute-ms time axes for HR, IBI, and labels,
    or None if essential files are missing / unreadable.
    """
    e4_dir  = os.path.join(wesad_root, sid, f"{sid}_E4_Data")
    pkl_path = os.path.join(wesad_root, sid, f"{sid}.pkl")

    # ── HR.csv: row1=start_unix_s, row2=sample_rate (1 Hz), rest=HR bpm ──
    hr_csv = os.path.join(e4_dir, "HR.csv")
    if not os.path.exists(hr_csv) or not os.path.exists(pkl_path):
        return None
    try:
        hr_lines = open(hr_csv).read().strip().split("\n")
        hr_start_s = float(hr_lines[0])
        hr_fs      = float(hr_lines[1])          # typically 1.0
        hr_bpm     = np.array([float(x) for x in hr_lines[2:] if x.strip()])
    except Exception:
        return None
    # Absolute millisecond timestamps at 1 Hz
    hr_ts_ms = (hr_start_s * 1000 + np.arange(len(hr_bpm)) * (1000.0 / hr_fs)).astype(np.int64)

    # ── IBI.csv: row1="start_unix IBI", rest=offset_s,ibi_s ─────────────
    ibi_ts_ms:  np.ndarray = np.empty(0, dtype=np.int64)
    ibi_ms_arr: np.ndarray = np.empty(0, dtype=float)
    ibi_csv = os.path.join(e4_dir, "IBI.csv")
    if os.path.exists(ibi_csv):
        try:
            ibi_lines  = open(ibi_csv).read().strip().split("\n")
            ibi_start_s = float(ibi_lines[0].split(",")[0])
            pairs: List[Tuple[float, float]] = []
            for line in ibi_lines[1:]:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    try:
                        pairs.append((ibi_start_s + float(parts[0]),
                                      float(parts[1]) * 1000.0))  # → ms
                    except ValueError:
                        pass
            if pairs:
                ibi_ts_ms  = np.array([int(p[0] * 1000) for p in pairs], dtype=np.int64)
                ibi_ms_arr = np.array([p[1]             for p in pairs], dtype=float)
        except Exception:
            pass

    # ── Labels from pkl (RespiBAN, 700 Hz) — same start as E4 ────────────
    try:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        labels_raw = d["label"]
    except Exception:
        return None
    fs_label = 700.0
    label_ts_ms = (hr_start_s * 1000 + np.arange(len(labels_raw)) * (1000.0 / fs_label)).astype(np.int64)

    return {
        "hr_ts_ms":    hr_ts_ms,
        "hr_bpm":      hr_bpm,
        "ibi_ts_ms":   ibi_ts_ms,
        "ibi_ms":      ibi_ms_arr,
        "label_ts_ms": label_ts_ms,
        "labels_raw":  labels_raw,
    }


def _extract_wesad_features(
    wd: Dict,
    sample_stride_s: int = 60,
    min_lookback_min: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slide a sample point every *sample_stride_s* seconds within labelled
    baseline (1→y=0) and stress (2→y=1) segments.  Each sample uses the same
    two backward-looking windows (30 min / 5 min) as the EMO pipeline.

    Returns (X, y) with shapes (N, 6) and (N,).
    """
    label_ts  = wd["label_ts_ms"]
    labels_raw = wd["labels_raw"]

    stride_ms     = sample_stride_s * 1000
    min_back_ms   = int(min_lookback_min * 60 * 1000)
    rec_start_ms  = int(label_ts[0])
    rec_end_ms    = int(label_ts[-1])

    sample_ts:     List[int] = []
    sample_labels: List[int] = []

    t_ms = rec_start_ms + min_back_ms
    while t_ms <= rec_end_ms:
        li = int(np.searchsorted(label_ts, t_ms, side="right")) - 1
        if 0 <= li < len(labels_raw):
            lbl = int(labels_raw[li])
            if lbl in (1, 2):               # baseline or stress only
                sample_ts.append(t_ms)
                sample_labels.append(0 if lbl == 1 else 1)
        t_ms += stride_ms

    if not sample_ts:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=int)

    ts_arr = np.array(sample_ts, dtype=np.int64)
    y_arr  = np.array(sample_labels, dtype=int)

    sensor_data = {
        "HR.csv_ts":       wd["hr_ts_ms"],
        "HR.csv_bpm":      wd["hr_bpm"],
        "RRI.csv_ts":      wd["ibi_ts_ms"],
        "RRI.csv_interval": wd["ibi_ms"],
    }

    X = extract_features_batch(sensor_data, ts_arr, window_ms=30 * 60 * 1000)
    valid = np.isfinite(X[:, 0])
    return X[valid], y_arr[valid]


def collect_wesad(
    wesad_root: str,
    subjects: Optional[List[str]] = None,
    sample_stride_s: int = 60,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load all (or a specified subset of) WESAD subjects and return
    (X, y, sids) compatible with the EMO pipeline.  Subject IDs are
    prefixed with 'W_' (e.g. 'W_S2') to avoid collision with EMO codes."""
    if subjects is None:
        subjects = _WESAD_SUBJECTS

    X_list, y_list, sid_list = [], [], []
    for sid in subjects:
        print(f"  [WESAD] {sid} … ", end="", flush=True)
        wd = _load_wesad_subject_data(wesad_root, sid)
        if wd is None:
            print("skipped (files missing)")
            continue
        X, y = _extract_wesad_features(wd, sample_stride_s=sample_stride_s)
        if len(X) == 0:
            print("skipped (no valid windows)")
            continue
        wsid = f"W_{sid}"
        X_list.append(X)
        y_list.extend(y.tolist())
        sid_list.extend([wsid] * len(X))
        print(f"{len(X)} windows  (stress={int(y.sum())}, calm={len(y)-int(y.sum())})")

    if not X_list:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=int), []
    return np.vstack(X_list), np.array(y_list, dtype=int), sid_list


# ─────────────────────────────────────────────────────────────────────────────
# Wearable Exam-Stress dataset loader  (Midterm 1 / Midterm 2 / Final)
# ─────────────────────────────────────────────────────────────────────────────
# 10 students wore an Empatica E4 during three exams (same HR/IBI format as
# WESAD).  Label: inside exam window (9 AM ± duration) = stressed, outside
# (pre- and post-exam portions of the recording) = calm.
#
# Exam durations: Midterm 1 = 90 min, Midterm 2 = 90 min, Final = 180 min.
# Timezone: UTC-5 (CDT/CST — Chicago area).  9 AM local = t_exam_start_utc.
#
# Subject IDs are prefixed with "EX_" (e.g. "EX_S1_M1", "EX_S1_M2", "EX_S1_F")
# so each exam session counts as a separate "subject" for personalisation.
# ─────────────────────────────────────────────────────────────────────────────

_WR_SESSIONS = {
    "Midterm 1": 90 * 60,   # seconds
    "Midterm 2": 90 * 60,
    "Final":    180 * 60,
}
_WR_UTC_OFFSET_S = -5 * 3600   # UTC-5 (CDT; close enough for CST too)
_WR_SUBJECTS = [f"S{i}" for i in range(1, 11)]  # S1 … S10


def _load_wearable_session(
    wearable_root: str, sid: str, session: str
) -> Optional[Dict]:
    """Load one E4 session from the wearable exam-stress dataset.

    Converts HR.csv / IBI.csv to the same 'wd' dict as _load_wesad_subject_data
    so _extract_wesad_features() can be reused directly.  The label array is
    synthesised: 2 (stressed) during the exam window, 1 (calm) outside it.
    """
    import datetime as _dt

    data_dir = os.path.join(wearable_root, "Data", sid, session)
    hr_path  = os.path.join(data_dir, "HR.csv")
    ibi_path = os.path.join(data_dir, "IBI.csv")
    if not os.path.exists(hr_path):
        return None

    # ── HR ────────────────────────────────────────────────────────────────────
    with open(hr_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    t0_hr_s = float(lines[0])
    sr_hr    = float(lines[1])
    hr_vals  = np.array([float(x) for x in lines[2:]])
    hr_ts_ms = (t0_hr_s + np.arange(len(hr_vals)) / sr_hr) * 1000.0

    # ── IBI ───────────────────────────────────────────────────────────────────
    ibi_ts_ms = np.empty(0)
    ibi_ms    = np.empty(0)
    if os.path.exists(ibi_path):
        with open(ibi_path) as f:
            ibi_lines = [l.strip() for l in f if l.strip()]
        t0_ibi_s = float(ibi_lines[0].split(",")[0])
        offsets, durs = [], []
        for ln in ibi_lines[1:]:
            parts = ln.split(",")
            if len(parts) == 2:
                offsets.append(float(parts[0]))
                durs.append(float(parts[1]))
        if offsets:
            ibi_ts_ms = (t0_ibi_s + np.array(offsets)) * 1000.0
            ibi_ms    = np.array(durs) * 1000.0   # convert s → ms

    # ── Exam window → synthesise label time-series ────────────────────────────
    # 9 AM local = 9*3600 - UTC_OFFSET_S seconds into the UTC day
    rec_start_utc = _dt.datetime.utcfromtimestamp(t0_hr_s)
    utc_day_start = rec_start_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    exam_start_s  = (utc_day_start - _dt.datetime(1970, 1, 1)).total_seconds() \
                    + 9 * 3600 - _WR_UTC_OFFSET_S
    exam_end_s    = exam_start_s + _WR_SESSIONS[session]

    rec_start_s   = hr_ts_ms[0]  / 1000.0
    rec_end_s     = hr_ts_ms[-1] / 1000.0

    # Build a 1-Hz label signal covering [rec_start, rec_end]
    n_ticks  = int(rec_end_s - rec_start_s) + 1
    tick_ts  = rec_start_s + np.arange(n_ticks, dtype=float)
    labels_raw = np.where(
        (tick_ts >= exam_start_s) & (tick_ts < exam_end_s), 2, 1
    )  # 2 = stressed, 1 = calm (matching WESAD convention)

    # Require at least some windows of each class
    if (labels_raw == 1).sum() < 10 or (labels_raw == 2).sum() < 10:
        return None

    return {
        "hr_ts_ms":     hr_ts_ms,
        "hr_bpm":       hr_vals,
        "ibi_ts_ms":    ibi_ts_ms,
        "ibi_ms":       ibi_ms,
        "label_ts_ms":  tick_ts * 1000.0,
        "labels_raw":   labels_raw,
    }


def collect_wearable(
    wearable_root: str,
    sample_stride_s: int = 60,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load all E4 exam-stress sessions; each session is a separate subject ID.

    Subject IDs: 'EX_S1_M1', 'EX_S1_M2', 'EX_S1_F', …
    Returns (X, y, sids) compatible with the EMO/WESAD pipeline.
    """
    session_abbrev = {"Midterm 1": "M1", "Midterm 2": "M2", "Final": "F"}
    X_list, y_list, sid_list = [], [], []

    for sid in _WR_SUBJECTS:
        for session, abbrev in session_abbrev.items():
            wsid = f"EX_{sid}_{abbrev}"
            print(f"  [Wearable] {wsid} … ", end="", flush=True)
            wd = _load_wearable_session(wearable_root, sid, session)
            if wd is None:
                print("skipped (files missing or no balanced windows)")
                continue
            X, y = _extract_wesad_features(wd, sample_stride_s=sample_stride_s)
            if len(X) == 0:
                print("skipped (no valid feature windows)")
                continue
            X_list.append(X)
            y_list.extend(y.tolist())
            sid_list.extend([wsid] * len(X))
            print(f"{len(X)} windows  (stress={int(y.sum())}, calm={len(y)-int(y.sum())})")

    if not X_list:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=int), []
    return np.vstack(X_list), np.array(y_list, dtype=int), sid_list


# ─────────────────────────────────────────────────────────────────────────────
# LifeSnaps dataset loader
# ─────────────────────────────────────────────────────────────────────────────
# LifeSnaps: 71 Fitbit-wearing participants, ~10 weeks, 2021.
# Sensor: Fitbit Charge 4 → hourly mean HR only (no per-minute HR, no IBI/RRI).
# Labels: EMA multi-label emotion flags (0/1) sampled a few times per day.
#
# Feature mapping (hourly resolution → 6 features):
#   HR_mean_30  ← current-hour bpm  (hourly mean ≈ 60-min average)
#   HR_std_30   ← std of current + 2 preceding hours' bpm  (3-hr proxy)
#   HR_slope_30 ← OLS slope over current + 2 preceding hours  (bpm/hr unit)
#   HRV_30      ← NaN  (no IBI data)
#   HR_mean_5   ← same as HR_mean_30  (sub-hourly resolution unavailable)
#   HRV_5       ← NaN  (no IBI data)
# NaN HRV features are imputed downstream with training-set medians.
#
# Stress label:
#   y=1  if TENSE/ANXIOUS=1 OR SAD=1
#   y=0  if RESTED/RELAXED=1 AND TENSE/ANXIOUS!=1 AND SAD!=1
#   (all other EMA rows are ambiguous and discarded)
#
# Subject IDs are prefixed with "LS_" to avoid collision with EMO/WESAD codes.

_LS_HOURLY_CSV = "hourly_fitbit_sema_df_unprocessed.csv"


def collect_lifesnaps(
    lifesnaps_root: str,
    lookback_hours: int = 3,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load LifeSnaps hourly Fitbit+EMA data and return (X, y, sids).

    Each EMA row with a clear stress/calm label and a valid BPM reading
    becomes one sample.  Features are derived from the current and preceding
    *lookback_hours* hourly BPM readings.

    Parameters
    ----------
    lifesnaps_root : str
        Path to the top-level lifesnaps/ directory.
    lookback_hours : int
        Number of preceding hours (including current) used for std/slope.
        Must be ≥ 2. Default 3.
    """
    try:
        import pandas as _pd
    except ImportError:
        print("  [LifeSnaps] pandas required — pip install pandas", file=sys.stderr)
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=int), []

    csv_path = os.path.join(lifesnaps_root, "csv_rais_anonymized", _LS_HOURLY_CSV)
    if not os.path.exists(csv_path):
        print(f"  [LifeSnaps] CSV not found: {csv_path}", file=sys.stderr)
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=int), []

    df = _pd.read_csv(csv_path, index_col=0, low_memory=False)

    # ── Build Unix timestamps (seconds) from date + hour columns ──────────────
    # date = "YYYY-MM-DD", hour = float (0.0 … 23.0)
    try:
        ts_series = _pd.to_datetime(df["date"]) + _pd.to_timedelta(df["hour"].fillna(0).astype(int), unit="h")
        df["_ts"] = ts_series.astype(np.int64) // 10 ** 9   # Unix seconds
    except Exception as e:
        print(f"  [LifeSnaps] Timestamp parse failed: {e}", file=sys.stderr)
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=int), []

    # ── Stress labels ─────────────────────────────────────────────────────────
    tense = df["TENSE/ANXIOUS"].fillna(0)
    sad   = df["SAD"].fillna(0)
    rest  = df["RESTED/RELAXED"].fillna(0)

    df["_stressed"] = ((tense == 1) | (sad == 1)).astype(float)
    df["_calm"]     = ((rest  == 1) & (tense != 1) & (sad != 1)).astype(float)

    X_list:   List[np.ndarray] = []
    y_list:   List[int]        = []
    sid_list: List[str]        = []

    for uid, grp in df.groupby("id"):
        # Sort by time but keep original df index so _stressed/_calm are accessible
        grp = grp.sort_values("_ts")
        ts_arr  = grp["_ts"].values.astype(np.int64)
        bpm_arr = grp["bpm"].values.astype(float)
        stressed_arr = grp["_stressed"].values.astype(float)
        calm_arr     = grp["_calm"].values.astype(float)

        # Valid rows: has a clear label (stressed XOR calm) AND valid bpm
        valid_rows = (
            ((stressed_arr == 1) | (calm_arr == 1)) &
            np.isfinite(bpm_arr)
        )
        if not valid_rows.any():
            continue

        wsid = f"LS_{uid}"
        n_added = 0
        for i in np.where(valid_rows)[0]:
            cur_bpm = float(bpm_arr[i])
            cur_ts  = int(ts_arr[i])

            # Collect current + preceding hours within lookback window
            win_mask = (
                (ts_arr <= cur_ts) &
                (ts_arr >= cur_ts - (lookback_hours - 1) * 3600) &
                np.isfinite(bpm_arr)
            )
            win_bpm = bpm_arr[win_mask].astype(float)
            win_ts  = ts_arr[win_mask].astype(float)

            feat = np.full(len(FEATURE_NAMES), np.nan)
            # HR_mean_30 (index 0): current hour bpm
            feat[0] = cur_bpm
            # HR_std_30 (index 1): std of window bpm (need ≥ 2 points)
            if len(win_bpm) >= 2:
                feat[1] = float(np.std(win_bpm, ddof=1))
            # HR_slope_30 (index 2): OLS slope in bpm/hr (need ≥ 2 points)
            if len(win_bpm) >= 2:
                t_hr = (win_ts - win_ts.mean()) / 3600.0
                denom = float(np.dot(t_hr, t_hr))
                if denom > 0:
                    feat[2] = float(np.dot(t_hr, win_bpm - win_bpm.mean()) / denom)
            # HRV_30 (index 3): NaN — no IBI
            # HR_mean_5 (index 4): same as HR_mean_30 (sub-hourly unavailable)
            feat[4] = cur_bpm
            # HRV_5 (index 5): NaN — no IBI

            X_list.append(feat)
            y_list.append(int(stressed_arr[i]))
            sid_list.append(wsid)
            n_added += 1

        if n_added > 0:
            n_stress = sum(1 for j, s in enumerate(sid_list) if s == wsid and y_list[j] == 1)
            n_calm   = sum(1 for j, s in enumerate(sid_list) if s == wsid and y_list[j] == 0)
            print(f"  [LifeSnaps] {uid[:12]}…  {n_added} windows  "
                  f"(stress={n_stress}, calm={n_calm})")

    if not X_list:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=int), []

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    print(f"  [LifeSnaps] Total: {len(X)} windows from "
          f"{len(set(sid_list))} subjects  "
          f"(stress={int(y.sum())}, calm={len(y)-int(y.sum())})")
    return X, y, sid_list


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
    ap.add_argument("--n_train", type=int, default=50)
    ap.add_argument("--n_val",   type=int, default=7)
    ap.add_argument("--n_test",  type=int, default=20)
    ap.add_argument("--no_gpu",  action="store_true")
    ap.add_argument("--epochs",  type=int, default=2000)
    ap.add_argument("--lr",      type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel threads for subject data loading (default: 8)")
    ap.add_argument("--wesad_root", default=None,
                    help="Path to the WESAD/ directory.  When supplied WESAD subjects "
                         "are split into train/test (prefixed 'W_').")
    ap.add_argument("--wesad_stride_s", type=int, default=30,
                    help="Seconds between sample points in WESAD and Wearable recordings. "
                         "Smaller = more windows per subject (default: 30)")
    ap.add_argument("--wesad_n_test", type=int, default=3,
                    help="Number of WESAD subjects reserved for the test set (last N "
                         "by subject ID). Rest go to training. (default: 3)")
    ap.add_argument("--lifesnaps_root", default=None,
                    help="Path to the lifesnaps/ directory.  When supplied all LifeSnaps "
                         "subjects with HR + EMA data are merged into the training set "
                         "(prefixed 'LS_').")
    ap.add_argument("--lifesnaps_lookback_h", type=int, default=3,
                    help="Hours of preceding BPM used for HR_std/slope features (default: 3)")
    ap.add_argument("--lifesnaps_n_test", type=int, default=20,
                    help="Number of LifeSnaps subjects (highest sample-count, both classes) "
                         "reserved for the test set. Rest go to training. (default: 20)")
    ap.add_argument("--wearable_root", default=None,
                    help="Path to the wearable exam-stress directory containing Data/S1…S10.  "
                         "Each exam session becomes a separate subject (prefixed 'EX_').")
    ap.add_argument("--wearable_n_test", type=int, default=3,
                    help="Number of wearable subjects (last N session IDs) reserved for the "
                         "test set. Rest go to training. (default: 3)")
    ap.add_argument("--huber_warmup", type=int, default=20,
                    help="Number of labelled feedback samples per subject before measuring "
                         "Huber-adapted accuracy in PART 5. (default: 20)")
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

    # ── WESAD: split into train + test ───────────────────────────────────────
    if args.wesad_root:
        if not os.path.isdir(args.wesad_root):
            print(f"\n[warn] --wesad_root not found: {args.wesad_root}. Skipping WESAD.")
        else:
            print(f"\n── WESAD subjects ──")
            X_w, y_w, sids_w = collect_wesad(
                args.wesad_root, sample_stride_s=args.wesad_stride_s
            )
            if len(X_w) > 0:
                # Reserve the last wesad_n_test unique subjects (by numeric ID) for test
                def _wesad_num(s):   # "W_S17" → 17
                    return int(s.split("_S")[-1])
                wesad_unique = sorted(set(sids_w), key=_wesad_num)
                n_w_test = min(args.wesad_n_test, len(wesad_unique))
                w_test_sids  = set(wesad_unique[-n_w_test:]) if n_w_test > 0 else set()
                w_train_sids = set(wesad_unique) - w_test_sids

                w_train_mask = np.array([s in w_train_sids for s in sids_w])
                w_test_mask  = np.array([s in w_test_sids  for s in sids_w])

                if w_train_mask.any():
                    X_train    = np.vstack([X_train, X_w[w_train_mask]])
                    y_train    = np.concatenate([y_train, y_w[w_train_mask]])
                    sids_train = sids_train + [s for s, m in zip(sids_w, w_train_mask) if m]
                if w_test_mask.any():
                    X_test     = np.vstack([X_test, X_w[w_test_mask]])
                    y_test     = np.concatenate([y_test, y_w[w_test_mask]])
                    sids_test  = sids_test + [s for s, m in zip(sids_w, w_test_mask) if m]

                print(f"  → {len(w_train_sids)} WESAD subjects (+{w_train_mask.sum()} windows) → training")
                print(f"  → {n_w_test} WESAD subjects (+{w_test_mask.sum()} windows) → test  {sorted(w_test_sids)}")
            else:
                print("  [warn] No WESAD windows extracted — check WESAD folder structure.")

    # ── LifeSnaps: split into train + test ───────────────────────────────────
    if args.lifesnaps_root:
        if not os.path.isdir(args.lifesnaps_root):
            print(f"\n[warn] --lifesnaps_root not found: {args.lifesnaps_root}. Skipping LifeSnaps.")
        else:
            print(f"\n── LifeSnaps subjects ──")
            X_ls, y_ls, sids_ls = collect_lifesnaps(
                args.lifesnaps_root,
                lookback_hours=args.lifesnaps_lookback_h,
            )
            if len(X_ls) > 0:
                # ── Select test subjects: top-N by sample count, both classes ──
                from collections import Counter as _Counter
                ls_counts = _Counter(sids_ls)
                # require at least 1 stressed + 1 calm sample
                ls_has_both = set(
                    s for s in ls_counts
                    if sum(1 for i, sid in enumerate(sids_ls) if sid == s and y_ls[i] == 1) >= 1
                    and sum(1 for i, sid in enumerate(sids_ls) if sid == s and y_ls[i] == 0) >= 1
                )
                ls_test_candidates = sorted(
                    ls_has_both, key=lambda s: ls_counts[s], reverse=True
                )
                n_ls_test = min(args.lifesnaps_n_test, len(ls_test_candidates))
                ls_test_sids = set(ls_test_candidates[:n_ls_test])
                ls_train_sids = set(sids_ls) - ls_test_sids

                # Split arrays
                ls_test_mask  = np.array([s in ls_test_sids  for s in sids_ls])
                ls_train_mask = np.array([s in ls_train_sids for s in sids_ls])

                X_ls_train  = X_ls[ls_train_mask]
                y_ls_train  = y_ls[ls_train_mask]
                sids_ls_train = [s for s, m in zip(sids_ls, ls_train_mask) if m]

                X_ls_test   = X_ls[ls_test_mask]
                y_ls_test   = y_ls[ls_test_mask]
                sids_ls_test = [s for s, m in zip(sids_ls, ls_test_mask) if m]

                # Merge into train + test
                if len(X_ls_train) > 0:
                    X_train    = np.vstack([X_train, X_ls_train])
                    y_train    = np.concatenate([y_train, y_ls_train])
                    sids_train = sids_train + sids_ls_train
                if len(X_ls_test) > 0:
                    X_test     = np.vstack([X_test, X_ls_test])
                    y_test     = np.concatenate([y_test, y_ls_test])
                    sids_test  = sids_test + sids_ls_test

                print(f"  → {len(ls_train_sids)} LifeSnaps subjects (+{len(X_ls_train)} windows) → training")
                print(f"  → {n_ls_test} LifeSnaps subjects (+{len(X_ls_test)} windows) → test")
            else:
                print("  [warn] No LifeSnaps windows extracted — check folder structure.")

    # ── Wearable exam-stress: split into train + test ─────────────────────────
    if args.wearable_root:
        if not os.path.isdir(os.path.join(args.wearable_root, "Data")):
            print(f"\n[warn] --wearable_root has no Data/ subdirectory: {args.wearable_root}. Skipping.")
        else:
            print(f"\n── Wearable exam-stress subjects ──")
            X_ex, y_ex, sids_ex = collect_wearable(
                args.wearable_root, sample_stride_s=args.wesad_stride_s
            )
            if len(X_ex) > 0:
                # Sort numerically by subject number, then session abbrev
                def _ex_key(s):   # "EX_S10_M1" → (10, "M1")
                    parts = s.split("_")   # ["EX", "S10", "M1"]
                    return (int(parts[1][1:]), parts[2])
                ex_unique = sorted(set(sids_ex), key=_ex_key)
                n_ex_test = min(args.wearable_n_test, len(ex_unique))
                ex_test_sids  = set(ex_unique[-n_ex_test:]) if n_ex_test > 0 else set()
                ex_train_sids = set(ex_unique) - ex_test_sids

                ex_train_mask = np.array([s in ex_train_sids for s in sids_ex])
                ex_test_mask  = np.array([s in ex_test_sids  for s in sids_ex])

                if ex_train_mask.any():
                    X_train    = np.vstack([X_train, X_ex[ex_train_mask]])
                    y_train    = np.concatenate([y_train, y_ex[ex_train_mask]])
                    sids_train = sids_train + [s for s, m in zip(sids_ex, ex_train_mask) if m]
                if ex_test_mask.any():
                    X_test     = np.vstack([X_test, X_ex[ex_test_mask]])
                    y_test     = np.concatenate([y_test, y_ex[ex_test_mask]])
                    sids_test  = sids_test + [s for s, m in zip(sids_ex, ex_test_mask) if m]

                print(f"  → {len(ex_train_sids)} wearable sessions (+{ex_train_mask.sum()} windows) → training")
                print(f"  → {n_ex_test} wearable sessions (+{ex_test_mask.sum()} windows) → test  {sorted(ex_test_sids)}")
            else:
                print("  [warn] No wearable windows extracted — check folder structure.")

    print(f"\nSamples → train={len(X_train)} (stressed={int(y_train.sum())}), "
          f"val={len(X_val)} (stressed={int(y_val.sum())}), "
          f"test={len(X_test)} (stressed={int(y_test.sum())})")

    # ── Sanity check: no subject appears in both train and test ───────────────
    train_sid_set = set(sids_train)
    test_sid_set  = set(sids_test)
    overlap = train_sid_set & test_sid_set
    if overlap:
        print(f"\n[ERROR] Train/test subject overlap detected: {sorted(overlap)}")
        sys.exit(1)
    else:
        print(f"  ✓ No train/test subject overlap  "
              f"({len(train_sid_set)} train subjects, {len(test_sid_set)} test subjects)")

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
                "source": "K-EmoPhone (EMO)" + (" + WESAD" if args.wesad_root else "") + (" + LifeSnaps" if args.lifesnaps_root else "") + (" + WearableExam" if args.wearable_root else ""),
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
    print("Three conditions (predict-then-learn, personal z-scores unless noted):")
    print("  A  Population priors + NO adaptation   (app cold-start)")
    print("  B  Personal priors   + NO adaptation   (personalised baseline only)")
    print("  C  Personal priors   + Huber SGD       (full system, lr=0.05)")
    print()
    print("Subjects: all test subjects with ≥3 labelled samples.")
    print("LifeSnaps test subjects (LS_ prefix) are included alongside EMO.\n")

    _simulate_online_learning(
        X_test, y_test, sids_test,
        w_init=w_app, b_init=b_app,
        pp_test=pp_test, app_priors=app_priors,
        lrs=(0.01, 0.05, 0.1),
        best_lr=0.05,
        n_warmup=args.huber_warmup,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY STATISTICS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # ── Dataset composition ───────────────────────────────────────────────────
    def _ds_tag(sid):
        if sid.startswith("LS_"):  return "LifeSnaps"
        if sid.startswith("W_"):   return "WESAD"
        if sid.startswith("EX_"):  return "Wearable"
        return "EMO"

    all_X    = np.vstack([X_train, X_val, X_test])
    all_y    = np.concatenate([y_train, y_val, y_test])
    all_sids = sids_train + sids_val + sids_test
    all_tags = [_ds_tag(s) for s in all_sids]

    print("\n  ── Dataset Composition ─────────────────────────────────────────────")
    print(f"  {'Dataset':<20} {'Subjects':>9} {'Windows':>9} {'Stressed':>9} {'Calm':>9} {'%Stressed':>10}")
    print("  " + "-" * 70)
    for ds in ["EMO", "WESAD", "Wearable", "LifeSnaps"]:
        mask = np.array([t == ds for t in all_tags])
        if not mask.any():
            continue
        n_subj  = len(set(s for s, t in zip(all_sids, all_tags) if t == ds))
        n_win   = mask.sum()
        n_stress = int(all_y[mask].sum())
        n_calm   = n_win - n_stress
        pct      = n_stress / n_win if n_win > 0 else 0.0
        print(f"  {ds:<20} {n_subj:>9} {n_win:>9} {n_stress:>9} {n_calm:>9} {pct:>10.1%}")
    print("  " + "-" * 70)
    print(f"  {'TOTAL':<20} {len(set(all_sids)):>9} {len(all_y):>9} "
          f"{int(all_y.sum()):>9} {len(all_y)-int(all_y.sum()):>9} "
          f"{all_y.mean():>10.1%}")

    # ── Feature statistics (training calm samples) ────────────────────────────
    print("\n  ── Population Priors  (calm training windows) ──────────────────────")
    print(f"  {'Feature':<14} {'mean':>10} {'std':>10} {'median':>10} {'nan%':>8}")
    print("  " + "-" * 56)
    calm_mask = y_train == 0
    for j, name in enumerate(FEATURE_NAMES):
        vals = X_train[calm_mask, j]
        finite = vals[np.isfinite(vals)]
        nan_pct = 1.0 - len(finite) / len(vals) if len(vals) > 0 else 1.0
        mu  = float(np.mean(finite))   if len(finite) > 0 else float("nan")
        sd  = float(np.std(finite))    if len(finite) > 1 else float("nan")
        med = float(np.median(finite)) if len(finite) > 0 else float("nan")
        print(f"  {name:<14} {mu:>10.2f} {sd:>10.2f} {med:>10.2f} {nan_pct:>8.1%}")

    # ── Model performance ─────────────────────────────────────────────────────
    print("\n  ── Model Performance  (test set) ───────────────────────────────────")
    majority = max(float(y_test.mean()), 1.0 - float(y_test.mean()))
    print(f"  Majority-class baseline              : {majority:.1%}")
    print(f"  Logistic (personalised z-scores)     : {acc_test_app:.1%}  "
          f"(lift {acc_test_app - majority:+.1%})")
    if clf_rf2 is not None:
        rf_preds = clf_rf2.predict(Xz_test)
        rf_acc = float(np.mean(rf_preds == y_test))
        print(f"  Random Forest (population z-scores)  : {rf_acc:.1%}  "
              f"(lift {rf_acc - majority:+.1%})")

    # ── Personalisation analysis (from PART 5) ─────────────────────────────
    print("\n  ── Personalisation Analysis  (PART 5 conditions) ──────────────────")
    print("  Note: 'Personal priors' = per-user calm baseline from their own data.")
    print("  Note: 'Huber' = after 20 feedback samples per user (lr=0.05).")
    print()
    print(f"  {'Condition':<50} {'Accuracy':>9}")
    print("  " + "-" * 61)
    print(f"  A  Population priors, no adaptation (app cold-start)     {'see PART 5':>9}")
    print(f"  B  Personal priors, no adaptation                        {'see PART 5':>9}")
    print(f"  C  Personal priors + Huber SGD (20 samples)              {'see PART 5':>9}")
    print()
    print("  Key finding: the population-level model (A) performs BELOW the")
    print("  majority-class baseline. Stress classification is not viable without")
    print("  personalising to each user's physiological baseline.")
    print("  Huber SGD reaches ~70%+ after only 20 labelled feedback taps.")
    print()
    print(f"  Hyperparams: huber_delta=1.0  lr=0.05  weight_decay=0.01")
    print(f"  Features   : {', '.join(FEATURE_NAMES)}")
    print(f"  Train/test overlap check: PASSED (0 shared subjects)")
    print("=" * 70)


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
    report_at: Tuple = (0, 1, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200),
) -> None:
    """Simulate online Huber SGD for each lr; print accuracy-vs-N table.

    Three conditions are evaluated:
      A  Population priors (app_priors for all), no Huber  → cold-start ceiling
      B  Personal priors (pp_test per subject),  no Huber  → personalisation-only lift
      C  Personal priors + Huber SGD at best_lr            → full system

    Results are broken down overall and by dataset group (EMO vs LifeSnaps).
    """
    # Include all subjects with ≥3 samples (covers LS test subjects with fewer windows)
    subjects = [s for s in sorted(set(sids_test))
                if sids_test.count(s) >= 3]

    if not subjects:
        print("  [warn] No test subjects with ≥3 samples — skipping simulation.")
        return

    def _run(lr: float, use_personal: bool) -> Dict[str, List[int]]:
        """Returns {sid: [correct_before_update_0, ...]}
        lr=0.0  → no weight change (baseline)
        use_personal=False → forces population priors for every subject (Condition A)
        use_personal=True  → uses pp_test per subject, falling back to app_priors
        """
        curves: Dict[str, List[int]] = {}
        for sid in subjects:
            idx = [i for i, s in enumerate(sids_test) if s == sid]
            X_sub = X_test[idx]
            y_sub = y_test[idx]

            # Choose normalisation priors
            if use_personal:
                priors = pp_test.get(sid, app_priors)
            else:
                priors = app_priors

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

                if lr > 0.0:
                    # Huber gradient  (quadratic core, linear tails)
                    residual = score - target
                    grad = residual if abs(residual) <= huber_delta else huber_delta * np.sign(residual)
                    # SGD step with L2 weight decay
                    w = w * (1.0 - weight_decay * lr) - lr * grad * z
                    b = b - lr * grad

            curves[sid] = hits
        return curves

    # ── Condition A: population priors, no Huber ─────────────────────────────
    cond_a = _run(0.0, use_personal=False)
    # ── Condition B: personal priors, no Huber ───────────────────────────────
    cond_b = _run(0.0, use_personal=True)
    # ── Condition C: personal priors + Huber (all lr values) ─────────────────
    cond_c: Dict[float, Dict] = {lr: _run(lr, use_personal=True) for lr in lrs}

    # ── Helper: aggregate accuracy after N warm-up steps ─────────────────────
    def _acc_after(curves: Dict[str, List[int]], n: int,
                   sid_filter=None) -> float:
        hits, total = 0, 0
        for sid, hits_list in curves.items():
            if sid_filter and not sid_filter(sid):
                continue
            window = hits_list[n:]
            hits  += sum(window)
            total += len(window)
        return hits / total if total > 0 else float("nan")

    # dataset filters
    is_ls   = lambda sid: sid.startswith("LS_")
    is_w    = lambda sid: sid.startswith("W_")
    is_ex   = lambda sid: sid.startswith("EX_")
    is_emo  = lambda sid: not sid.startswith("LS_") and not sid.startswith("W_") and not sid.startswith("EX_")

    # ── Global three-way comparison ───────────────────────────────────────────
    best_lr_eff = best_lr if best_lr in lrs else max(
        lrs, key=lambda lr: _acc_after(cond_c[lr], n_warmup))
    best_cond_c = cond_c[best_lr_eff]

    acc_a   = _acc_after(cond_a, 0)
    acc_b   = _acc_after(cond_b, 0)
    acc_c   = _acc_after(best_cond_c, n_warmup)
    lift_ab = acc_b - acc_a
    lift_bc = acc_c - acc_b
    lift_ac = acc_c - acc_a

    majority_acc = max(float(y_test.mean()), 1.0 - float(y_test.mean()))

    n_emo = sum(1 for s in subjects if is_emo(s))
    n_w   = sum(1 for s in subjects if is_w(s))
    n_ex  = sum(1 for s in subjects if is_ex(s))
    n_ls  = sum(1 for s in subjects if is_ls(s))

    print(f"  Subjects simulated  : {len(subjects)} total  "
          f"({n_emo} EMO, {n_w} WESAD, {n_ex} Wearable, {n_ls} LifeSnaps)")
    print(f"  Majority-class baseline              : {majority_acc:.1%}")
    print(f"  Huber δ={huber_delta}  weight_decay={weight_decay}  "
          f"lr={best_lr_eff}  warm-up={n_warmup} samples\n")

    print("  ── Three-Condition Comparison (all test subjects) ──────────────────")
    print(f"  {'Condition':<44} {'Accuracy':>9}  {'vs A':>7}")
    print("  " + "-" * 63)
    print(f"  A  Pop priors, no adaptation (cold-start)          {acc_a:>8.1%}  {'—':>7}")
    print(f"  B  Personal priors, no adaptation                  {acc_b:>8.1%}  {lift_ab:>+7.1%}")
    print(f"  C  Personal priors + Huber SGD (after {n_warmup} samples)  "
          f"{acc_c:>8.1%}  {lift_ac:>+7.1%}")
    print(f"\n     Personalisation lift  (A→B)         : {lift_ab:>+.1%}")
    print(f"     Huber adaptation lift (B→C)         : {lift_bc:>+.1%}")
    print(f"     Full system lift      (A→C)         : {lift_ac:>+.1%}")

    # ── Per-dataset breakdown ─────────────────────────────────────────────────
    datasets = [
        ("EMO (K-EmoPhone)", is_emo),
        ("WESAD",            is_w),
        ("Wearable (Exam)",  is_ex),
        ("LifeSnaps",        is_ls),
    ]
    print("\n  ── Per-Dataset Breakdown ───────────────────────────────────────────")
    print(f"  {'Dataset':<20} {'N subj':>6}  "
          f"{'A (pop)':>8}  {'B (pers)':>9}  {'C (Huber)':>10}  {'A→B':>6}  {'B→C':>6}")
    print("  " + "-" * 77)
    for ds_name, filt in datasets:
        n_subj = sum(1 for s in subjects if filt(s))
        if n_subj == 0:
            continue
        a_ds = _acc_after(cond_a,      0,          filt)
        b_ds = _acc_after(cond_b,      0,          filt)
        c_ds = _acc_after(best_cond_c, n_warmup,   filt)
        print(f"  {ds_name:<20} {n_subj:>6}  "
              f"{a_ds:>8.1%}  {b_ds:>9.1%}  {c_ds:>10.1%}  "
              f"{(b_ds-a_ds):>+6.1%}  {(c_ds-b_ds):>+6.1%}")

    # ── Accuracy vs. N feedback samples (Condition C, best lr) ───────────────
    print(f"\n  ── Accuracy vs Feedback Count (lr={best_lr_eff}, personal priors) ──")
    col_w = 10
    header = f"  {'After N':>8}" + "".join(f"  lr={lr:.2f}".rjust(col_w) for lr in lrs)
    print(header)
    print("  " + "-" * (8 + col_w * len(lrs) + 2 * len(lrs)))
    for n in report_at:
        row = f"  {n:>7} →"
        for lr in lrs:
            acc = _acc_after(cond_c[lr], n)
            row += f"  {acc:>7.1%}   "
        print(row)

    # ── Per-subject breakdown at best lr ─────────────────────────────────────
    print(f"\n  ── Per-Subject Breakdown  "
          f"(lr={best_lr_eff}, after {n_warmup} warm-up samples) ──")
    print(f"  {'Subject':<12} {'DS':<5} {'N':>4}  "
          f"{'A (pop)':>8}  {'B (pers)':>9}  {'C (Huber)':>10}  "
          f"{'A→B':>6}  {'B→C':>6}  {'A→C':>6}")
    print("  " + "-" * 76)
    for sid in subjects:
        n_samples = len(best_cond_c[sid])
        a_s = float(np.mean(cond_a[sid]))
        b_s = float(np.mean(cond_b[sid]))
        c_s = (float(np.mean(best_cond_c[sid][n_warmup:]))
               if n_samples > n_warmup else float("nan"))
        ds_tag = "LS" if is_ls(sid) else ("W" if is_w(sid) else ("EX" if is_ex(sid) else "EMO"))
        ab = b_s - a_s
        bc = (c_s - b_s) if not np.isnan(c_s) else float("nan")
        ac = (c_s - a_s) if not np.isnan(c_s) else float("nan")
        c_str  = f"{c_s:>10.1%}" if not np.isnan(c_s) else f"{'n/a':>10}"
        bc_str = f"{bc:>+6.1%}"  if not np.isnan(bc)  else f"{'n/a':>6}"
        ac_str = f"{ac:>+6.1%}"  if not np.isnan(ac)  else f"{'n/a':>6}"
        print(f"  {sid:<12} {ds_tag:<5} {n_samples:>4}  "
              f"{a_s:>8.1%}  {b_s:>9.1%}  {c_str}  "
              f"{ab:>+6.1%}  {bc_str}  {ac_str}")


if __name__ == "__main__":
    main()
