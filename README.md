# WatchStress

**Version 1.0.2**

An Apple Watch + iPhone app that estimates stress in real-time from wrist physiological signals (heart rate and HRV), trained on four labelled datasets and personalized per-user through online Huber learning.

---

## Overview

WatchStress is a SwiftUI iOS/watchOS companion app plus a Python toolkit for training stress models. The app reads HealthKit vitals (heart rate, HRV SDNN), converts them into a compact 6-feature vector, and computes a stress score on a −100…+100 scale (positive = calm, negative = stressed). It collects self-reports and adapts to each user over time via on-device Huber SGD.

```
Offline training  ──────────────────────────────────────────────────────────
  EMO (K-EmoPhone) + WESAD + LifeSnaps + Wearable Exam-Stress
  ↓  app_accuracy.py
  priors.json  (population μ/σ, logistic weights w, bias b)
  ↓
App cold-start  ────────────────────────────────────────────────────────────
  ScoreEngine.swift reads priors.json
  score = clip(b + Σ wᵢ·zᵢ,  −3, +3) × (100/3)
  zᵢ   = clip((xᵢ − μᵢ) / σᵢ, −3, +3)   ← population μ/σ
  ↓
Personal calibration (after ~7 days of HealthKit data)  ────────────────────
  μᵢ / σᵢ  replaced by user's own calm-window baseline (RunningStats)
  ↓
Online adaptation (Huber SGD)  ─────────────────────────────────────────────
  User taps 👍 Calm / 👎 Stressed  → recordFeedback()
  One Huber SGD step updates w, b  (δ=1.0, lr=0.05, λ=0.01)
  AdaptedModel persisted in UserDefaults
```

---

## Score Convention

- **Negative score** → stressed (red ring)
- **Positive score** → calm (green ring)
- **0** = neutral boundary
- Range: **−100 … +100**

Formula: `score = clip(b + Σ wᵢ·zᵢ, −3, +3) × (100/3)`

---

## Features (6)

| Name | Description | Window |
|---|---|---|
| `HR_mean_30` | Mean heart rate (bpm) | 30 min |
| `HR_std_30` | Std dev of HR (bpm) | 30 min |
| `HR_slope_30` | Linear slope of HR (bpm/min) | 30 min |
| `HRV_30` | RMSSD of RR intervals (ms) | 30 min |
| `HR_mean_5` | Mean heart rate (bpm) | 5 min |
| `HRV_5` | RMSSD of RR intervals (ms) | 5 min |

LifeSnaps has hourly HR only (no IBI), so `HRV_30` and `HRV_5` are `NaN` for those samples and imputed with the EMO+WESAD training median.

---

## Datasets

| Dataset | Sensor | Subjects | Labels | Split |
|---|---|---|---|---|
| **K-EmoPhone (EMO)** | Samsung wearable — 1 Hz HR, RR intervals | 77 | ESM `stress > 0` → stressed | 50 train / 7 val / 20 test |
| **WESAD** | Empatica E4 — 1 Hz HR, IBI | 15 | `label==2` (TSST) → stressed; `label==1` (baseline) → calm | 12 train / 3 test |
| **LifeSnaps** | Fitbit Charge 4 — hourly mean HR | 59 | `TENSE/ANXIOUS=1 OR SAD=1` → stressed | 39 train / 20 test |
| **Wearable Exam-Stress** | Empatica E4 — 1 Hz HR | 10 students × 3 sessions | In-exam window → stressed | 27 train / 3 test |

**Total (typical run):** ~16,700 training windows, ~4,000 test windows, 128 train / 46 test subjects.

### Expected folder layout

```
WatchStress/
├── emo/
│   ├── P01/ … P80/          # each has HR.csv, RRI.csv
│   └── SubjData/
│       └── EsmResponse.csv
├── WESAD/
│   └── S2/ … S17/           # each has S2.pkl, S2_E4_Data/HR.csv, IBI.csv
├── lifesnaps/
│   └── csv_rais_anonymized/
│       └── hourly_fitbit_sema_df_unprocessed.csv
└── wearable/
    └── Data/
        └── S1/ … S10/       # each has Midterm1/, Midterm2/, Final/ subfolders
```

---

## Training Pipeline (`app_accuracy.py`)

```bash
# EMO only (baseline)
python app_accuracy.py

# All datasets (recommended)
python app_accuracy.py \
    --wesad_root ./WESAD \
    --lifesnaps_root ./lifesnaps \
    --wearable_root ./wearable

# Write priors.json directly into the app bundle
python app_accuracy.py \
    --wesad_root ./WESAD \
    --lifesnaps_root ./lifesnaps \
    --wearable_root ./wearable \
    --out WatchStress/priors.json
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--emo_root` | `./emo` | K-EmoPhone data directory |
| `--wesad_root` | *(off)* | WESAD directory; omit to skip |
| `--lifesnaps_root` | *(off)* | LifeSnaps directory; omit to skip |
| `--wearable_root` | *(off)* | Wearable exam-stress directory; omit to skip |
| `--n_train` | `50` | EMO subjects for training |
| `--n_val` | `7` | EMO subjects for validation |
| `--n_test` | `20` | EMO subjects for test |
| `--l2` | `1.0` | L2 regularisation strength |
| `--huber_warmup` | `20` | Feedback samples before measuring Huber accuracy |
| `--out` | `priors.json` | Output path for priors.json |

### Pipeline parts

| Part | Description |
|---|---|
| **PART 1** | Per-person z-scored logistic regression; writes `priors.json` |
| **PART 2** | Comparison: SVM (RBF), Random Forest (300 trees), XGBoost, MLP |
| **PART 3** | Per-subject accuracy breakdown (RF) on the combined test set |
| **PART 4** | 7-method feature importance (logistic weight, Cohen's d, Mann-Whitney, mutual info, within-subject Pearson r, permutation importance, SHAP) |
| **PART 5** | Online Huber SGD simulation — 4 conditions A/B/C/D |

---

## Accuracy (all 4 datasets — 46 test subjects)

### Offline model comparison

| Model | Test accuracy |
|---|---|
| Majority-class baseline | 68.5% |
| Logistic (app model, personal z-scores) | 56.9% |
| SVM RBF | 69.0% |
| **Random Forest 300T** | **69.9%** |

### Personalisation analysis (PART 5)

Stress classification is **not viable at the population level** — it requires adapting to each individual's physiological baseline. Four conditions are simulated using predict-then-learn on all 46 test subjects:

| Condition | Description | Micro | Macro |
|---|---|---|---|
| **A** | Population priors, no adaptation (cold-start) | 53.7% | 48.2% |
| **B** | Personal priors, no adaptation | 56.9% | 54.4% |
| **C** | Personal priors + Huber SGD (≥20 taps) | **83.8%** | **70.4%** |
| **D** | RF cold-start → Huber SGD (honest end-to-end) | 73.0% | 59.3% |
| **D′** | D — Huber phase only (post warm-up) | 82.9% | 70.1% |

> *Micro = sample-weighted. Macro = subject-weighted (each person counts equally — more honest summary).*

| Lift | Micro | Macro |
|---|---|---|
| Personalisation (A→B) | +3.2% | +6.2% |
| Huber adaptation (B→C) | +26.9% | +16.0% |
| **Full system (A→C)** | **+30.1%** | **+22.2%** |
| C vs best static model (RF) | +13.9% | +0.5% |

**Condition D** is the honest end-to-end story: the app starts using a Random Forest (61.5%) during the first 20 interactions, then switches to Huber-adapted distilled logistic weights. Post warm-up (D′), it reaches **82.9% micro / 70.1% macro** — beating the static RF oracle after just 20 user taps.

#### Per-dataset breakdown

| Dataset | N subj | A (pop) | B (personal) | C (Huber) | B→C lift |
|---|---|---|---|---|---|
| EMO (K-EmoPhone) | 20 | 43.0% | 53.9% | 65.9% | +12.1% |
| WESAD | 3 | 53.0% | 71.4% | 84.3% | +12.8% |
| Wearable (Exam) | 3 | 65.2% | 62.5% | **98.9%** | +36.4% |
| LifeSnaps | 20 | 49.1% | 50.3% | 71.7% | +21.4% |

#### Accuracy vs feedback count (lr=0.05, macro)

| Feedback taps | Micro | Macro |
|---|---|---|
| 0 (personal priors only) | 79.5% | 69.2% |
| 10 | 81.5% | 70.1% |
| 20 | 83.8% | 70.4% |
| 30 | 86.1% | 71.6% |
| 50 | 91.6% | 74.8% |

---

## `priors.json` format

```json
{
  "meta": { "source": "K-EmoPhone + WESAD + LifeSnaps + Wearable", ... },
  "priors": {
    "HR_mean_30": { "mean": 99.4, "std": 22.1 },
    ...
  },
  "weights": { "HR_mean_30": -0.053, ... },
  "weights_display": { "HR_mean_30": -10.3, ... },
  "bias": 0.17
}
```

- `priors` — population calm baseline (cold-start fallback in app)
- `weights` — raw logistic coefficients used in `ScoreEngine.swift` (negative = ↑ stress)
- `weights_display` — scaled to ±100 for readability
- `bias` — always active; threshold is always 0

---

## iOS / watchOS App

### Swift targets

| File | Role |
|---|---|
| `ScoreEngine.swift` | Model inference, personal baseline tracking (RunningStats), Huber online learning |
| `VitalsViewModel.swift` | HealthKit queries — 30-min and 5-min windowed HR/HRV |
| `iOSRootView.swift` | Main iPhone UI — stress ring, feedback prompt |
| `WatchRootView.swift` | Apple Watch UI — mini ring, 👍/👎 feedback buttons |
| `HealthKitManager.swift` | HK authorization and sample fetching helpers |
| `StressLogScore.swift` | Persistent JSONL log of scored entries |
| `SettingsView.swift` | Scheduled auto-logging, debug overrides |
| `AIAssistantView.swift` | On-device AI assistant (Apple Intelligence / FoundationModels) |

### Personalization tiers

1. **Cold-start** — population priors from `priors.json`; uses trained bias
2. **Warm** — personal μ/σ from `RunningStats` (≥10 calm samples, ~7 days)
3. **Adapted** — Huber SGD steps from user 👍/👎 feedback; `AdaptedModel` persisted in `UserDefaults`

### Feedback loop

Each scored window prompts a 👍 / 👎 tap:
- **👍 Calm** → `recordFeedback(wasStressed: false)` → Huber target `+1.0`
- **👎 Stressed** → `recordFeedback(wasStressed: true)` → Huber target `−1.0`

One Huber SGD step updates `AdaptedModel.weights` and `.bias` (δ=1.0, lr=0.05, weight decay λ=0.01). Adaptations persist across app restarts and sync between iPhone and Apple Watch.

To reset: `scoreEngine.resetAdaptedModel()`.

---

## Setup

### Python environment

```bash
conda env create -f WatchStress/environment.yml
conda activate watchstress
pip install shap xgboost   # optional: PART 4 SHAP + PART 2 XGBoost
```

### Xcode

1. Open `WatchStress.xcodeproj`
2. Set your development team in **Signing & Capabilities**
3. Ensure `priors.json` is listed under **Copy Bundle Resources** for both iOS and Watch targets
4. Build and run on a real device (HealthKit requires physical hardware)

---

## Version History

| Version | Notes |
|---|---|
| **1.0.2** | Wearable Exam-Stress dataset added; Condition D (RF cold-start → Huber SGD); Info.plist fixes for App Store validation |
| 1.0.1 | All-dataset support (EMO + WESAD + LifeSnaps); micro/macro accuracy reporting |
| 1.0.0 | Initial release — EMO-only training, Huber online learning, HealthKit integration |
