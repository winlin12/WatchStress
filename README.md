# WatchStress# WatchStress



An Apple Watch + iPhone app that estimates stress in real-time from wrist physiological signals (heart rate and HRV), trained on three labelled datasets and personalized per-user through online Huber learning.## Overview

WatchStress is a SwiftUI iOS/watchOS companion app plus a small Python toolkit for deriving WESAD-based priors. The app reads HealthKit vitals (heart rate, HRV SDNN, respiration rate), converts them into a compact feature vector, and computes a wellness/stress score on a 0–100 ring (higher = calmer). It also logs snapshots, collects self-reports, and adapts to each user over time.

---

## Intended Function (current direction)

## Overview- Load WESAD-derived features (via `wesad.py`) and produce a feature vector for stress detection.

- Compile an app that outputs a wellness score from 0–100 (more stress → lower score).

```- Use the user’s prior HealthKit history as a cold-start baseline for the feature vector.

Offline training  ─────────────────────────────────────────────────────────- Apply a Huber-regression style update loop to tune weights based on self-reported 0–100 scores over days/weeks.

  EMO (K-EmoPhone) + WESAD + LifeSnaps

  ↓  app_accuracy.py## How the App Works

  priors.json  (population μ/σ, logistic weights w, bias b)### Core scoring pipeline

  ↓1. **HealthKit ingestion**: `VitalsViewModel` queries HealthKit via `HealthKitManager` for HR, HRV SDNN, and respiration rate.

App cold-start  ────────────────────────────────────────────────────────────2. **Feature vector**: `ScoreEngine.FeatureSample` is built from the latest vitals (or debug overrides).

  ScoreEngine.swift reads priors.json3. **WESAD priors + weights**: `ScoreEngine` loads `priors.json`, applies z-scoring, and computes a stress index.

  score = clip(b + Σ wᵢ·zᵢ,  −3, +3)4. **Ring score**: The stress index is mapped to a 0–100 score with asymmetric slopes and smoothing.

  zᵢ   = clip((xᵢ − μᵢ) / σᵢ, −3, +3)   ← population μ/σ5. **Confidence + driver breakdown**: `ScoreEngine` surfaces confidence and per-feature contributions for transparency.

  ↓

Personal calibration (after ~7 days of HealthKit data)  ────────────────────### Personalization and self-report feedback

  μᵢ / σᵢ  replaced by user's own calm-window baseline (RunningStats)- **Self-reports** are captured in `SelfReportStore` and used to update `PersonalCalibrationModel`.

  score 0  = exactly your personal average  (bias zeroed when personalized)- **Huber-style updates** with L2 regularization gently shift weights toward the user over time.

  ↓- **Blended score**: `iOSRootView` blends global score with personal calibration via `blendAlpha()`.

Online adaptation (Huber SGD)  ─────────────────────────────────────────────

  User taps 👍 Calm / 👎 Stressed  → recordFeedback()### Logging and scheduling

  One Huber SGD step updates w, b  (δ=1.0, lr=0.05, λ=0.01)- **Stress snapshots** are stored in `StressLogStore` (JSONL + CSV export).

  AdaptedModel persisted in UserDefaults- **Scheduled logging** runs via `BackgroundLoggingManager` and user-configurable time slots.

```- **Daily check-ins** and self-report prompts keep the model anchored to real-world labels.



---## Python Tooling (WESAD / Training)

- **`wesad.py`** extracts HR/HRV/respiration windows from the WESAD dataset and computes baseline priors + weights.

## Score Formula- **`priors.json`** stores the WESAD-derived feature priors and linear weights used by `ScoreEngine`.

- **`continual_training.py`** watches an incoming folder for labeled JSON windows, updates priors, and promotes models based on validation accuracy.

$$z_i = \text{clip}\!\left(\frac{x_i - \mu_i}{\sigma_i},\ -3,\ +3\right)$$- **`environment.yml`** defines a minimal Conda environment for the Python scripts.



$$\text{score} = \text{clip}\!\left(b + \sum_i w_i z_i,\ -3,\ +3\right) \times \frac{100}{3}$$## Key Files

- `WatchStress/WatchStressApp.swift` — app entry point and background logging scheduler.

- **Positive** → stressed, **Negative** → calm, **0** = your personal normal- `WatchStress/iOSRootView.swift` — main UI, score computation, self-report flow.

- Displayed as **−100 … +100**- `WatchStress/ScoreEngine.swift` — WESAD-based scoring logic and personalization hooks.

- `WatchStress/PersonalCalibrationModel.swift` — Huber regression updates and blending.

---- `WatchStress/VitalsViewModel.swift` — HealthKit integration + baseline aggregation.

- `WatchStress/StressLogScore.swift` — JSONL logging + CSV export.

## Features (6)- `WatchStress/SettingsView.swift` — debug tools, scheduled logging, data views.

- `WatchStress/CalibrationStatsView.swift` — summary metrics for accuracy calibration.

| Name | Description | Window |- `WatchStress/continual_training.py` — continual training loop for incoming labels.

|---|---|---|- `WatchStress/wesad.py` — WESAD feature extraction and priors generation.

| `HR_mean_30` | Mean heart rate (bpm) | 30 min |- `WatchStress/priors.json` — current WESAD priors + weights used by the app.

| `HR_std_30` | Std dev of HR (bpm) | 30 min |

| `HR_slope_30` | Linear slope of HR (bpm/min) | 30 min |## Data & Assets

| `HRV_30` | RMSSD of RR intervals (ms) | 30 min |The `Assets.xcassets` bundle includes icons, PDFs, and dataset archives (WESAD/HRV/SWELL). These are referenced for research and asset packaging, not for runtime code execution.

| `HR_mean_5` | Mean heart rate (bpm) | 5 min |

| `HRV_5` | RMSSD of RR intervals (ms) | 5 min |## Notes

- HealthKit permissions are required for heart rate, HRV SDNN, and respiration rate.

LifeSnaps has hourly HR only (no IBI), so `HRV_30` and `HRV_5` are `NaN` for those samples and imputed with the EMO+WESAD training median.- The app supports debug overrides to simulate stressed/relaxed states when HealthKit data is missing.

- The AI assistant view uses on-device Apple Intelligence via `FoundationModels` when available.

---

## Datasets

| Dataset | Sensor | Subjects | Labels | Used for |
|---|---|---|---|---|
| **K-EmoPhone (EMO)** | Samsung wearable — 1 Hz HR, RR intervals | 77 | ESM `stress > 0` → stressed | 50 train / 7 val / 20 test |
| **WESAD** | Empatica E4 — 1 Hz HR, IBI | 15 | `label==2` (TSST) → stressed; `label==1` (baseline) → calm | All merged into training |
| **LifeSnaps** | Fitbit Charge 4 — hourly mean HR | 59 with HR+EMA | `TENSE/ANXIOUS=1 OR SAD=1` → stressed; `RESTED/RELAXED=1 AND !TENSE AND !SAD` → calm | 39 train / 20 test (top-N by sample count) |

**Total training windows (typical run):** ~3,500 from 124 subjects  
**Total test windows:** ~2,300 from 40 subjects (20 EMO + 20 LifeSnaps)

### Expected folder layout

```
WatchStress/
├── emo/
│   ├── P01/ … P80/          # each has HR.csv, RRI.csv
│   └── SubjData/
│       └── EsmResponse.csv
├── WESAD/
│   ├── S2/ … S17/           # each has S2.pkl, S2_E4_Data/HR.csv, IBI.csv
│   └── wesad_readme.pdf
└── lifesnaps/
    └── csv_rais_anonymized/
        └── hourly_fitbit_sema_df_unprocessed.csv
```

---

## Training Pipeline (`app_accuracy.py`)

```bash
# EMO only (baseline)
python app_accuracy.py

# EMO + WESAD
python app_accuracy.py --wesad_root ./WESAD

# EMO + WESAD + LifeSnaps  (recommended)
python app_accuracy.py --wesad_root ./WESAD --lifesnaps_root ./lifesnaps

# Write priors.json to app bundle path
python app_accuracy.py --wesad_root ./WESAD --lifesnaps_root ./lifesnaps \
    --out WatchStress/priors.json

# Compare a previously deployed model
python app_accuracy.py --wesad_root ./WESAD --lifesnaps_root ./lifesnaps \
    --priors old_priors.json
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--emo_root` | `./emo` | K-EmoPhone data directory |
| `--wesad_root` | *(off)* | WESAD directory; omit to skip |
| `--lifesnaps_root` | *(off)* | LifeSnaps directory; omit to skip |
| `--lifesnaps_n_test` | `20` | LifeSnaps subjects reserved for test |
| `--n_train` | `50` | EMO subjects for training |
| `--n_val` | `7` | EMO subjects for validation |
| `--n_test` | `20` | EMO subjects for test |
| `--l2` | `1.0` | L2 regularisation strength |
| `--epochs` | `2000` | PyTorch training epochs |
| `--lr` | `0.05` | Adam learning rate |
| `--no_gpu` | *(off)* | Force CPU even if CUDA available |
| `--out` | `priors.json` | Output path for priors.json |

### Pipeline parts

| Part | Description |
|---|---|
| **PART 1** | Per-person z-scored logistic regression; writes `priors.json` |
| **PART 2** | Comparison: SVM (RBF), Random Forest (300 trees), XGBoost, MLP |
| **PART 3** | Per-subject accuracy breakdown on the combined test set |
| **PART 4** | 7-method feature importance analysis (logistic weight, Cohen's d, Mann-Whitney, mutual information, within-subject Pearson r, permutation importance, SHAP) |
| **PART 5** | Online Huber SGD simulation — mirrors in-app `recordFeedback()` |

### Personalization during training

Training uses **per-person z-scoring**: each sample row is normalized by that subject's own calm-window μ/σ (not the population). Subjects with fewer than 3 calm samples fall back to population priors. This means the learned weights directly transfer to the app's runtime normalization.

---

## `priors.json` format

```json
{
  "meta": { "source": "K-EmoPhone (EMO) + WESAD + LifeSnaps", ... },
  "priors": {
    "HR_mean_30": { "mean": 75.8, "std": 7.0 },
    ...
  },
  "weights": { "HR_mean_30": 0.123, ... },
  "weights_display": { "HR_mean_30": 45.2, ... },
  "bias": -0.04
}
```

`priors` = population calm baseline (cold-start fallback in app).  
`weights` = raw logistic coefficients (used in `ScoreEngine.swift`).  
`weights_display` = linearly scaled to ±100 for readability.

After retraining, copy `priors.json` into `WatchStress/` and add it to **Copy Bundle Resources** in Xcode, or update the embedded fallback string in `ScoreEngine.swift`.

---

## iOS / watchOS App

### Swift targets

| File | Role |
|---|---|
| `ScoreEngine.swift` | Model inference, personal baseline tracking (RunningStats), Huber online learning |
| `VitalsViewModel.swift` | HealthKit queries — 30-min and 5-min windowed HR/HRV |
| `iOSRootView.swift` | Main iPhone UI — stress ring, feedback prompt, schedule logging |
| `WatchRootView.swift` | Apple Watch UI — mini ring, 👍/👎 feedback buttons |
| `HealthKitManager.swift` | HK authorization and sample fetching helpers |
| `StressLogScore.swift` | Persistent log of scored entries |
| `SettingsView.swift` | Scheduled auto-logging, debug overrides |
| `AIAssistantView.swift` | On-device AI assistant integration |

### Personalization tiers

1. **Cold-start** — population priors from `priors.json`; bias active
2. **Warm** — personal μ/σ from `RunningStats` (≥10 samples per feature, ~7 days); bias zeroed so score 0 = your own average
3. **Adapted** — Huber SGD steps applied from user feedback; `AdaptedModel` replaces offline weights in `UserDefaults`

### Feedback loop

Each time a score is displayed, a 👍 / 👎 prompt appears:
- **👍 Calm** → `recordFeedback(sample:wasStressed:false)`
- **👎 Stressed** → `recordFeedback(sample:wasStressed:true)`

One Huber SGD step updates `AdaptedModel.weights` and `.bias` (δ=1.0, lr=0.05, weight decay λ=0.01). Adaptations persist across app restarts via `UserDefaults` and are shared between iPhone and Apple Watch.

To reset adaptations: `scoreEngine.resetAdaptedModel()`.

---

## Setup

### Python environment

```bash
conda env create -f WatchStress/environment.yml
conda activate watchstress
pip install shap xgboost   # optional, for PART 4 SHAP + PART 2 XGBoost
```

### Xcode

1. Open `WatchStress.xcodeproj`
2. Set your development team in **Signing & Capabilities**
3. Ensure `priors.json` is listed under **Copy Bundle Resources** for both the iOS and Watch targets
4. Build and run on a real device (HealthKit requires physical hardware)

---

## Accuracy (typical run — EMO + WESAD + LifeSnaps)

### Offline model comparison (test set: 40 subjects, 20 EMO + 20 LifeSnaps)

| Model | Test accuracy | Notes |
|---|---|---|
| Logistic (app model) | ~51% | Per-person z-scores, population weights |
| SVM RBF | ~62% | Population z-scores |
| Random Forest 300T | ~62% | Population z-scores |
| XGBoost | ~61% | Population z-scores |
| MLP | ~57% | Population z-scores |
| Majority-class baseline | ~64% | Predict "calm" always |

### Personalization necessity analysis (PART 5)

This is the core finding: **stress classification is not a viable population-level problem — it requires adapting to an individual's personal physiological baseline.**

Three conditions are simulated using predict-then-learn on each test subject:

| Condition | Description | Accuracy |
|---|---|---|
| **A** | Population priors, no adaptation (app cold-start) | ~47% |
| **B** | Personal priors, no adaptation (baseline calibration only) | ~51% |
| **C** | Personal priors + Huber SGD (30 feedback samples) | **~70%** |

| Lift | Value |
|---|---|
| Personalisation (A→B) | +4% |
| Huber adaptation (B→C) | +19% |
| **Full system (A→C)** | **+23%** |

**The population model (A) performs below majority-class baseline** (~47% vs ~64%). Once the app learns a user's calm baseline (B), it crosses the chance line. The Huber feedback loop (C) pushes accuracy to **~70%**, a 23 percentage-point lift, and reaches that level after only ~30 labelled feedback taps.

#### Per-dataset breakdown

| Dataset | N subjects | A (pop) | B (personal) | C (Huber) | A→B | B→C |
|---|---|---|---|---|---|---|
| EMO (K-EmoPhone) | 20 | ~50% | ~54% | ~68% | +4% | +14% |
| LifeSnaps | 20 | ~44% | ~48% | ~72% | +4% | +24% |

LifeSnaps benefits even more from Huber adaptation (+24% B→C vs +14% for EMO), likely because the Fitbit's hourly HR granularity means the personal baseline needs more weight adaptation to compensate for the missing HRV signal.

#### Accuracy vs feedback count

| Feedback samples received | Accuracy (lr=0.05) |
|---|---|
| 0 (cold-start with personal priors) | ~66% |
| 10 | ~67% |
| 20 | ~69% |
| 30 | **~70%** |
| 50 | ~68% (mild overfit for some subjects) |
