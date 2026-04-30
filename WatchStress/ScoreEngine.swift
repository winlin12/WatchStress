//
//  ScoreEngine.swift
//  WatchStress
//
//  score = clip(b + Σ w_i · z_i, -3, +3)
//  where z_i = clip((x_i - μ_i) / σ_i, -3, +3)  using per-person priors once accumulated
//  Negative score = stressed, positive = calm, 0 = neutral
//  (scaled to -100…+100 for display: -100 = fully stressed/red, +100 = fully calm/green)
//
//  Features (must match FEATURE_NAMES in app_accuracy.py):
//    HR_mean_30  — mean heart rate over last 30 min (bpm)
//    HR_std_30   — std dev of HR over last 30 min (bpm)
//    HR_slope_30 — linear slope of HR over last 30 min (bpm/min)
//    HRV_30      — HRV SDNN over last 30 min (ms), most recent sample in window
//    HR_mean_5   — mean HR over last 5 min (bpm)
//    HRV_5       — HRV SDNN over last 5 min (ms), most recent sample in window
//
//  Weights and priors are loaded from priors.json (Copy Bundle Resources).
//  If the bundle resource is missing, the embedded fallback below is used.
//

import Foundation

// MARK: - Embedded fallback priors (matches WatchStress/priors.json)
// Re-run emo_train.py and update this string whenever the model is retrained.
private let embeddedPriorsJSON = """
{
  "priors": {
    "HR_mean_30":  { "mean": 70.0, "std": 12.0 },
    "HR_std_30":   { "mean": 8.0,  "std": 4.0  },
    "HR_slope_30": { "mean": 0.0,  "std": 2.0  },
    "HRV_30":      { "mean": 45.0, "std": 20.0 },
    "HR_mean_5":   { "mean": 72.0, "std": 15.0 },
    "HRV_5":       { "mean": 45.0, "std": 20.0 }
  },
  "weights": {
    "HR_mean_30":  -0.30,
    "HR_std_30":   -0.20,
    "HR_slope_30": -0.20,
    "HRV_30":       0.40,
    "HR_mean_5":   -0.50,
    "HRV_5":        0.30
  },
  "bias": 0.0
}
"""

final class ScoreEngine {

    // MARK: - Model feature space (matches priors.json keys)

    /// These raw-value strings MUST match the keys written by emo_train.py into priors.json.
    enum Feature: String, CaseIterable, Codable {
        case HR_mean_30   // mean HR over last 30 min (bpm)
        case HR_std_30    // std dev of HR over last 30 min (bpm)
        case HR_slope_30  // linear slope of HR over last 30 min (bpm/min)
        case HRV_30       // HRV SDNN in last 30 min (ms)
        case HR_mean_5    // mean HR over last 5 min (bpm)
        case HRV_5        // HRV SDNN in last 5 min (ms)
    }

    struct FeatureSample {
        var HR_mean_30:  Double?   // mean bpm, 30 min window
        var HR_std_30:   Double?   // std dev bpm, 30 min window
        var HR_slope_30: Double?   // slope bpm/min, 30 min window
        var HRV_30:      Double?   // HRV SDNN ms, 30 min window
        var HR_mean_5:   Double?   // mean bpm, 5 min window
        var HRV_5:       Double?   // HRV SDNN ms, 5 min window

        func value(for f: Feature) -> Double? {
            switch f {
            case .HR_mean_30:  return HR_mean_30
            case .HR_std_30:   return HR_std_30
            case .HR_slope_30: return HR_slope_30
            case .HRV_30:      return HRV_30
            case .HR_mean_5:   return HR_mean_5
            case .HRV_5:       return HRV_5
            }
        }
    }

    struct DriverContribution {
        let feature: Feature
        /// Model weight w_i (stress-direction)
        let weight: Double
        /// z-score deviation (x-mu)/sigma
        let z: Double
        /// Change in final score from this feature alone (same score mapping as total)
        let scoreDelta: Double
        /// Raw feature value x
        let value: Double
        /// Blended baseline mean (mu)
        let mean: Double
        /// Blended baseline std (sigma)
        let std: Double
        /// Blend factor a (0..1) used for personalization
        let blendA: Double
    }

    enum Confidence: String {
        case low = "Low"
        case medium = "Medium"
        case high = "High"
    }

    struct ScoreResult {
        /// Stress score in −100…+100. Negative = stressed (red), positive = calm (green), 0 = neutral.
        let score: Double
        /// Raw linear output: b + Σ w_i · z_i (before clamping)
        let linearOutput: Double
        /// Number of features that contributed to this score
        let featuresUsed: Int
        let confidence: Confidence
        let bias: Double
        let drivers: [DriverContribution]   // sorted by absolute contribution
    }

    struct BaselineStats {
        let count: Int
        let mean: Double
        let stdDev: Double
    }

    // MARK: - priors.json schema

    struct PriorsFile: Codable {
        struct Meta: Codable {
            let source: String?
            let window_s: Double?
            let stride_s: Double?
            let notes: String?
            let labels: [String: Int]?
        }
        struct Prior: Codable {
            let mean: Double
            let std: Double
        }

        let meta: Meta?
        let priors: [String: Prior]     // keys like "hrMeanBPM"
        let weights: [String: Double]   // keys like "hrMeanBPM"
        let bias: Double
    }

    // MARK: - Configuration

    private let eps: Double = 1e-6
    private let zClip: Double = 3.0

    /// Minimum number of stored HealthKit samples before a feature's
    /// personal baseline is trusted over the population prior.
    private let minUserSamples: Int = 10

    // MARK: - Huber online learning hyperparameters (mirror app_accuracy.py PART 5)
    /// Huber loss delta: quadratic for |residual| ≤ delta, linear beyond.
    private let huberDelta:   Double = 1.0
    /// SGD learning rate for weight updates.
    private let adaptLR:      Double = 0.05
    /// L2 weight decay — pulls w back toward population values each step.
    private let weightDecay:  Double = 0.01

    // MARK: - Adapted model (Huber SGD state, persisted in UserDefaults)

    /// Mutable model weights that are updated via `recordFeedback`.
    /// Starts as nil (falls back to priorsFile weights) and is initialised
    /// on the first feedback call.
    struct AdaptedModel: Codable {
        var weights: [String: Double]
        var bias:    Double
        var feedbackCount: Int = 0
    }

    private static let adaptedModelKey = "ScoreEngine.adaptedModel"

    static func loadAdaptedModel() -> AdaptedModel? {
        guard let data = UserDefaults.standard.data(forKey: adaptedModelKey),
              let m = try? JSONDecoder().decode(AdaptedModel.self, from: data) else { return nil }
        return m
    }

    private static func saveAdaptedModel(_ m: AdaptedModel) {
        if let data = try? JSONEncoder().encode(m) {
            UserDefaults.standard.set(data, forKey: adaptedModelKey)
        }
    }

    /// Reset all Huber-adapted weights back to the offline-trained population values.
    func resetAdaptedModel() {
        UserDefaults.standard.removeObject(forKey: Self.adaptedModelKey)
    }

    /// Number of user feedback events that have been incorporated so far.
    var feedbackCount: Int { Self.loadAdaptedModel()?.feedbackCount ?? 0 }

    // MARK: - State

    private let priorsFile: PriorsFile

    // MARK: - Hardcoded fallback (updated by emo_train.py)

    /// Pure-Swift fallback — no JSON parsing, never fails.
    static let hardcodedPriorsFile: PriorsFile = {
        return PriorsFile(
            meta: nil,
            priors: [
                "HR_mean_30":  PriorsFile.Prior(mean: 70.0, std: 12.0),
                "HR_std_30":   PriorsFile.Prior(mean: 8.0,  std: 4.0),
                "HR_slope_30": PriorsFile.Prior(mean: 0.0,  std: 2.0),
                "HRV_30":      PriorsFile.Prior(mean: 45.0, std: 20.0),
                "HR_mean_5":   PriorsFile.Prior(mean: 72.0, std: 15.0),
                "HRV_5":       PriorsFile.Prior(mean: 45.0, std: 20.0),
            ],
            weights: [
                "HR_mean_30":  -0.30,
                "HR_std_30":   -0.20,
                "HR_slope_30": -0.20,
                "HRV_30":       0.40,
                "HR_mean_5":   -0.50,
                "HRV_5":        0.30,
            ],
            bias: 0.0
        )
    }()

    // MARK: - Init

    /// Load priors.json from app bundle; falls back to embedded JSON, then hardcoded values.
    /// Never returns nil — a valid ScoreEngine is always created.
    init(priorsResourceName: String = "priors") {
        // 1. Try bundle resource
        if let url = Bundle.main.url(forResource: priorsResourceName, withExtension: "json"),
           let data = try? Data(contentsOf: url),
           let decoded = try? JSONDecoder().decode(PriorsFile.self, from: data) {
            priorsFile = decoded
            return
        }
        // 2. Try embedded JSON string constant
        if let data = embeddedPriorsJSON.data(using: .utf8),
           let decoded = try? JSONDecoder().decode(PriorsFile.self, from: data) {
            priorsFile = decoded
            return
        }
        // 3. Absolute fallback: hardcoded Swift values — never fails
        priorsFile = Self.hardcodedPriorsFile
    }

    init(priorsFile: PriorsFile) {
        self.priorsFile = priorsFile
    }

    // MARK: - Public API

    /// Optional: expose the learned vector (feature order + weights).
    func modelVector() -> (features: [Feature], weights: [Double], bias: Double) {
        let feats = Feature.allCases
        let ws = feats.map { priorsFile.weights[$0.rawValue] ?? 0.0 }
        return (feats, ws, priorsFile.bias)
    }

    /// Compute score:
    ///   1. mu/sigma = user's personal baseline if enough data; else population prior.
    ///      Personal normalization reduces person-to-person variance but does NOT
    ///      force score 0 at the user's average — the model bias drives the threshold.
    ///   2. z_i    = clip((x_i − mu_i) / sigma_i,  −3, +3)
    ///   3. linear = bias + Σ w_i · z_i
    ///      The bias is always the model's trained value (from adapted model or priors.json).
    ///      Negative output = stressed, positive output = calm. Threshold at 0.
    ///   4. score  = clip(linear, −3, +3) × (100/3)  →  −100…+100
    func computeScore(sample: FeatureSample) -> ScoreResult {
        let userStats    = Self.loadAllUserStats()
        let adaptedModel = Self.loadAdaptedModel()

        // Always use the model's trained bias — never zero it.
        // The sign of the score is the only classifier (negative = stressed, positive = calm).
        let activeBias: Double
        if let adapted = adaptedModel {
            activeBias = adapted.bias
        } else {
            activeBias = priorsFile.bias
        }
        var linear = activeBias

        var rawContributions: [(
            feature: Feature, weight: Double, z: Double,
            value: Double, mean: Double, std: Double, blendA: Double
        )] = []

        for f in Feature.allCases {
            guard let x = sample.value(for: f) else { continue }

            // Prefer adapted weight; fall back to priors.json weight
            let w: Double
            if let adapted = adaptedModel, let aw = adapted.weights[f.rawValue] {
                w = aw
            } else if let pw = priorsFile.weights[f.rawValue] {
                w = pw
            } else {
                continue
            }

            let mu:     Double
            let sigma:  Double
            let blendA: Double

            if let us = userStats[f], us.count >= minUserSamples {
                mu     = us.mean
                sigma  = max(us.stdDev, eps)
                blendA = 1.0
            } else if let prior = priorsFile.priors[f.rawValue] {
                mu     = prior.mean
                sigma  = prior.std + eps
                blendA = 0.0
            } else {
                continue
            }

            let zRaw = (x - mu) / sigma
            let z    = clamp(zRaw, lo: -zClip, hi: zClip)
            linear  += w * z

            rawContributions.append((f, w, z, x, mu, sigma, blendA))
        }

        let rawScore = clamp(linear, lo: -zClip, hi: zClip)
        let scale    = 100.0 / zClip          // maps −3…+3  →  −100…+100
        let score    = rawScore * scale
        let used     = rawContributions.count
        let conf: Confidence = used >= 5 ? .high : used >= 3 ? .medium : .low

        let drivers: [DriverContribution] = rawContributions.map { item in
            let linearWithout = linear - item.weight * item.z
            let scoreWithout  = clamp(linearWithout, lo: -zClip, hi: zClip) * scale
            return DriverContribution(
                feature:    item.feature,
                weight:     item.weight,
                z:          item.z,
                scoreDelta: score - scoreWithout,
                value:      item.value,
                mean:       item.mean,
                std:        item.std,
                blendA:     item.blendA
            )
        }.sorted { abs($0.scoreDelta) > abs($1.scoreDelta) }

        return ScoreResult(
            score:        score,
            linearOutput: linear,
            featuresUsed: used,
            confidence:   conf,
            bias:         activeBias,
            drivers:      drivers
        )
    }

    /// True once personal z-score normalization is active (uses user's own mu/sigma).
    /// This does NOT change the decision threshold — negative is still stressed, positive calm.
    var isPersonalized: Bool {
        let userStats = Self.loadAllUserStats()
        let count = Feature.allCases.filter {
            (userStats[$0]?.count ?? 0) >= minUserSamples
        }.count
        return count >= max(1, Feature.allCases.count / 2)
    }

    /// Update rolling per-user baselines (suggested: once per day).
    func updateUserBaselinesIfNeeded(with sample: FeatureSample, today: Date = Date()) {
        let cal = Calendar.current
        let defaults = UserDefaults.standard

        let lastKey = "ScoreEngine.lastUpdateDay"
        let lastDay = defaults.string(forKey: lastKey)
        let currentDay = Self.dayKey(for: today, calendar: cal)
        guard lastDay != currentDay else { return } // already updated today

        var stats = Self.loadAllUserStats()
        for f in Feature.allCases {
            if let x = sample.value(for: f) {
                var st = stats[f] ?? RunningStats()
                st.update(with: x)
                stats[f] = st
            }
        }

        Self.saveAllUserStats(stats)
        defaults.set(currentDay, forKey: lastKey)
    }

    /// Record explicit user feedback ("was I right?") and perform one Huber SGD step.
    ///
    /// - Parameters:
    ///   - sample:      The `FeatureSample` that produced the score being corrected.
    ///   - wasStressed: `true` if the user confirms they were stressed; `false` if calm.
    ///
    /// This mirrors `_simulate_online_learning` in `app_accuracy.py` (PART 5):
    ///   target   = -1 (stressed) or +1 (calm)       ← negative = stressed convention
    ///   residual = score − target
    ///   grad     = residual  if |residual| ≤ δ,  else δ·sign(residual)   (Huber)
    ///   w ← w · (1 − λ·lr) − lr · grad · z        (L2 weight decay)
    ///   b ← b − lr · grad
    func recordFeedback(sample: FeatureSample, wasStressed: Bool) {
        let userStats = Self.loadAllUserStats()

        // Initialise adapted model from priors.json weights on first call
        var adapted = Self.loadAdaptedModel() ?? AdaptedModel(
            weights: Dictionary(
                uniqueKeysWithValues: Feature.allCases.compactMap { f in
                    guard let w = priorsFile.weights[f.rawValue] else { return nil }
                    return (f.rawValue, w)
                }
            ),
            bias: priorsFile.bias,
            feedbackCount: 0
        )

        let target = wasStressed ? -1.0 : 1.0

        // Compute per-feature z-scores (same logic as computeScore)
        var zScores: [Feature: Double] = [:]
        for f in Feature.allCases {
            guard let x = sample.value(for: f) else { continue }
            let mu:    Double
            let sigma: Double
            if let us = userStats[f], us.count >= minUserSamples {
                mu    = us.mean
                sigma = max(us.stdDev, eps)
            } else if let prior = priorsFile.priors[f.rawValue] {
                mu    = prior.mean
                sigma = prior.std + eps
            } else { continue }
            zScores[f] = clamp((x - mu) / sigma, lo: -zClip, hi: zClip)
        }

        // Current prediction with adapted weights
        var linear = adapted.bias
        for f in Feature.allCases {
            if let z = zScores[f], let w = adapted.weights[f.rawValue] {
                linear += w * z
            }
        }
        let score    = clamp(linear, lo: -zClip, hi: zClip)
        let residual = score - target
        let grad     = abs(residual) <= huberDelta
            ? residual
            : huberDelta * (residual > 0 ? 1.0 : -1.0)

        // Huber SGD step with L2 weight decay
        for f in Feature.allCases {
            guard let z = zScores[f] else { continue }
            let w = adapted.weights[f.rawValue] ?? 0.0
            adapted.weights[f.rawValue] = w * (1.0 - weightDecay * adaptLR) - adaptLR * grad * z
        }
        adapted.bias -= adaptLR * grad
        adapted.feedbackCount += 1

        Self.saveAdaptedModel(adapted)
    }



    static func sample(
        HR_mean_30:  Double? = nil,
        HR_std_30:   Double? = nil,
        HR_slope_30: Double? = nil,
        HRV_30:      Double? = nil,
        HR_mean_5:   Double? = nil,
        HRV_5:       Double? = nil
    ) -> FeatureSample {
        FeatureSample(
            HR_mean_30:  HR_mean_30,
            HR_std_30:   HR_std_30,
            HR_slope_30: HR_slope_30,
            HRV_30:      HRV_30,
            HR_mean_5:   HR_mean_5,
            HRV_5:       HRV_5
        )
    }

    // MARK: - Persistence (UserDefaults)

    private struct RunningStats: Codable {
        var count: Int = 0
        var mean: Double = 0
        var m2: Double = 0
        init() {}
        mutating func update(with x: Double) {
            count += 1
            let delta = x - mean
            mean += delta / Double(count)
            m2 += delta * (x - mean)
        }
        /// Population std dev derived from Welford's m2 accumulator.
        var stdDev: Double {
            guard count > 1 else { return 0 }
            return sqrt(m2 / Double(count - 1))
        }
    }

    static func storeUserBaselines(_ baselines: [Feature: BaselineStats]) {
        var stats = loadAllUserStats()
        for (feature, baseline) in baselines {
            var st = RunningStats()
            st.count = baseline.count
            st.mean  = baseline.mean
            // Reconstruct m2 = variance × (n−1) so that stdDev works correctly.
            // variance = stdDev²  →  m2 = stdDev² × max(n−1, 1)
            st.m2 = baseline.stdDev * baseline.stdDev * Double(max(baseline.count - 1, 1))
            stats[feature] = st
        }
        saveAllUserStats(stats)
    }

    private static func key(for feature: Feature) -> String { "ScoreEngine.user.\(feature.rawValue)" }

    private static func loadAllUserStats() -> [Feature: RunningStats] {
        var result: [Feature: RunningStats] = [:]
        let decoder = JSONDecoder()
        for f in Feature.allCases {
            if let data = UserDefaults.standard.data(forKey: key(for: f)),
               let st = try? decoder.decode(RunningStats.self, from: data) {
                result[f] = st
            }
        }
        return result
    }

    private static func saveAllUserStats(_ stats: [Feature: RunningStats]) {
        let encoder = JSONEncoder()
        for (f, st) in stats {
            if let data = try? encoder.encode(st) {
                UserDefaults.standard.set(data, forKey: key(for: f))
            }
        }
    }

    private static func dayKey(for date: Date, calendar: Calendar) -> String {
        let c = calendar.dateComponents([.year, .month, .day], from: date)
        return "\(c.year ?? 0)-\(c.month ?? 0)-\(c.day ?? 0)"
    }

    // MARK: - Utilities

    private func clamp(_ x: Double, lo: Double, hi: Double) -> Double {
        min(hi, max(lo, x))
    }
}
