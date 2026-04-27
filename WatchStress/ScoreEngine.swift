//
//  ScoreEngine.swift
//  WatchStress
//
//  score = clip(b + Σ w_i · z_i, -3, +3)
//  where z_i = clip((x_i - μ_i) / σ_i, -3, +3)  using per-person priors once accumulated
//  Positive score = stressed, negative = calm, 0 = neutral
//  (will be scaled to -100…+100 in a later pass)
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
    "HR_mean_30":  { "mean": 75.8,  "std": 7.0  },
    "HR_std_30":   { "mean": 4.5,   "std": 2.5  },
    "HR_slope_30": { "mean": 0.0,   "std": 0.5  },
    "HRV_30":      { "mean": 45.0,  "std": 18.0 },
    "HR_mean_5":   { "mean": 75.8,  "std": 7.0  },
    "HRV_5":       { "mean": 45.0,  "std": 18.0 }
  },
  "weights": {
    "HR_mean_30":  0.0,
    "HR_std_30":   0.0,
    "HR_slope_30": 0.0,
    "HRV_30":      0.0,
    "HR_mean_5":   0.0,
    "HRV_5":       0.0
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
        /// Stress score in −3…+3. Positive = stressed, negative = calm, 0 = neutral.
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

    // MARK: - State

    private let priorsFile: PriorsFile

    // MARK: - Hardcoded fallback (updated by emo_train.py)

    /// Pure-Swift fallback — no JSON parsing, never fails.
    static let hardcodedPriorsFile: PriorsFile = {
        return PriorsFile(
            meta: nil,
            priors: [
                "HR_mean_30":  PriorsFile.Prior(mean: 75.8, std: 7.0),
                "HR_std_30":   PriorsFile.Prior(mean: 4.5,  std: 2.5),
                "HR_slope_30": PriorsFile.Prior(mean: 0.0,  std: 0.5),
                "HRV_30":      PriorsFile.Prior(mean: 45.0, std: 18.0),
                "HR_mean_5":   PriorsFile.Prior(mean: 75.8, std: 7.0),
                "HRV_5":       PriorsFile.Prior(mean: 45.0, std: 18.0),
            ],
            weights: [
                "HR_mean_30":  0.0,
                "HR_std_30":   0.0,
                "HR_slope_30": 0.0,
                "HRV_30":      0.0,
                "HR_mean_5":   0.0,
                "HRV_5":       0.0,
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
    ///   1. z_i    = clip((x_i - μ_i) / σ_i, -3, +3)
    ///   2. linear = b + Σ w_i · z_i
    ///   3. score  = clip(linear, -3, +3)   (positive = stressed, negative = calm)
    func computeScore(sample: FeatureSample) -> ScoreResult {
        let bias = priorsFile.bias
        var linear = bias
        var rawContributions: [(feature: Feature, weight: Double, z: Double, value: Double, mean: Double, std: Double)] = []

        for f in Feature.allCases {
            guard let x = sample.value(for: f) else { continue }
            guard let w = priorsFile.weights[f.rawValue] else { continue }
            guard let prior = priorsFile.priors[f.rawValue] else { continue }

            let zRaw = (x - prior.mean) / (prior.std + eps)
            let z = clamp(zRaw, lo: -zClip, hi: zClip)
            linear += w * z

            rawContributions.append((feature: f, weight: w, z: z, value: x, mean: prior.mean, std: prior.std))
        }

        let score = clamp(linear, lo: -zClip, hi: zClip)

        // Confidence: based on how many features contributed
        let used = rawContributions.count
        let conf: Confidence = used >= 5 ? .high : used >= 3 ? .medium : .low

        // Per-feature score delta: score(linear) - score(linear - w_i·z_i)
        let drivers: [DriverContribution] = rawContributions.map { item in
            let linearWithout = linear - item.weight * item.z
            let scoreWithout = clamp(linearWithout, lo: -zClip, hi: zClip)
            let delta = score - scoreWithout
            return DriverContribution(
                feature: item.feature,
                weight: item.weight,
                z: item.z,
                scoreDelta: delta,
                value: item.value,
                mean: item.mean,
                std: item.std,
                blendA: 0.0
            )
        }.sorted { abs($0.scoreDelta) > abs($1.scoreDelta) }

        return ScoreResult(
            score: score,
            linearOutput: linear,
            featuresUsed: used,
            confidence: conf,
            bias: bias,
            drivers: drivers
        )
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
    }

    static func storeUserBaselines(_ baselines: [Feature: BaselineStats]) {
        var stats = loadAllUserStats()
        for (feature, baseline) in baselines {
            var st = RunningStats()
            st.count = baseline.count
            st.mean  = baseline.mean
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
