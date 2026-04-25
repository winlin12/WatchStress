//
//  ScoreEngine.swift
//  WatchStress
//
//  score = sigmoid(b + Σ w_i · z_i) × 100
//  where z_i = clip((x_i - μ_i) / σ_i, -3, +3)
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
    "HR":              { "mean": 75.799037,    "std": 7.016697    },
    "HRV":             { "mean": 69.574095,    "std": 11.529691   },
    "skinTemperature": { "mean": 32.475552,    "std": 1.656109    },
    "UltraViolet":     { "mean": 0.025315,     "std": 0.090567    },
    "stepCount":       { "mean": 313.647486,   "std": 560.849122  },
    "Calorie":         { "mean": 31.801333,    "std": 112.468417  },
    "Distance":        { "mean": 23986.717141, "std": 43445.689561 }
  },
  "weights": {
    "HR":              0.087509,
    "HRV":            -0.007194,
    "skinTemperature":-0.077186,
    "UltraViolet":    -0.039582,
    "stepCount":      -0.059210,
    "Calorie":         0.286595,
    "Distance":       -0.209232
  },
  "bias": -0.573793
}
"""

final class ScoreEngine {

    // MARK: - Model feature space (matches priors.json keys)

    /// These raw-value strings MUST match the keys written by emo_train.py into priors.json.
    enum Feature: String, CaseIterable, Codable {
        // Physiological
        case HR                 // heart rate (bpm)
        case HRV                // HRV SDNN (ms)
        case skinTemperature    // wrist skin temp (°C)
        case UltraViolet        // UV index (0=NONE … 4=VERY_HIGH)
        // Activity
        case stepCount          // steps in window
        case Calorie            // active kcal in window
        case Distance           // metres in window
    }

    struct FeatureSample {
        var HR: Double?               // heart rate bpm
        var HRV: Double?              // SDNN ms
        var skinTemperature: Double?  // °C
        var UltraViolet: Double?      // encoded 0-4
        var stepCount: Double?        // steps
        var Calorie: Double?          // active kcal
        var Distance: Double?         // metres

        func value(for f: Feature) -> Double? {
            switch f {
            case .HR:              return HR
            case .HRV:             return HRV
            case .skinTemperature: return skinTemperature
            case .UltraViolet:     return UltraViolet
            case .stepCount:       return stepCount
            case .Calorie:         return Calorie
            case .Distance:        return Distance
            }
        }
    }

    struct DriverContribution {
        let feature: Feature
        /// Model weight w_i (stress-direction)
        let weight: Double
        /// z-score deviation (x-mu)/sigma
        let z: Double
        /// Change in final score from this feature alone (tanh-mapped)
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
        /// Final stress probability mapped to 0–100. Higher = more stressed.
        let score: Double
        /// Raw linear output: b + Σ w_i · z_i
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
        let P = PriorsFile.Prior.self
        return PriorsFile(
            meta: nil,
            priors: [
                "HR":              PriorsFile.Prior(mean: 75.799037,    std: 7.016697),
                "HRV":             PriorsFile.Prior(mean: 69.574095,    std: 11.529691),
                "skinTemperature": PriorsFile.Prior(mean: 32.475552,    std: 1.656109),
                "UltraViolet":     PriorsFile.Prior(mean: 0.025315,     std: 0.090567),
                "stepCount":       PriorsFile.Prior(mean: 313.647486,   std: 560.849122),
                "Calorie":         PriorsFile.Prior(mean: 31.801333,    std: 112.468417),
                "Distance":        PriorsFile.Prior(mean: 23986.717141, std: 43445.689561),
            ],
            weights: [
                "HR":               0.087509,
                "HRV":             -0.007194,
                "skinTemperature": -0.077186,
                "UltraViolet":     -0.039582,
                "stepCount":       -0.059210,
                "Calorie":          0.286595,
                "Distance":        -0.209232,
            ],
            bias: -0.573793
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
    ///   1. z_i = clip((x_i - μ_i) / σ_i, -3, +3)
    ///   2. linear = b + Σ w_i · z_i          (Wx + b)
    ///   3. score  = sigmoid(linear) × 100     (0 = calm, 100 = stressed)
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

        // sigmoid(linear) × 100
        let score = sigmoid(linear) * 100.0

        // Confidence: based on how many features contributed
        let used = rawContributions.count
        let conf: Confidence = used >= 5 ? .high : used >= 3 ? .medium : .low

        // Per-feature score delta: score(linear) - score(linear - w_i·z_i)
        let drivers: [DriverContribution] = rawContributions.map { item in
            let linearWithout = linear - item.weight * item.z
            let delta = score - sigmoid(linearWithout) * 100.0
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



    /// Convenience constructor if you're pulling numbers directly (bypass string parsing).
    static func sample(
        HR: Double? = nil,
        HRV: Double? = nil,
        skinTemperature: Double? = nil,
        UltraViolet: Double? = nil,
        stepCount: Double? = nil,
        Calorie: Double? = nil,
        Distance: Double? = nil
    ) -> FeatureSample {
        FeatureSample(
            HR: HR,
            HRV: HRV,
            skinTemperature: skinTemperature,
            UltraViolet: UltraViolet,
            stepCount: stepCount,
            Calorie: Calorie,
            Distance: Distance
        )
    }

    /// Convenience parser for formatted strings like "54 bpm", "42 ms", "36.5 C".
    static func sampleFromFormattedStrings(
        heartRate: String,
        hrvSDNN: String,
        wristTemperature: String
    ) -> FeatureSample {
        func numeric(from string: String) -> Double? {
            let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
            let allowed = "0123456789.-"
            let filtered = trimmed.filter { allowed.contains($0) }
            guard !filtered.isEmpty else { return nil }
            return Double(filtered)
        }

        let hr   = numeric(from: heartRate)
        let hrv  = numeric(from: hrvSDNN)
        let temp = numeric(from: wristTemperature)

        return FeatureSample(HR: hr, HRV: hrv, skinTemperature: temp)
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

    private func sigmoid(_ x: Double) -> Double {
        1.0 / (1.0 + exp(-min(max(x, -50), 50)))
    }

    private func clamp(_ x: Double, lo: Double, hi: Double) -> Double {
        min(hi, max(lo, x))
    }
}
