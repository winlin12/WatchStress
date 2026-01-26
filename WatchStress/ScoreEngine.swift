//
//  ScoreEngine.swift
//  WatchStress
//
//  Loads WESAD-derived priors/weights from priors.json and computes:
//
//    z_i = clip((x_i - mu_i) / (sigma_i + eps), -3, +3)
//    rawStress = b + sum(w_i * z_i)
//    stressIndex = lambda * prev + (1 - lambda) * rawStress
//    score = 80 - 80 * tanh(k * s)            for s >= 0
//    score = 80 + 20 * (1 - exp(k * s))       for s < 0
//
//  Notes:
//  - The weights in priors.json are stress-direction: positive means feature tends to
//    increase under stress, so the tanh mapping lowers the score as stressIndex rises.
//  - Baseline adapts from WESAD -> user by blending WESAD priors with user stats.
//

import Foundation

final class ScoreEngine {

    // MARK: - Model feature space (matches priors.json keys)

    enum Feature: String, CaseIterable, Codable {
        case hrMeanBPM
        case hrvSDNNms
        case wristTempC
    }

    struct FeatureSample {
        var hrMeanBPM: Double?
        var hrvSDNNms: Double?
        var wristTempC: Double?

        func value(for f: Feature) -> Double? {
            switch f {
            case .hrMeanBPM: return hrMeanBPM
            case .hrvSDNNms: return hrvSDNNms
            case .wristTempC: return wristTempC
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
        let score: Double
        let rawScore: Double
        let rawStressIndex: Double
        let stressIndex: Double
        let confidence: Confidence
        let k: Double
        let lambda: Double
        let bias: Double
        let drivers: [DriverContribution]   // sorted by absolute effect on score
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
    private let smoothingLambda: Double = 0.90

    /// "How extreme the ring feels". Larger => bigger swings for the same stressIndex.
    /// Tune this in-app without retraining.
    private(set) var k: Double

    /// Baseline adapts from WESAD priors -> user stats.
    /// Blend factor: a = n / (n + k). Larger k = slower personalization.
    private let baselineBlendK: Double = 30.0

    /// Keep user std from collapsing too far below WESAD std (stability).
    private let userStdFloorFracOfWesad: Double = 0.20

    /// Require at least this many user samples for a feature before blending in std meaningfully.
    private let minUserSamplesForStd: Int = 3

    // MARK: - State

    private let priorsFile: PriorsFile
    private var lastStressIndex: Double?

    // MARK: - Init

    /// Load priors.json from app bundle (recommended).
    /// Add priors.json to your Xcode target's "Copy Bundle Resources".
    convenience init?(k: Double = 0.20, priorsResourceName: String = "priors") {
        guard let url = Bundle.main.url(forResource: priorsResourceName, withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode(PriorsFile.self, from: data)
        else {
            return nil
        }
        self.init(k: k, priorsFile: decoded)
    }

    init(k: Double = 0.20, priorsFile: PriorsFile) {
        self.k = k
        self.priorsFile = priorsFile
    }

    // MARK: - Public API

    /// Optional: expose the learned vector (feature order + weights).
    func modelVector() -> (features: [Feature], weights: [Double], bias: Double) {
        let feats = Feature.allCases
        let ws = feats.map { priorsFile.weights[$0.rawValue] ?? 0.0 }
        return (feats, ws, priorsFile.bias)
    }

    /// Compute score without mutating user baselines.
    func computeScore(sample: FeatureSample) -> ScoreResult {
        let userStats = Self.loadAllUserStats()
        let bias = priorsFile.bias

        // Build stress index from available features
        var stressIndex = bias
        var contributions: [DriverContribution] = []
        var rawContributions: [(feature: Feature, weight: Double, z: Double, value: Double, mean: Double, std: Double, blendA: Double, stressContribution: Double)] = []
        var blendAlphas: [Double] = []

        for f in Feature.allCases {
            guard let x = sample.value(for: f) else { continue }
            guard let w = priorsFile.weights[f.rawValue] else { continue }
            guard let prior = priorsFile.priors[f.rawValue] else { continue }

            let st = userStats[f]  // may be nil
            let (mu, sd, blendA) = blendedBaseline(prior: prior, user: st)
            blendAlphas.append(blendA)

            let zRaw = (x - mu) / (sd + eps)
            let z = clamp(zRaw, lo: -zClip, hi: zClip)
            stressIndex += w * z

            rawContributions.append(
                (
                    feature: f,
                    weight: w,
                    z: z,
                    value: x,
                    mean: mu,
                    std: sd,
                    blendA: blendA,
                    stressContribution: w * z
                )
            )
        }

        // If no usable features, baseline score with low confidence.
        if rawContributions.isEmpty {
            let fallbackScore = scoreFromStressIndex(bias)
            return ScoreResult(
                score: fallbackScore,
                rawScore: fallbackScore,
                rawStressIndex: bias,
                stressIndex: bias,
                confidence: .low,
                k: k,
                lambda: smoothingLambda,
                bias: bias,
                drivers: []
            )
        }

        let rawStressIndex = stressIndex
        let previousStressIndex = lastStressIndex
        let smoothedStressIndex: Double
        if let previousStressIndex {
            smoothedStressIndex = (smoothingLambda * previousStressIndex) + ((1.0 - smoothingLambda) * rawStressIndex)
        } else {
            smoothedStressIndex = rawStressIndex
        }
        lastStressIndex = smoothedStressIndex

        // Map to 0-100 ring score using tanh
        let raw = scoreFromStressIndex(smoothedStressIndex)
        let score = clamp(raw, lo: 0.0, hi: 100.0)

        // Confidence heuristic
        let avgBlend = blendAlphas.isEmpty ? 0.0 : (blendAlphas.reduce(0.0, +) / Double(blendAlphas.count))
        let used = rawContributions.count

        var conf: Confidence = .low
        if used >= 2 { conf = .medium }
        if used >= 3 { conf = .high }
        if avgBlend >= 0.50 && used >= 2 { conf = .high } // personalized + enough signals
        if rawContributions.contains(where: { abs($0.z) >= zClip }) { conf = .low } // clipped => less trust

        contributions = rawContributions.map { item in
            let withoutRaw = rawStressIndex - item.stressContribution
            let withoutSmoothed: Double
            if let previousStressIndex {
                withoutSmoothed = (smoothingLambda * previousStressIndex) + ((1.0 - smoothingLambda) * withoutRaw)
            } else {
                withoutSmoothed = withoutRaw
            }
            let delta = scoreFromStressIndex(smoothedStressIndex) - scoreFromStressIndex(withoutSmoothed)
            return DriverContribution(
                feature: item.feature,
                weight: item.weight,
                z: item.z,
                scoreDelta: delta,
                value: item.value,
                mean: item.mean,
                std: item.std,
                blendA: item.blendA
            )
        }
        let sortedDrivers = contributions.sorted { abs($0.scoreDelta) > abs($1.scoreDelta) }
        return ScoreResult(
            score: score,
            rawScore: raw,
            rawStressIndex: rawStressIndex,
            stressIndex: smoothedStressIndex,
            confidence: conf,
            k: k,
            lambda: smoothingLambda,
            bias: bias,
            drivers: sortedDrivers
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

    /// Optional: adjust ring "extremeness" on the fly.
    func setK(_ newK: Double) {
        k = max(0.0, newK)
    }

    // MARK: - Baseline blending (WESAD -> user)

    private func blendedBaseline(prior: PriorsFile.Prior, user: RunningStats?) -> (mean: Double, std: Double, blendA: Double) {
        guard let user = user, user.count >= 1 else {
            return (prior.mean, max(prior.std, eps), 0.0)
        }

        // Blend factor goes from 0 -> 1 as n grows
        let n = Double(user.count)
        let a = n / (n + baselineBlendK)

        // Mean blends immediately
        let mu = (1.0 - a) * prior.mean + a * user.mean

        // Std: only reliable after a few samples; also floor it for stability
        let userStdRaw = user.stdDev
        let userStdFloor = max(prior.std * userStdFloorFracOfWesad, eps)
        let userStd = (user.count >= minUserSamplesForStd) ? max(userStdRaw, userStdFloor) : prior.std

        let sd = (1.0 - a) * prior.std + a * userStd
        return (mu, max(sd, eps), a)
    }

    // MARK: - Parsing helpers (optional)

    /// Convenience constructor if you're pulling numbers directly (bypass string parsing).
    static func sample(
        hrMeanBPM: Double? = nil,
        hrvSDNNms: Double? = nil,
        wristTempC: Double? = nil
    ) -> FeatureSample {
        FeatureSample(
            hrMeanBPM: hrMeanBPM,
            hrvSDNNms: hrvSDNNms,
            wristTempC: wristTempC
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

        let hr = numeric(from: heartRate)
        let hrv = numeric(from: hrvSDNN)
        let temp = numeric(from: wristTemperature)

        return FeatureSample(hrMeanBPM: hr, hrvSDNNms: hrv, wristTempC: temp)
    }

    // MARK: - Persistence (UserDefaults)

    private struct RunningStats: Codable {
        var count: Int = 0
        var mean: Double = 0
        var m2: Double = 0

        init() {}

        init(count: Int, mean: Double, stdDev: Double) {
            self.count = max(0, count)
            self.mean = mean
            if count > 1 {
                let variance = max(0.0, stdDev * stdDev)
                self.m2 = variance * Double(count - 1)
            } else {
                self.m2 = 0
            }
        }

        mutating func update(with x: Double) {
            count += 1
            let delta = x - mean
            mean += delta / Double(count)
            let delta2 = x - mean
            m2 += delta * delta2
        }

        var variance: Double { count > 1 ? m2 / Double(count - 1) : 0 }
        var stdDev: Double { max(variance, 0).squareRoot() }
    }

    static func storeUserBaselines(_ baselines: [Feature: BaselineStats]) {
        var stats = loadAllUserStats()
        for (feature, baseline) in baselines {
            stats[feature] = RunningStats(count: baseline.count, mean: baseline.mean, stdDev: baseline.stdDev)
        }
        saveAllUserStats(stats)
    }

    private static func key(for feature: Feature) -> String { "ScoreEngine.user.\(feature.rawValue)" }

    private static func loadAllUserStats() -> [Feature: RunningStats] {
        var result: [Feature: RunningStats] = [:]
        let defaults = UserDefaults.standard
        let decoder = JSONDecoder()

        for f in Feature.allCases {
            let k = key(for: f)
            if let data = defaults.data(forKey: k),
               let st = try? decoder.decode(RunningStats.self, from: data) {
                result[f] = st
            }
        }
        return result
    }

    private static func saveAllUserStats(_ stats: [Feature: RunningStats]) {
        let defaults = UserDefaults.standard
        let encoder = JSONEncoder()

        for (f, st) in stats {
            let k = key(for: f)
            if let data = try? encoder.encode(st) {
                defaults.set(data, forKey: k)
            }
        }
    }

    private static func dayKey(for date: Date, calendar: Calendar) -> String {
        let comps = calendar.dateComponents([.year, .month, .day], from: date)
        let y = comps.year ?? 0
        let m = comps.month ?? 0
        let d = comps.day ?? 0
        return "\(y)-\(m)-\(d)"
    }

    // MARK: - Utilities

    private func scoreFromStressIndex(_ stressIndex: Double) -> Double {
        let base = 80.0
        if stressIndex >= 0.0 {
            return base - base * tanh(k * stressIndex)
        }
        return base + (100.0 - base) * (1.0 - exp(k * stressIndex))
    }

    private func clamp(_ x: Double, lo: Double, hi: Double) -> Double {
        return min(hi, max(lo, x))
    }
}
