import Foundation

/// A lightweight scoring engine that computes a daily stress score from health features
/// using rolling per-user baselines (mean/std) persisted in UserDefaults.
///
/// Scoring formula:
///   s = clip(50 + sum_i w_i * d_i, 0, 100), where d_i = (x_i - mu_i) / (sigma_i + eps)
/// Confidence is derived from signal presence, baseline maturity, and outlier checks.
final class ScoreEngine {
    // MARK: - Feature definitions

    enum Feature: String, CaseIterable {
        case sleepHours       // hours (Double)
        case restingHR        // bpm
        case hrvSDNN          // ms
        case steps            // count
        case exerciseMinutes  // minutes
    }

    struct FeatureSample {
        var sleepHours: Double?
        var restingHR: Double?
        var hrvSDNN: Double?
        var steps: Double?
        var exerciseMinutes: Double?

        func value(for feature: Feature) -> Double? {
            switch feature {
            case .sleepHours: return sleepHours
            case .restingHR: return restingHR
            case .hrvSDNN: return hrvSDNN
            case .steps: return steps
            case .exerciseMinutes: return exerciseMinutes
            }
        }
    }

    struct DriverContribution {
        let feature: Feature
        let weight: Double
        let delta: Double
        var contribution: Double { weight * delta }
    }

    // Confidence mapped to a user-facing label
    enum Confidence: String {
        case low = "Low"
        case medium = "Medium"
        case high = "High"
    }

    struct ScoreResult {
        let score: Double?      // nil if insufficient maturity/signals
        let confidence: Confidence
        let drivers: [DriverContribution]
    }

    // MARK: - Configuration

    private let eps: Double = 1e-6
    private let minMatureSamplesPerCore = 3   // minimum samples for core features to be considered mature
    private let coreFeatures: [Feature] = [.sleepHours, .restingHR, .hrvSDNN]

    // Default weights: positive means higher-than-baseline increases score, negative decreases.
    // These are conservative and chosen for interpretability.
    private let weights: [Feature: Double] = [
        .sleepHours: 8.0,
        .restingHR: -6.0,
        .hrvSDNN: 6.0,
        .steps: 4.0,
        .exerciseMinutes: 4.0
    ]

    // MARK: - Public API

    /// Compute a score and confidence for the given sample without mutating baselines.
    func computeScore(sample: FeatureSample) -> ScoreResult {
        // Load baselines
        let stats = loadAllStats()

        // Evaluate maturity for core features
        let matureCoreCount = coreFeatures.filter { (stats[$0]?.count ?? 0) >= minMatureSamplesPerCore }.count
        let presentCoreCount = coreFeatures.filter { sample.value(for: $0) != nil }.count

        // Build contributions for available features with mature baselines
        var contributions: [DriverContribution] = []
        for f in Feature.allCases {
            guard let x = sample.value(for: f), let st = stats[f], st.count >= 1 else { continue }
            let d = (x - st.mean) / (st.stdDev + eps)
            if let w = weights[f] {
                contributions.append(DriverContribution(feature: f, weight: w, delta: d))
            }
        }

        // Sum contributions
        let total = contributions.reduce(0.0) { $0 + $1.contribution }
        let raw = 50.0 + total
        let clipped = max(0.0, min(100.0, raw))

        // Confidence: start medium, adjust
        var conf: Confidence = .medium
        let keyPresent = presentCoreCount
        if keyPresent >= 2 && matureCoreCount >= 2 { conf = .medium } else { conf = .low }
        if keyPresent == coreFeatures.count && matureCoreCount == coreFeatures.count { conf = .high }

        // Outlier check: if any |delta| is huge, reduce confidence
        if contributions.contains(where: { abs($0.delta) > 3.5 }) {
            conf = .low
        }

        // If no contributions (e.g., all missing), return a neutral score with low confidence
        if contributions.isEmpty {
            return ScoreResult(score: 50.0, confidence: .low, drivers: [])
        }

        return ScoreResult(score: clipped, confidence: conf, drivers: contributions.sorted { abs($0.contribution) > abs($1.contribution) })
    }

    /// Update rolling baselines with today's sample once per day.
    func updateBaselinesIfNeeded(with sample: FeatureSample, today: Date = Date()) {
        let cal = Calendar.current
        let lastKey = "ScoreEngine.lastUpdateDay"
        let defaults = UserDefaults.standard
        let lastDay = defaults.string(forKey: lastKey)
        let currentDay = Self.dayKey(for: today, calendar: cal)
        guard lastDay != currentDay else { return } // already updated today

        var stats = loadAllStats()
        for f in Feature.allCases {
            if let x = sample.value(for: f) {
                var st = stats[f] ?? RunningStats()
                st.update(with: x)
                stats[f] = st
            }
        }
        saveAllStats(stats)
        defaults.set(currentDay, forKey: lastKey)
    }

    // MARK: - Parsing helpers

    /// Build a FeatureSample from formatted strings (e.g., from VitalsViewModel).
    /// Accepted formats:
    ///  - Sleep: "7h 45m"
    ///  - Resting HR / Heart Rate: "54 bpm"
    ///  - HRV: "42 ms"
    ///  - Steps: "1234"
    ///  - Exercise: "23 min"
    static func sampleFromFormattedStrings(
        sleep: String,
        restingHR: String,
        hrvSDNN: String,
        steps: String,
        exerciseMinutes: String
    ) -> FeatureSample {
        func numeric(from string: String, suffixes: [String] = []) -> Double? {
            var s = string
            for suf in suffixes { s = s.replacingOccurrences(of: suf, with: "") }
            s = s.trimmingCharacters(in: .whitespacesAndNewlines)
            return Double(s)
        }

        var sleepHours: Double? = nil
        if sleep != "—" {
            // Format like "7h 45m"
            let comps = sleep.split(separator: " ")
            var hours = 0.0
            for c in comps {
                if c.hasSuffix("h"), let h = Double(c.dropLast()) { hours += h }
                else if c.hasSuffix("m"), let m = Double(c.dropLast()) { hours += m / 60.0 }
            }
            sleepHours = hours > 0 ? hours : nil
        }

        var rhr: Double? = nil
        if restingHR != "—" {
            rhr = numeric(from: restingHR, suffixes: [" bpm"])
        }

        var hrv: Double? = nil
        if hrvSDNN != "—" {
            hrv = numeric(from: hrvSDNN, suffixes: [" ms"])
        }

        var stepsValue: Double? = nil
        if steps != "—" {
            stepsValue = numeric(from: steps)
        }

        var exercise: Double? = nil
        if exerciseMinutes != "—" {
            exercise = numeric(from: exerciseMinutes, suffixes: [" min"])
        }

        return FeatureSample(
            sleepHours: sleepHours,
            restingHR: rhr,
            hrvSDNN: hrv,
            steps: stepsValue,
            exerciseMinutes: exercise
        )
    }

    // MARK: - Persistence (UserDefaults)

    private struct RunningStats: Codable {
        var count: Int = 0
        var mean: Double = 0
        var m2: Double = 0 // sum of squares of differences from the current mean

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

    private func key(for feature: Feature) -> String { "ScoreEngine.\(feature.rawValue)" }

    private func loadAllStats() -> [Feature: RunningStats] {
        var result: [Feature: RunningStats] = [:]
        let defaults = UserDefaults.standard
        for f in Feature.allCases {
            let k = key(for: f)
            if let data = defaults.data(forKey: k),
               let st = try? JSONDecoder().decode(RunningStats.self, from: data) {
                result[f] = st
            }
        }
        return result
    }

    private func saveAllStats(_ stats: [Feature: RunningStats]) {
        let defaults = UserDefaults.standard
        for (f, st) in stats {
            let k = key(for: f)
            if let data = try? JSONEncoder().encode(st) {
                defaults.set(data, forKey: k)
            }
        }
    }

    private static func dayKey(for date: Date, calendar: Calendar) -> String {
        let comps = calendar.dateComponents([.year, .month, .day], from: date)
        return "\(comps.year ?? 0)-\(comps.month ?? 0)-\(comps.day ?? 0)"
    }
}

