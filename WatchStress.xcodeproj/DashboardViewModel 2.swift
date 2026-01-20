import Foundation
import SwiftUI

final class DashboardViewModel: ObservableObject {
    enum DataConfidence { case low, medium, high }

    // Published properties observed by the UI
    @Published var score: Double
    @Published var confidence: DataConfidence

    // Published metrics (nil if unavailable)
    @Published var restingHR: Double?
    @Published var hrvSDNN: Double?
    @Published var respiratoryRate: Double?
    @Published var sleepHours: Double?

    private let hk = HealthKitManager()

    // Baselines (could be 7/14-day averages). Placeholder values for now.
    private var baselineRestingHR: Double? = 60
    private var baselineHRV: Double? = 50
    private var baselineRespRate: Double? = 15
    private var baselineSleepHours: Double? = 7.0

    init(score: Double = 67, confidence: DataConfidence = .medium) {
        self.score = score
        self.confidence = confidence
    }

    @MainActor
    func start() async {
        do {
            try await hk.requestAuthorization()
            let agg = try await hk.fetchTodayAggregates()
            update(restingHR: agg.restingHR, hrvSDNN: agg.hrvSDNN, respiratoryRate: agg.respiratoryRate, sleepHours: agg.sleepHours)
        } catch {
            // Keep UI responsive even if auth fails; show low confidence
            update(restingHR: nil, hrvSDNN: nil, respiratoryRate: nil, sleepHours: nil)
        }
    }

    @MainActor
    func update(restingHR: Double?, hrvSDNN: Double?, respiratoryRate: Double?, sleepHours: Double?) {
        self.restingHR = restingHR
        self.hrvSDNN = hrvSDNN
        self.respiratoryRate = respiratoryRate
        self.sleepHours = sleepHours

        // Update confidence based on availability
        let available = [restingHR != nil, hrvSDNN != nil, respiratoryRate != nil, sleepHours != nil].filter { $0 }.count
        switch available {
        case 0...1: confidence = .low
        case 2...3: confidence = .medium
        default: confidence = .high
        }

        // Recompute score using a simple heuristic vs baselines
        score = computeScore()
    }

    private func computeScore() -> Double {
        var components: [Double] = []

        if let rhr = restingHR, let base = baselineRestingHR {
            // Lower resting HR better; penalize increases
            let delta = rhr - base
            components.append(scaleDelta(delta: -delta, range: 15))
        }

        if let hrv = hrvSDNN, let base = baselineHRV {
            // Higher HRV better; reward increases
            let delta = hrv - base
            components.append(scaleDelta(delta: delta, range: 30))
        }

        if let rr = respiratoryRate, let base = baselineRespRate {
            // Lower resp rate slightly better
            let delta = base - rr
            components.append(scaleDelta(delta: delta, range: 6))
        }

        if let sleep = sleepHours, let base = baselineSleepHours {
            // Closer to baseline sleep is better; penalize short sleep more than long
            let delta = sleep - base
            components.append(scaleDelta(delta: delta, range: 2.5))
        }

        if components.isEmpty { return 50 } // neutral when nothing available

        let avg = components.reduce(0, +) / Double(components.count)
        return max(0, min(100, avg))
    }

    // Map a signed delta into 0..100 centered near 50 with clamping
    private func scaleDelta(delta: Double, range: Double) -> Double {
        let normalized = max(-range, min(range, delta)) / range // -1..1
        let score = 50 + (normalized * 40) // 10..90 spread
        return score
    }
}
