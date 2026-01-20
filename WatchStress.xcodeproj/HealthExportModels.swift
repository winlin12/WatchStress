import Foundation

struct HealthExport: Codable {
    let date: String
    let features: HealthFeatures
    let baselines: HealthBaselines
    let selfReport: HealthSelfReport?

    enum CodingKeys: String, CodingKey {
        case date
        case features
        case baselines
        case selfReport = "self_report"
    }
}

struct HealthFeatures: Codable {
    let sleepMinutes: Int?
    let restingHR: Double?
    let hrvSDNNms: Double?
    let steps: Int?
    let activeEnergyKcal: Double?

    enum CodingKeys: String, CodingKey {
        case sleepMinutes = "sleep_minutes"
        case restingHR = "resting_hr"
        case hrvSDNNms = "hrv_sdnn_ms"
        case steps
        case activeEnergyKcal = "active_energy_kcal"
    }
}

struct HealthBaselines: Codable {
    let sleepMinutes14d: Int?
    let restingHR14d: Double?
    let hrvSDNNms14d: Double?

    enum CodingKeys: String, CodingKey {
        case sleepMinutes14d = "sleep_minutes_14d"
        case restingHR14d = "resting_hr_14d"
        case hrvSDNNms14d = "hrv_sdnn_ms_14d"
    }
}

struct HealthSelfReport: Codable {
    let stress0to100: Int?

    enum CodingKeys: String, CodingKey {
        case stress0to100 = "stress_0_100"
    }
}

struct ScoreBreakdown: Codable {
    let date: String
    let score0to100: Int
    let confidence: String?
    let drivers: [ScoreDriver]?
    let missing: [String]?
    let safeActions: [String]?

    enum CodingKeys: String, CodingKey {
        case date
        case score0to100 = "score_0_100"
        case confidence
        case drivers
        case missing
        case safeActions = "safe_actions"
    }
}

struct ScoreDriver: Codable {
    let name: String
    let delta: Double
    let unit: String?
    let impact: Double?
}
