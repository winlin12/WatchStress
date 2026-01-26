import Foundation

/// One logged snapshot at a point in time.
struct StressLogEntry: Codable, Identifiable {
    let id: UUID
    let timestamp: Date

    // Inputs (aligned snapshot)
    let sleepHours: Double?
    let heartRate: Double?
    let restingHR: Double?
    let hrvSDNN: Double?
    let bloodPressureSystolic: Double?
    let bloodPressureDiastolic: Double?
    let wristTemperature: Double?
    let respirationRate: Double?
    let bloodOxygen: Double?
    let steps: Double?
    let exerciseMinutes: Double?

    // Outputs
    let score: Double?
    let confidence: String

    // Optional user feedback (leave nil for now)
    let selfReport: Double?
    let note: String?
}

actor StressLogStore {
    static let shared = StressLogStore()

    private let filename = "stress_log.jsonl"
    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        return e
    }()

    private func fileURL() throws -> URL {
        let dir = try FileManager.default.url(for: .documentDirectory,
                                              in: .userDomainMask,
                                              appropriateFor: nil,
                                              create: true)
        return dir.appendingPathComponent(filename)
    }

    func append(entry: StressLogEntry) throws {
        let url = try fileURL()
        let data = try encoder.encode(entry)
        let line = data + Data([0x0A]) // newline

        if FileManager.default.fileExists(atPath: url.path) {
            let handle = try FileHandle(forWritingTo: url)
            try handle.seekToEnd()
            try handle.write(contentsOf: line)
            try handle.close()
        } else {
            try line.write(to: url, options: .atomic)
        }
    }

    func deleteAll() throws {
        let url = try fileURL()
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        try FileManager.default.removeItem(at: url)
    }

    /// Export logged data as CSV for sharing.
    func exportCSV() throws -> URL {
        let entries = try loadAll()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("stress_log_export.csv")

        var lines: [String] = []
        lines.append([
            "id",
            "timestamp",
            "score",
            "confidence",
            "sleepHours",
            "heartRate",
            "restingHR",
            "hrvSDNN",
            "bloodPressureSystolic",
            "bloodPressureDiastolic",
            "wristTemperature",
            "respirationRate",
            "bloodOxygen",
            "steps",
            "exerciseMinutes",
            "selfReport",
            "note"
        ].joined(separator: ","))

        let formatter = ISO8601DateFormatter()
        for entry in entries {
            let row: [String] = [
                csvField(entry.id.uuidString),
                csvField(formatter.string(from: entry.timestamp)),
                formatDouble(entry.score),
                csvField(entry.confidence),
                formatDouble(entry.sleepHours),
                formatDouble(entry.heartRate),
                formatDouble(entry.restingHR),
                formatDouble(entry.hrvSDNN),
                formatDouble(entry.bloodPressureSystolic),
                formatDouble(entry.bloodPressureDiastolic),
                formatDouble(entry.wristTemperature),
                formatDouble(entry.respirationRate),
                formatDouble(entry.bloodOxygen),
                formatDouble(entry.steps),
                formatDouble(entry.exerciseMinutes),
                formatDouble(entry.selfReport),
                csvField(entry.note ?? "")
            ]
            lines.append(row.joined(separator: ","))
        }

        let text = lines.joined(separator: "\n")
        try text.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    /// (Optional) read back all entries for debugging / plots later.
    func loadAll() throws -> [StressLogEntry] {
        let url = try fileURL()
        guard FileManager.default.fileExists(atPath: url.path) else { return [] }
        let text = try String(contentsOf: url, encoding: .utf8)
        var out: [StressLogEntry] = []
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        for line in text.split(separator: "\n") {
            if let data = line.data(using: .utf8),
               let entry = try? decoder.decode(StressLogEntry.self, from: data) {
                out.append(entry)
            }
        }
        return out
    }

    private func formatDouble(_ value: Double?) -> String {
        guard let value else { return "" }
        return String(format: "%.4f", value)
    }

    private func csvField(_ value: String) -> String {
        guard !value.isEmpty else { return "" }
        let needsQuotes = value.contains(",") || value.contains("\"") || value.contains("\n")
        var escaped = value.replacingOccurrences(of: "\"", with: "\"\"")
        if needsQuotes {
            escaped = "\"\(escaped)\""
        }
        return escaped
    }
}
