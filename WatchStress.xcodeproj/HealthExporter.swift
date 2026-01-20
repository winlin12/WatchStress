import Foundation
import HealthKit

final class HealthExporter {
    static let shared = HealthExporter()

    private var healthStore: HKHealthStore {
        HealthKitManager.shared.healthStore
    }

    func buildDailyExport(for date: Date) async throws -> HealthExport {
        let (start, end) = dateBounds(for: date)
        let dateString = Self.dateString(from: date)

        async let sleepMinutes = querySleepMinutes(start: start, end: end)
        async let restingHR = queryAverage(for: .restingHeartRate, unit: HKUnit.count().unitDivided(by: HKUnit.minute()), start: start, end: end)
        async let hrvSDNN = queryMedianHRV(start: start, end: end)
        async let steps = querySum(for: .stepCount, unit: HKUnit.count(), start: start, end: end)
        async let activeEnergy = querySum(for: .activeEnergyBurned, unit: HKUnit.kilocalorie(), start: start, end: end)

        async let baselineSleepMinutes = fourteenDayAverage(date: date, query: { dayStart, dayEnd in
            querySleepMinutes(start: dayStart, end: dayEnd).map { Double($0) }
        })
        async let baselineRestingHR = fourteenDayAverage(date: date, query: { dayStart, dayEnd in
            queryAverage(for: .restingHeartRate, unit: HKUnit.count().unitDivided(by: HKUnit.minute()), start: dayStart, end: dayEnd)
        })
        async let baselineHRVSDNN = fourteenDayAverage(date: date, query: { dayStart, dayEnd in
            queryMedianHRV(start: dayStart, end: dayEnd)
        })

        let features = HealthExport.Features(
            sleep_minutes: try await sleepMinutes.map(Int.init) ?? 0,
            resting_hr: try await restingHR ?? 0,
            hrv_sdnn_ms: try await hrvSDNN ?? 0,
            steps: try await steps.map(Int.init) ?? 0,
            active_energy_kcal: try await activeEnergy ?? 0
        )

        let baselines = HealthExport.Baselines(
            sleep_minutes: try await baselineSleepMinutes.map(Int.init) ?? 0,
            resting_hr: try await baselineRestingHR ?? 0,
            hrv_sdnn_ms: try await baselineHRVSDNN ?? 0
        )

        return HealthExport(date: dateString, features: features, baselines: baselines)
    }

    private func dateBounds(for date: Date) -> (start: Date, end: Date) {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!

        let start = calendar.startOfDay(for: date)
        let end = calendar.date(byAdding: .day, value: 1, to: start)!
        return (start, end)
    }

    private static func dateString(from date: Date) -> String {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.string(from: date)
    }

    private func querySum(for identifier: HKQuantityTypeIdentifier, unit: HKUnit, start: Date, end: Date) async -> Double? {
        guard let quantityType = HKObjectType.quantityType(forIdentifier: identifier) else { return nil }

        return await withCheckedContinuation { continuation in
            let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: [])
            let query = HKStatisticsQuery(quantityType: quantityType, quantitySamplePredicate: predicate, options: .cumulativeSum) { _, result, _ in
                guard let sum = result?.sumQuantity() else {
                    continuation.resume(returning: nil)
                    return
                }
                continuation.resume(returning: sum.doubleValue(for: unit))
            }
            healthStore.execute(query)
        }
    }

    private func queryAverage(for identifier: HKQuantityTypeIdentifier, unit: HKUnit, start: Date, end: Date) async -> Double? {
        guard let quantityType = HKObjectType.quantityType(forIdentifier: identifier) else { return nil }

        return await withCheckedContinuation { continuation in
            let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: [])
            let query = HKStatisticsQuery(quantityType: quantityType, quantitySamplePredicate: predicate, options: .discreteAverage) { _, result, _ in
                guard let avg = result?.averageQuantity() else {
                    continuation.resume(returning: nil)
                    return
                }
                continuation.resume(returning: avg.doubleValue(for: unit))
            }
            healthStore.execute(query)
        }
    }

    private func queryMedianHRV(start: Date, end: Date) async -> Double? {
        guard let sampleType = HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else { return nil }

        return await withCheckedContinuation { continuation in
            let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: [])
            let query = HKSampleQuery(sampleType: sampleType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, samplesOrNil, errorOrNil in
                guard errorOrNil == nil, let samples = samplesOrNil as? [HKQuantitySample], !samples.isEmpty else {
                    continuation.resume(returning: nil)
                    return
                }
                let valuesMs = samples.map { $0.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli)) }
                let sorted = valuesMs.sorted()
                let mid = sorted.count / 2
                let median: Double
                if sorted.count % 2 == 0 {
                    median = (sorted[mid - 1] + sorted[mid]) / 2
                } else {
                    median = sorted[mid]
                }
                continuation.resume(returning: median)
            }
            healthStore.execute(query)
        }
    }

    private func querySleepMinutes(start: Date, end: Date) async -> Int? {
        guard let categoryType = HKObjectType.categoryType(forIdentifier: .sleepAnalysis) else { return nil }

        return await withCheckedContinuation { continuation in
            let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: [])
            let query = HKSampleQuery(sampleType: categoryType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, samplesOrNil, errorOrNil in
                guard errorOrNil == nil, let samples = samplesOrNil as? [HKCategorySample], !samples.isEmpty else {
                    continuation.resume(returning: nil)
                    return
                }

                let minutes = samples.reduce(0) { partialResult, sample in
                    if sample.value == HKCategoryValueSleepAnalysis.inBed.rawValue ||
                        sample.value == HKCategoryValueSleepAnalysis.asleep.rawValue {
                        let overlapStart = max(sample.startDate, start)
                        let overlapEnd = min(sample.endDate, end)
                        let interval = overlapEnd.timeIntervalSince(overlapStart)
                        if interval > 0 {
                            return partialResult + Int(interval / 60)
                        }
                    }
                    return partialResult
                }
                continuation.resume(returning: minutes)
            }
            healthStore.execute(query)
        }
    }

    private func fourteenDayAverage(date: Date, query: @escaping (Date, Date) async -> Double?) async -> Double? {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!

        var values: [Double] = []
        for dayOffset in -14..<0 {
            guard let dayDate = calendar.date(byAdding: .day, value: dayOffset, to: date) else { continue }
            let (start, end) = dateBounds(for: dayDate)
            if let value = await query(start, end) {
                values.append(value)
            }
        }
        guard !values.isEmpty else { return nil }
        let sum = values.reduce(0, +)
        return sum / Double(values.count)
    }
}

public struct HealthExport: Codable {
    public let date: String
    public let features: Features
    public let baselines: Baselines

    public struct Features: Codable {
        public let sleep_minutes: Int
        public let resting_hr: Double
        public let hrv_sdnn_ms: Double
        public let steps: Int
        public let active_energy_kcal: Double

        public init(sleep_minutes: Int, resting_hr: Double, hrv_sdnn_ms: Double, steps: Int, active_energy_kcal: Double) {
            self.sleep_minutes = sleep_minutes
            self.resting_hr = resting_hr
            self.hrv_sdnn_ms = hrv_sdnn_ms
            self.steps = steps
            self.active_energy_kcal = active_energy_kcal
        }
    }

    public struct Baselines: Codable {
        public let sleep_minutes: Int
        public let resting_hr: Double
        public let hrv_sdnn_ms: Double

        public init(sleep_minutes: Int, resting_hr: Double, hrv_sdnn_ms: Double) {
            self.sleep_minutes = sleep_minutes
            self.resting_hr = resting_hr
            self.hrv_sdnn_ms = hrv_sdnn_ms
        }
    }

    public init(date: String, features: Features, baselines: Baselines) {
        self.date = date
        self.features = features
        self.baselines = baselines
    }
}
