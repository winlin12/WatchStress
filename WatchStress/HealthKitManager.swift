//  HealthKitManager.swift
//  WatchStress
//
//  Centralizes HealthKit authorization and queries for vitals used in the app.

import Foundation
import HealthKit

/// HealthKitManager centralizes HealthKit authorization and query helpers used by the app.
/// Use the shared singleton to request authorization and fetch HealthKit values.
/// Note: HealthKit invokes completion handlers on a background queue; callers updating UI should hop to the main actor.
final class HealthKitManager {
    /// Shared singleton instance used across the app.
    static let shared = HealthKitManager()
    /// Underlying HKHealthStore used to perform authorization and execute queries.
    let healthStore = HKHealthStore()

    /// Private initializer to enforce singleton usage.
    private init() {}

    // MARK: - Authorization

    // Authorization: core vs optional metrics
    // Core (high availability): heartRate, restingHeartRate, heartRateVariabilitySDNN, stepCount, appleExerciseTime, sleepAnalysis
    // Optional (often missing / device-dependent): respiratoryRate, oxygenSaturation, blood pressure (systolic/diastolic), wrist/body temperature
    // Pass includeOptional = false to request only core metrics (e.g., first run), and later call again with includeOptional = true if the user opts in.
    /// Requests read authorization for HealthKit types used by the app.
    /// - Parameter includeOptional: When false, requests only core metrics (heart rate, resting HR, HRV, steps, exercise, sleep). When true, also requests optional metrics (respiration, SpO₂, blood pressure, temperature).
    /// - Parameter completion: Called with the system authorization result. Error may be nil even if success is false.
    func requestAuthorization(includeOptional: Bool = true, completion: @escaping (Bool, Error?) -> Void) {
        guard HKHealthStore.isHealthDataAvailable() else {
            completion(false, NSError(domain: "HealthKit", code: 1, userInfo: [NSLocalizedDescriptionKey: "Health data not available on this device."]))
            return
        }

        var readTypes: Set<HKObjectType> = []

        // Core quantity identifiers
        let coreQuantityIdentifiers: [HKQuantityTypeIdentifier] = [
            .heartRate,
            .restingHeartRate,
            .heartRateVariabilitySDNN,
            .stepCount,
            .appleExerciseTime
        ]

        // Optional quantity identifiers
        let optionalQuantityIdentifiers: [HKQuantityTypeIdentifier] = [
            .respiratoryRate,
            .oxygenSaturation,
            .bloodPressureSystolic,
            .bloodPressureDiastolic
        ]

        // Always include core metrics
        for id in coreQuantityIdentifiers {
            if let t = HKObjectType.quantityType(forIdentifier: id) { readTypes.insert(t) }
        }

        // Optionally include additional metrics
        if includeOptional {
            for id in optionalQuantityIdentifiers {
                if let t = HKObjectType.quantityType(forIdentifier: id) { readTypes.insert(t) }
            }
        }

        // Wrist/body temperature treated as optional
        if includeOptional {
            if #available(iOS 16.0, *) {
                if let wristTemp = HKObjectType.quantityType(forIdentifier: .appleSleepingWristTemperature) {
                    readTypes.insert(wristTemp)
                }
            }
            if let bodyTemp = HKObjectType.quantityType(forIdentifier: .bodyTemperature) {
                readTypes.insert(bodyTemp)
            }
        }

        // Sleep is a core metric
        if let sleepType = HKObjectType.categoryType(forIdentifier: .sleepAnalysis) {
            readTypes.insert(sleepType)
        }

        healthStore.requestAuthorization(toShare: nil, read: readTypes) { success, error in
            completion(success, error)
        }
    }

    // MARK: - Queries

    /// Fetches the most recent quantity sample for a given type.
    /// - Parameters:
    ///   - type: The HKQuantityType to query (e.g., heartRate).
    ///   - completion: Invoked with the latest HKQuantitySample if available.
    /// - Note: Units are not converted here; callers should convert using appropriate HKUnit.
    func mostRecentQuantitySample(for type: HKQuantityType, completion: @escaping (HKQuantitySample?) -> Void) {
        // Sort by endDate descending to get the latest sample
        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        // Query the most recent sample (limit 1)
        let query = HKSampleQuery(sampleType: type, predicate: nil, limit: 1, sortDescriptors: [sort]) { _, results, _ in
            completion(results?.first as? HKQuantitySample)
        }
        // Execute the query
        healthStore.execute(query)
    }

    /// Computes an aggregate statistic for the current day (from midnight to now).
    /// - Parameters:
    ///   - type: The HKQuantityType to aggregate (e.g., stepCount).
    ///   - options: Statistics options (.cumulativeSum by default).
    ///   - completion: Invoked with HKStatistics containing the result (if any).
    func todaySum(for type: HKQuantityType, options: HKStatisticsOptions = .cumulativeSum, completion: @escaping (HKStatistics?) -> Void) {
        // Aggregate over the current day (midnight -> now)
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: Date())
        let predicate = HKQuery.predicateForSamples(withStart: startOfDay, end: Date(), options: .strictStartDate)
        // Use HKStatisticsQuery to compute the requested statistic
        let query = HKStatisticsQuery(quantityType: type, quantitySamplePredicate: predicate, options: options) { _, stats, _ in
            completion(stats)
        }
        // Execute the query
        healthStore.execute(query)
    }

    /// Computes total sleep duration for last night (asleep segments only).
    /// The window is defined as yesterday 12:00 PM to today 12:00 PM to capture sleep crossing midnight.
    /// - Parameter completion: Invoked with the total TimeInterval spent asleep, or nil if unavailable.
    func lastNightSleepDuration(completion: @escaping (TimeInterval?) -> Void) {
        // Sleep category type
        guard let sleepType = HKObjectType.categoryType(forIdentifier: .sleepAnalysis) else {
            completion(nil)
            return
        }
        // Define a noon-to-noon window to capture overnight sleep
        let calendar = Calendar.current
        // Yesterday at 12:00 PM to today at 12:00 PM
        guard let end = calendar.date(bySettingHour: 12, minute: 0, second: 0, of: Date()),
              let start = calendar.date(byAdding: .day, value: -1, to: end) else {
            completion(nil)
            return
        }
        // Build a predicate for the window
        let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: .strictStartDate)
        // Sort newest to oldest (not strictly necessary for summing)
        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(sampleType: sleepType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: [sort]) { _, results, _ in
            // No sleep records in the window
            guard let samples = results as? [HKCategorySample], !samples.isEmpty else {
                completion(nil)
                return
            }
            var total: TimeInterval = 0
            // Accumulate only 'asleep' segments
            for s in samples {
                let val = s.value
                if #available(iOS 16.0, *) {
                    // On iOS 16+, count core/deep/REM states
                    if let state = HKCategoryValueSleepAnalysis(rawValue: val) {
                        switch state {
                        case .asleepCore, .asleepDeep, .asleepREM:
                            total += s.endDate.timeIntervalSince(s.startDate)
                        default:
                            break
                        }
                    }
                } else {
                    // On earlier iOS, count .asleep value
                    if val == HKCategoryValueSleepAnalysis.asleep.rawValue {
                        total += s.endDate.timeIntervalSince(s.startDate)
                    }
                }
            }
            // Return total duration if any; otherwise nil
            completion(total > 0 ? total : nil)
        }
        // Execute the query
        healthStore.execute(query)
    }
}

