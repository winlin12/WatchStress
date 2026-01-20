//  HealthKitManager.swift
//  WatchStress
//
//  Centralizes HealthKit authorization and queries for vitals used in the app.

import Foundation
import HealthKit

final class HealthKitManager {
    static let shared = HealthKitManager()
    let healthStore = HKHealthStore()

    private init() {}

    // MARK: - Authorization
    func requestAuthorization(completion: @escaping (Bool, Error?) -> Void) {
        guard HKHealthStore.isHealthDataAvailable() else {
            completion(false, NSError(domain: "HealthKit", code: 1, userInfo: [NSLocalizedDescriptionKey: "Health data not available on this device."]))
            return
        }

        var readTypes: Set<HKObjectType> = []

        // Quantity types
        let quantityIdentifiers: [HKQuantityTypeIdentifier] = [
            .heartRate,
            .restingHeartRate,
            .heartRateVariabilitySDNN,
            .bloodPressureSystolic,
            .bloodPressureDiastolic,
            .respiratoryRate,
            .oxygenSaturation,
            .stepCount,
            .appleExerciseTime
        ]

        for id in quantityIdentifiers {
            if let t = HKObjectType.quantityType(forIdentifier: id) { readTypes.insert(t) }
        }

        // Wrist temperature (available on supported devices)
        if #available(iOS 16.0, *) {
            if let wristTemp = HKObjectType.quantityType(forIdentifier: .appleSleepingWristTemperature) {
                readTypes.insert(wristTemp)
            }
        }
        // Fallback to body temperature if wrist temperature isn't available
        if let bodyTemp = HKObjectType.quantityType(forIdentifier: .bodyTemperature) {
            readTypes.insert(bodyTemp)
        }

        // Category types
        if let sleepType = HKObjectType.categoryType(forIdentifier: .sleepAnalysis) {
            readTypes.insert(sleepType)
        }

        healthStore.requestAuthorization(toShare: nil, read: readTypes) { success, error in
            completion(success, error)
        }
    }

    // MARK: - Queries

    func mostRecentQuantitySample(for type: HKQuantityType, completion: @escaping (HKQuantitySample?) -> Void) {
        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(sampleType: type, predicate: nil, limit: 1, sortDescriptors: [sort]) { _, results, _ in
            completion(results?.first as? HKQuantitySample)
        }
        healthStore.execute(query)
    }

    func todaySum(for type: HKQuantityType, options: HKStatisticsOptions = .cumulativeSum, completion: @escaping (HKStatistics?) -> Void) {
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: Date())
        let predicate = HKQuery.predicateForSamples(withStart: startOfDay, end: Date(), options: .strictStartDate)
        let query = HKStatisticsQuery(quantityType: type, quantitySamplePredicate: predicate, options: options) { _, stats, _ in
            completion(stats)
        }
        healthStore.execute(query)
    }

    func lastNightSleepDuration(completion: @escaping (TimeInterval?) -> Void) {
        guard let sleepType = HKObjectType.categoryType(forIdentifier: .sleepAnalysis) else {
            completion(nil)
            return
        }
        let calendar = Calendar.current
        let now = Date()
        guard let end = calendar.date(bySettingHour: 12, minute: 0, second: 0, of: now),
              let start = calendar.date(byAdding: .day, value: -1, to: end) else {
            completion(nil)
            return
        }
        let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: .strictStartDate)
        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(sampleType: sleepType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: [sort]) { _, results, _ in
            guard let samples = results as? [HKCategorySample], !samples.isEmpty else {
                completion(nil)
                return
            }
            var total: TimeInterval = 0
            for s in samples {
                let val = s.value
                if #available(iOS 16.0, *) {
                    if let state = HKCategoryValueSleepAnalysis(rawValue: val) {
                        switch state {
                        case .asleepCore, .asleepDeep, .asleepREM:
                            total += s.endDate.timeIntervalSince(s.startDate)
                        default:
                            break
                        }
                    }
                } else {
                    if val == HKCategoryValueSleepAnalysis.asleep.rawValue {
                        total += s.endDate.timeIntervalSince(s.startDate)
                    }
                }
            }
            completion(total > 0 ? total : nil)
        }
        healthStore.execute(query)
    }
}
