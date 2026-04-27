//  VitalsViewModel.swift
//  WatchStress
//
//  Bridges HealthKit values to SwiftUI-friendly strings for display.

import Foundation
import Combine
import HealthKit

/// VitalsViewModel bridges HealthKit values to SwiftUI-friendly strings.
/// It requests authorization (via HealthKitManager) and exposes formatted metrics
/// suitable for display in the UI. All published properties are updated on the main actor.

/// Observable object that loads and formats core/optional vitals.
@MainActor
final class VitalsViewModel: ObservableObject {
    @Published var authorized: Bool = false            // Whether HealthKit authorization succeeded

    @Published var sleep: String = "—"                 // Last night's total sleep (e.g., "7h 45m")
    @Published var restingHR: String = "—"             // Most recent resting heart rate (e.g., "54 bpm")
    @Published var heartRate: String = "—"             // Latest heart rate sample (e.g., "72 bpm")
    @Published var hrvSDNN: String = "—"               // Most recent HRV SDNN (e.g., "42 ms")
    @Published var bloodPressure: String = "—"         // Most recent BP (e.g., "120/80 mmHg") — optional/external
    @Published var wristTemperature: String = "—"      // Wrist/body temperature (°C) — optional
    @Published var steps: String = "—"                 // Today's step count
    @Published var exerciseMinutes: String = "—"       // Today's exercise minutes
    @Published var respirationRate: String = "—"       // Most recent respiration rate (brpm) — optional
    @Published var bloodOxygen: String = "—"           // Most recent SpO₂ (%) — optional
    @Published var uvExposure: String = "—"            // UV exposure (J/m²) today
    @Published var activeCalories: String = "—"        // Active energy burned today (kcal)
    @Published var distance: String = "—"              // Walking/running distance today (m)

    private let hk = HealthKitManager.shared
    private let baselineRefreshKey = "ScoreEngine.lastBaselineRefreshDay"

    /// Requests HealthKit authorization and, if successful, loads all metrics.
    /// Safe to call multiple times; authorization is only requested once by the system.
    func requestAuthorizationAndLoad() {
        hk.requestAuthorization { [weak self] success, _ in
            Task { @MainActor in
                self?.authorized = success
                if success {
                    await self?.reloadAll()
                }
            }
        }
    }

    /// Reloads all supported metrics in parallel and updates published strings.
    func reloadAll() async {
        await withTaskGroup(of: Void.self) { group in
            group.addTask { await self.loadSleep() }
            group.addTask { await self.loadRestingHeartRate() }
            group.addTask { await self.loadLatestHeartRate() }
            group.addTask { await self.loadHRV() }
            group.addTask { await self.loadBloodPressure() }
            group.addTask { await self.loadTemperature() }
            group.addTask { await self.loadSteps() }
            group.addTask { await self.loadExerciseMinutes() }
            group.addTask { await self.loadRespirationRate() }
            group.addTask { await self.loadBloodOxygen() }
            group.addTask { await self.loadUVExposure() }
            group.addTask { await self.loadActiveCalories() }
            group.addTask { await self.loadDistance() }
        }
        await refreshHistoricalBaselinesIfNeeded()
    }

    private struct StatsAccumulator {
        var count: Int = 0
        var mean: Double = 0
        var m2: Double = 0

        mutating func update(with x: Double) {
            count += 1
            let delta = x - mean
            mean += delta / Double(count)
            let delta2 = x - mean
            m2 += delta * delta2
        }

        var stdDev: Double {
            guard count > 1 else { return 0 }
            return max(m2 / Double(count - 1), 0).squareRoot()
        }
    }

    private func refreshHistoricalBaselinesIfNeeded(daysBack: Int = 7) async {
        guard authorized else { return }

        let defaults = UserDefaults.standard
        let cal = Calendar.current
        let currentDay = Self.dayKey(for: Date(), calendar: cal)
        if defaults.string(forKey: baselineRefreshKey) == currentDay {
            return
        }

        guard let start = cal.date(byAdding: .day, value: -daysBack, to: Date()) else { return }
        let end = Date()

        guard let hrType  = HKObjectType.quantityType(forIdentifier: .restingHeartRate),
              let hrvType = HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN)
        else { return }

        let bpmUnit = HKUnit.count().unitDivided(by: .minute())
        let msUnit  = HKUnit.secondUnit(with: .milli)

        async let hrSamples  = hk.quantitySamples(for: hrType,  start: start, end: end)
        async let hrvSamples = hk.quantitySamples(for: hrvType, start: start, end: end)
        let (hrAll, hrvAll) = await (hrSamples, hrvSamples)

        let hrVals  = hrAll.map  { $0.quantity.doubleValue(for: bpmUnit) }
        let hrvVals = hrvAll.map { $0.quantity.doubleValue(for: msUnit)  }

        var baselines: [ScoreEngine.Feature: ScoreEngine.BaselineStats] = [:]

        if let s = Self.makeBaselineStats(hrVals)  { baselines[.HR_mean_30] = s; baselines[.HR_mean_5] = s }
        if let s = Self.makeBaselineStats(hrvVals) { baselines[.HRV_30] = s;     baselines[.HRV_5] = s     }

        // HR_std_30 baseline: std of individual HR samples (proxy for within-window spread)
        if hrVals.count >= 2 {
            let mean = hrVals.reduce(0, +) / Double(hrVals.count)
            let std  = (hrVals.map { ($0-mean)*($0-mean) }.reduce(0,+) / Double(hrVals.count-1)).squareRoot()
            baselines[.HR_std_30] = ScoreEngine.BaselineStats(count: hrVals.count, mean: mean, stdDev: std)
        }

        // HR_slope_30 baseline: near-zero slope is expected at rest
        baselines[.HR_slope_30] = ScoreEngine.BaselineStats(count: 1, mean: 0.0, stdDev: 0.5)

        guard !baselines.isEmpty else { return }
        ScoreEngine.storeUserBaselines(baselines)
        defaults.set(currentDay, forKey: baselineRefreshKey)
    }

    private static func makeBaselineStats(_ vals: [Double]) -> ScoreEngine.BaselineStats? {
        guard vals.count >= 1 else { return nil }
        var acc = StatsAccumulator()
        vals.forEach { acc.update(with: $0) }
        return ScoreEngine.BaselineStats(count: acc.count, mean: acc.mean, stdDev: acc.stdDev)
    }

    // MARK: - Score sample computation

    /// Builds a FeatureSample by querying HR and HRV from HealthKit over the
    /// last 30-min and 5-min windows ending at `asOf`.
    /// Call this right before invoking ScoreEngine.computeScore(sample:).
    func computeScoreSample(asOf now: Date = Date()) async -> ScoreEngine.FeatureSample {
        guard let hrType  = HKObjectType.quantityType(forIdentifier: .heartRate),
              let hrvType = HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN)
        else { return ScoreEngine.FeatureSample() }

        let bpmUnit = HKUnit.count().unitDivided(by: .minute())
        let msUnit  = HKUnit.secondUnit(with: .milli)

        let w30start = now.addingTimeInterval(-30 * 60)
        let w5start  = now.addingTimeInterval(-5  * 60)

        async let hrSamples30  = hk.quantitySamples(for: hrType,  start: w30start, end: now)
        async let hrSamples5   = hk.quantitySamples(for: hrType,  start: w5start,  end: now)
        async let hrvSamples30 = hk.quantitySamples(for: hrvType, start: w30start, end: now)
        async let hrvSamples5  = hk.quantitySamples(for: hrvType, start: w5start,  end: now)

        let (hr30, hr5, hrv30, hrv5) = await (hrSamples30, hrSamples5, hrvSamples30, hrvSamples5)

        let bpms30 = hr30.map  { $0.quantity.doubleValue(for: bpmUnit) }
        let bpms5  = hr5.map   { $0.quantity.doubleValue(for: bpmUnit) }
        let times30 = hr30.map { $0.startDate.timeIntervalSince1970 }

        return ScoreEngine.FeatureSample(
            HR_mean_30:  Self.windowMean(bpms30),
            HR_std_30:   Self.windowStd(bpms30),
            HR_slope_30: Self.windowSlope(times: times30, values: bpms30),
            HRV_30:      hrvSamples30Last(hrv30, unit: msUnit),
            HR_mean_5:   Self.windowMean(bpms5),
            HRV_5:       hrvSamples30Last(hrv5,  unit: msUnit)
        )
    }

    private func hrvSamples30Last(_ samples: [HKQuantitySample], unit: HKUnit) -> Double? {
        samples.last.map { $0.quantity.doubleValue(for: unit) }
    }

    private static func windowMean(_ vals: [Double]) -> Double? {
        guard !vals.isEmpty else { return nil }
        return vals.reduce(0, +) / Double(vals.count)
    }

    private static func windowStd(_ vals: [Double]) -> Double? {
        guard vals.count >= 2 else { return nil }
        let mean = vals.reduce(0, +) / Double(vals.count)
        let variance = vals.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(vals.count - 1)
        return variance.squareRoot()
    }

    /// Ordinary least-squares slope in bpm/min.
    private static func windowSlope(times: [Double], values: [Double]) -> Double? {
        guard times.count >= 3, times.count == values.count else { return nil }
        let n = Double(times.count)
        let meanT = times.reduce(0, +) / n
        let meanV = values.reduce(0, +) / n
        let num = zip(times, values).map { ($0 - meanT) * ($1 - meanV) }.reduce(0, +)
        let den = times.map { ($0 - meanT) * ($0 - meanT) }.reduce(0, +)
        guard den > 0 else { return nil }
        return (num / den) * 60.0   // convert per-second → per-minute
    }
    private func loadSleep() async {
        await withCheckedContinuation { continuation in
            hk.lastNightSleepDuration { [weak self] duration in
                Task { @MainActor in
                    if let d = duration {
                        let hours = Int(d / 3600)
                        let minutes = Int((d.truncatingRemainder(dividingBy: 3600)) / 60)
                        self?.sleep = String(format: "%dh %dm", hours, minutes)
                    } else { self?.sleep = "—" }
                    continuation.resume()
                }
            }
        }
    }

    /// Loads the most recent resting heart rate (bpm).
    private func loadRestingHeartRate() async {
        guard let type = HKObjectType.quantityType(forIdentifier: .restingHeartRate) else { return }
        await withCheckedContinuation { continuation in
            hk.mostRecentQuantitySample(for: type) { [weak self] sample in
                Task { @MainActor in
                    if let s = sample {
                        let bpm = s.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
                        self?.restingHR = String(format: "%.0f bpm", bpm)
                    } else { self?.restingHR = "—" }
                    continuation.resume()
                }
            }
        }
    }

    /// Loads the latest heart rate sample (bpm).
    private func loadLatestHeartRate() async {
        guard let type = HKObjectType.quantityType(forIdentifier: .heartRate) else { return }
        await withCheckedContinuation { continuation in
            hk.mostRecentQuantitySample(for: type) { [weak self] sample in
                Task { @MainActor in
                    if let s = sample {
                        let bpm = s.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
                        self?.heartRate = String(format: "%.0f bpm", bpm)
                    } else { self?.heartRate = "—" }
                    continuation.resume()
                }
            }
        }
    }

    /// Loads the most recent HRV SDNN (milliseconds).
    private func loadHRV() async {
        guard let type = HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else { return }
        await withCheckedContinuation { continuation in
            hk.mostRecentQuantitySample(for: type) { [weak self] sample in
                Task { @MainActor in
                    if let s = sample {
                        let ms = s.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
                        self?.hrvSDNN = String(format: "%.0f ms", ms)
                    } else { self?.hrvSDNN = "—" }
                    continuation.resume()
                }
            }
        }
    }

    /// Loads the most recent systolic/diastolic blood pressure (mmHg) — optional/external.
    private func loadBloodPressure() async {
        guard let systolicType = HKObjectType.quantityType(forIdentifier: .bloodPressureSystolic),
              let diastolicType = HKObjectType.quantityType(forIdentifier: .bloodPressureDiastolic) else { return }

        // Blood pressure is stored as correlation samples; fetch most recent systolic and diastolic quantities separately and merge.
        await withTaskGroup(of: (Double?, Double?).self) { group in
            group.addTask { () -> (Double?, Double?) in
                await withCheckedContinuation { continuation in
                    self.hk.mostRecentQuantitySample(for: systolicType) { sample in
                        let sys = sample?.quantity.doubleValue(for: HKUnit.millimeterOfMercury())
                        continuation.resume(returning: (sys, nil))
                    }
                }
            }
            group.addTask { () -> (Double?, Double?) in
                await withCheckedContinuation { continuation in
                    self.hk.mostRecentQuantitySample(for: diastolicType) { sample in
                        let dia = sample?.quantity.doubleValue(for: HKUnit.millimeterOfMercury())
                        continuation.resume(returning: (nil, dia))
                    }
                }
            }

            var sys: Double?
            var dia: Double?
            for await pair in group {
                if let s = pair.0 { sys = s }
                if let d = pair.1 { dia = d }
            }
            await MainActor.run {
                if let s = sys, let d = dia {
                    self.bloodPressure = String(format: "%.0f/%.0f mmHg", s, d)
                } else {
                    self.bloodPressure = "—"
                }
            }
        }
    }

    /// Loads wrist/body temperature (°C) — optional.
    private func loadTemperature() async {
        var value: Double?
        if #available(iOS 16.0, *), let wristType = HKObjectType.quantityType(forIdentifier: .appleSleepingWristTemperature) {
            await withCheckedContinuation { continuation in
                hk.mostRecentQuantitySample(for: wristType) { sample in
                    value = sample?.quantity.doubleValue(for: HKUnit.degreeCelsius())
                    continuation.resume()
                }
            }
            if let v = value {
                await MainActor.run { self.wristTemperature = String(format: "%.1f ℃", v) }
                return
            }
        }
        if let bodyType = HKObjectType.quantityType(forIdentifier: .bodyTemperature) {
            await withCheckedContinuation { continuation in
                hk.mostRecentQuantitySample(for: bodyType) { sample in
                    value = sample?.quantity.doubleValue(for: HKUnit.degreeCelsius())
                    continuation.resume()
                }
            }
        }
        await MainActor.run {
            if let v = value { self.wristTemperature = String(format: "%.1f ℃", v) } else { self.wristTemperature = "—" }
        }
    }

    /// Loads today's step count.
    private func loadSteps() async {
        guard let type = HKObjectType.quantityType(forIdentifier: .stepCount) else { return }
        await withCheckedContinuation { continuation in
            hk.todaySum(for: type) { [weak self] stats in
                Task { @MainActor in
                    if let q = stats?.sumQuantity() {
                        let steps = q.doubleValue(for: HKUnit.count())
                        self?.steps = String(format: "%.0f", steps)
                    } else { self?.steps = "—" }
                    continuation.resume()
                }
            }
        }
    }

    /// Loads today's exercise minutes.
    private func loadExerciseMinutes() async {
        guard let type = HKObjectType.quantityType(forIdentifier: .appleExerciseTime) else { return }
        await withCheckedContinuation { continuation in
            hk.todaySum(for: type) { [weak self] stats in
                Task { @MainActor in
                    if let q = stats?.sumQuantity() {
                        let minutes = q.doubleValue(for: HKUnit.minute())
                        self?.exerciseMinutes = String(format: "%.0f min", minutes)
                    } else { self?.exerciseMinutes = "—" }
                    continuation.resume()
                }
            }
        }
    }

    /// Loads today's UV exposure index (discrete average).
    private func loadUVExposure() async {
        guard let type = HKObjectType.quantityType(forIdentifier: .uvExposure) else { return }
        await withCheckedContinuation { continuation in
            // uvExposure is a discrete type — must use .discreteAverage, not .cumulativeSum
            hk.todaySum(for: type, options: .discreteAverage) { [weak self] stats in
                Task { @MainActor in
                    if let q = stats?.averageQuantity() {
                        let val = q.doubleValue(for: HKUnit(from: "count/s"))
                        self?.uvExposure = String(format: "%.2f", val)
                    } else { self?.uvExposure = "—" }
                    continuation.resume()
                }
            }
        }
    }

    /// Loads today's active energy burned (kcal).
    private func loadActiveCalories() async {
        guard let type = HKObjectType.quantityType(forIdentifier: .activeEnergyBurned) else { return }
        await withCheckedContinuation { continuation in
            hk.todaySum(for: type) { [weak self] stats in
                Task { @MainActor in
                    if let q = stats?.sumQuantity() {
                        let kcal = q.doubleValue(for: HKUnit.kilocalorie())
                        self?.activeCalories = String(format: "%.0f kcal", kcal)
                    } else { self?.activeCalories = "—" }
                    continuation.resume()
                }
            }
        }
    }

    /// Loads today's walking + running distance (metres).
    private func loadDistance() async {
        guard let type = HKObjectType.quantityType(forIdentifier: .distanceWalkingRunning) else { return }
        await withCheckedContinuation { continuation in
            hk.todaySum(for: type) { [weak self] stats in
                Task { @MainActor in
                    if let q = stats?.sumQuantity() {
                        let metres = q.doubleValue(for: HKUnit.meter())
                        self?.distance = metres >= 1000
                            ? String(format: "%.2f km", metres / 1000)
                            : String(format: "%.0f m", metres)
                    } else { self?.distance = "—" }
                    continuation.resume()
                }
            }
        }
    }

    /// Loads the most recent respiration rate (breaths per minute) — optional.
    private func loadRespirationRate() async {
        guard let type = HKObjectType.quantityType(forIdentifier: .respiratoryRate) else { return }
        await withCheckedContinuation { continuation in
            hk.mostRecentQuantitySample(for: type) { [weak self] sample in
                Task { @MainActor in
                    if let s = sample {
                        let rr = s.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
                        self?.respirationRate = String(format: "%.0f bpm", rr)
                    } else { self?.respirationRate = "—" }
                    continuation.resume()
                }
            }
        }
    }

    /// Loads the most recent blood oxygen saturation (%) — optional.
    private func loadBloodOxygen() async {
        guard let type = HKObjectType.quantityType(forIdentifier: .oxygenSaturation) else { return }
        await withCheckedContinuation { continuation in
            hk.mostRecentQuantitySample(for: type) { [weak self] sample in
                Task { @MainActor in
                    if let s = sample {
                        let sat = s.quantity.doubleValue(for: HKUnit.percent()) * 100.0
                        self?.bloodOxygen = String(format: "%.0f%%", sat)
                    } else { self?.bloodOxygen = "—" }
                    continuation.resume()
                }
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
}
