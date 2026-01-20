//  VitalsViewModel.swift
//  WatchStress
//
//  Bridges HealthKit values to SwiftUI-friendly strings for display.

import Foundation
import Combine
import HealthKit

@MainActor
final class VitalsViewModel: ObservableObject {
    @Published var authorized: Bool = false

    @Published var sleep: String = "—"
    @Published var restingHR: String = "—"
    @Published var heartRate: String = "—"
    @Published var hrvSDNN: String = "—"
    @Published var bloodPressure: String = "—"
    @Published var wristTemperature: String = "—"
    @Published var steps: String = "—"
    @Published var exerciseMinutes: String = "—"
    @Published var respirationRate: String = "—"
    @Published var bloodOxygen: String = "—"

    private let hk = HealthKitManager.shared

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
        }
    }

    // MARK: - Individual loaders

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
}
