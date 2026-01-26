import SwiftUI
import Foundation
import HealthKit

/// iOSRootView displays the stress ring and entry points for viewing health vitals.
/// Presents a sheet with detailed vitals and includes a slider to simulate a score.
struct iOSRootView: View {
    /// Simulated stress score (0...100) used to drive the ring UI.
    @State private var score: Double = 72
    /// View model bridging HealthKit values for the sheet.
    @StateObject private var vitals = VitalsViewModel()
    /// Controls presentation of the vitals sheet.
    @State private var showVitalsSheet: Bool = false
    /// Controls presentation of the AI assistant sheet.
    @State private var showAssistant: Bool = false
    /// Controls presentation of the score details sheet.
    @State private var showScoreDetails: Bool = false
    /// Controls presentation of the debug tuning sheet.
    @State private var showDebugSheet: Bool = false
    /// Controls presentation of the settings sheet.
    @State private var showSettings: Bool = false

    // Logging status/alerts
    @State private var logAlert: LogAlert? = nil

    // Debug override controls
    @State private var useDebugOverrides: Bool = false
    @State private var debugHeartRate: Double = 70
    @State private var debugHRV: Double = 60
    @State private var debugWristTemp: Double = 33.5
    @State private var smoothingTask: Task<Void, Never>? = nil
    @State private var scheduleTask: Task<Void, Never>? = nil

    // Scheduled logging configuration
    @AppStorage("logScheduleEnabled") private var logScheduleEnabled: Bool = false
    @AppStorage("logScheduleSlot1Enabled") private var logScheduleSlot1Enabled: Bool = true
    @AppStorage("logScheduleSlot2Enabled") private var logScheduleSlot2Enabled: Bool = false
    @AppStorage("logScheduleSlot3Enabled") private var logScheduleSlot3Enabled: Bool = false
    @AppStorage("logScheduleSlot1Minutes") private var logScheduleSlot1Minutes: Int = 9 * 60
    @AppStorage("logScheduleSlot2Minutes") private var logScheduleSlot2Minutes: Int = 14 * 60
    @AppStorage("logScheduleSlot3Minutes") private var logScheduleSlot3Minutes: Int = 20 * 60
    @AppStorage("logScheduleSkipSleep") private var logScheduleSkipSleep: Bool = true
    @AppStorage("logScheduleSkipExercise") private var logScheduleSkipExercise: Bool = true

    // Scoring engine and state
    private let scoreEngine: ScoreEngine? = ScoreEngine()
    @State private var scoreDetails: ScoreEngine.ScoreResult? = nil

    var body: some View {
        ZStack {
            // Lightweight background to match brand tone
            Color.yellow.opacity(0.18).ignoresSafeArea()

            VStack(spacing: 16) {
                SettingsButtonRow(showSettings: $showSettings)

                // Show computed score if available; otherwise allow manual slider simulation
                if let result = scoreDetails {
                    StressRingView(score: result.score)
                        .animation(.easeInOut(duration: 0.6), value: result.score)
                    Text("Confidence: \(result.confidence.rawValue)")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .accessibilityLabel("Confidence \(result.confidence.rawValue)")
                    if useDebugOverrides {
                        Text("Debug overrides enabled")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    // Debug: show computed score under the ring
                    Text("Computed: \(Int(result.score.rounded()))")
                        .font(.caption)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                        .accessibilityLabel("Computed score \(Int(result.score.rounded()))")
                } else {
                    StressRingView(score: score)
                        .animation(.easeInOut(duration: 0.6), value: score)
                    Text("Confidence: Low (estimating)")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    // Debug: no computed score yet
                    Text("Computed: —")
                        .font(.caption)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                        .accessibilityLabel("Computed score unavailable")
                }

                // Entry point to view detailed vitals
                Button {
                    showVitalsSheet = true
                } label: {
                    Label("Show Vitals", systemImage: "heart.text.square")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .padding(.horizontal)

                // Entry point to the on-device Apple Intelligence assistant
                Button {
                    showAssistant = true
                } label: {
                    Label("Ask AI", systemImage: "sparkles")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .padding(.horizontal)

                if scoreDetails == nil {
                    // Quick way to simulate different scores during development
                    Slider(value: $score, in: 0...100, step: 1)
                        .padding(.horizontal)
                }

                Spacer(minLength: 8)
            }
            .padding(.top, 8)
        }
        .onAppear {
            // Ensure vitals authorization and initial load even if the sheet isn't opened
            if !vitals.authorized {
                vitals.requestAuthorizationAndLoad()
            } else {
                // If already authorized, reload and then compute
                Task {
                    await vitals.reloadAll()
                    await MainActor.run { triggerScoreUpdate() }
                }
            }
            startScheduleLoop()
        }
        // Recompute score whenever any HealthKit values or debug overrides change
        .onChange(of: vitalsSnapshot, initial: true) { _, _ in
            triggerScoreUpdate()
        }
        .onChange(of: debugSnapshot) { _, _ in
            triggerScoreUpdate()
        }
        .onChange(of: scheduleSnapshot) { _, _ in
            startScheduleLoop()
            Task { await checkScheduledLogging() }
        }
        .onDisappear {
            scheduleTask?.cancel()
        }
        // Present the vitals sheet
        .sheet(isPresented: $showVitalsSheet) {
            VitalsSheetView(vm: vitals)
                .presentationDragIndicator(.visible)
        }
        // Present the AI assistant
        .sheet(isPresented: $showAssistant) {
            AIAssistantView(vm: vitals, score: scoreDetails?.score ?? score)
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
        .sheet(isPresented: $showScoreDetails) {
            ScoreDetailsView(result: scoreDetails)
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
        .sheet(isPresented: $showDebugSheet) {
            DebugTuningView(
                useDebugOverrides: $useDebugOverrides,
                debugHeartRate: $debugHeartRate,
                debugHRV: $debugHRV,
                debugWristTemp: $debugWristTemp,
                applyCurrentVitals: applyCurrentVitalsToDebug,
                applyRelaxedPreset: applyRelaxedPreset,
                applyStressedPreset: applyStressedPreset
            )
            .presentationDetents([.medium, .large])
            .presentationDragIndicator(.visible)
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(
                useDebugOverrides: useDebugOverrides,
                logScheduleEnabled: $logScheduleEnabled,
                logScheduleSlot1Enabled: $logScheduleSlot1Enabled,
                logScheduleSlot2Enabled: $logScheduleSlot2Enabled,
                logScheduleSlot3Enabled: $logScheduleSlot3Enabled,
                logScheduleSlot1Minutes: $logScheduleSlot1Minutes,
                logScheduleSlot2Minutes: $logScheduleSlot2Minutes,
                logScheduleSlot3Minutes: $logScheduleSlot3Minutes,
                logScheduleSkipSleep: $logScheduleSkipSleep,
                logScheduleSkipExercise: $logScheduleSkipExercise,
                computeScore: { triggerScoreUpdate() },
                openScoreDetails: { showScoreDetails = true },
                openDebugTuning: { showDebugSheet = true },
                saveLogEntry: { await saveLogEntry(note: "Manual log") },
                deleteAllLogs: { await deleteAllLogs() }
            )
            .presentationDetents([.medium, .large])
            .presentationDragIndicator(.visible)
        }
        .alert(item: $logAlert) { alert in
            Alert(
                title: Text(alert.title),
                message: Text(alert.message),
                dismissButton: .default(Text("OK"))
            )
        }
    }
}

/// Simple metric card with a title and monospaced value.
/// Used across the app to display health metrics consistently.
private struct MetricCard: View {
    let title: String
    let value: String

    var body: some View {
        // Title on the left, monospaced value on the right
        HStack {
            Text(title)
            Spacer()
            Text(value).monospacedDigit()
        }
        // Subtle material background with rounded corners
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }
}

private struct LogAlert: Identifiable {
    let id = UUID()
    let title: String
    let message: String
}

/// ScoreDetailsView presents a detailed breakdown of the score calculation.
private struct ScoreDetailsView: View {
    let result: ScoreEngine.ScoreResult?

    var body: some View {
        NavigationStack {
            ScrollView {
                if let result {
                    VStack(spacing: 14) {
                        GroupBox("Summary") {
                            VStack(alignment: .leading, spacing: 8) {
                                SummaryRow(label: "Score (clamped)", value: format(result.score, decimals: 1))
                                SummaryRow(label: "Raw score", value: format(result.rawScore, decimals: 2))
                                SummaryRow(label: "Stress index (raw)", value: format(result.rawStressIndex, decimals: 3))
                                SummaryRow(label: "Stress index (smoothed)", value: format(result.stressIndex, decimals: 3))
                                SummaryRow(label: "Confidence", value: result.confidence.rawValue)
                                SummaryRow(label: "k", value: format(result.k, decimals: 2))
                                SummaryRow(label: "Lambda", value: format(result.lambda, decimals: 2))
                                SummaryRow(label: "Bias", value: format(result.bias, decimals: 3))
                            }
                        }

                        GroupBox("Formula") {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("z = clip((x - mu) / (sigma + 1e-6), -3, +3)")
                                Text("rawStress = bias + sum(w * z)")
                                Text("stressIndex = lambda * prev + (1 - lambda) * rawStress")
                                Text("s = stressIndex")
                                Text("score = 80 - 80 * tanh(k * s)        for s >= 0")
                                Text("score = 80 + 20 * (1 - exp(k * s))     for s < 0")
                                Text("sum(w * z) = \(format(sumContribution(for: result), decimals: 3))")
                                Text("rawStress = \(format(result.rawStressIndex, decimals: 3))")
                                Text("score(s) = \(format(result.rawScore, decimals: 2))\(isClamped(result) ? " (clamped)" : "")")
                            }
                            .font(.subheadline.monospaced())
                        }

                        GroupBox("Feature Contributions") {
                            if result.drivers.isEmpty {
                                Text("No usable features available. Compute again after HealthKit updates.")
                                    .foregroundStyle(.secondary)
                            } else {
                                VStack(spacing: 10) {
                                    ForEach(result.drivers, id: \.feature) { driver in
                                        VStack(alignment: .leading, spacing: 6) {
                                            Text(featureName(for: driver.feature))
                                                .font(.headline)
                                            Text("Value: \(format(driver.value, decimals: 2)) \(featureUnit(for: driver.feature))")
                                            Text("Baseline: mu \(format(driver.mean, decimals: 2)) \(featureUnit(for: driver.feature)), sigma \(format(driver.std, decimals: 2))")
                                            Text("z: \(formatSigned(driver.z, decimals: 2))")
                                            Text("Weight: \(formatSigned(driver.weight, decimals: 3))")
                                            Text("Score delta: \(formatSigned(driver.scoreDelta, decimals: 2))")
                                                .foregroundStyle(driver.scoreDelta >= 0 ? .green : .red)
                                            Text("Blend a: \(format(driver.blendA, decimals: 2))")
                                        }
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .padding()
                                        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
                                    }
                                }
                            }
                        }

                        if !missingFeatures(from: result).isEmpty {
                            GroupBox("Not Included (missing data)") {
                                Text(missingFeatures(from: result).map(featureName(for:)).joined(separator: ", "))
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    .padding(.horizontal)
                    .padding(.top)
                } else {
                    VStack(spacing: 10) {
                        Text("No score available yet.")
                            .font(.headline)
                        Text("Tap Compute Score or wait for new vitals, then open details again.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .padding()
                }
            }
            .navigationTitle("Score Details")
        }
    }

    private func format(_ value: Double, decimals: Int) -> String {
        String(format: "%.\(decimals)f", value)
    }

    private func formatSigned(_ value: Double, decimals: Int) -> String {
        String(format: "%+.\(decimals)f", value)
    }

    private func sumContribution(for result: ScoreEngine.ScoreResult) -> Double {
        result.rawStressIndex - result.bias
    }

    private func isClamped(_ result: ScoreEngine.ScoreResult) -> Bool {
        abs(result.rawScore - result.score) > 0.001
    }

    private func featureName(for feature: ScoreEngine.Feature) -> String {
        switch feature {
        case .hrMeanBPM: return "Heart Rate"
        case .hrvSDNNms: return "HRV (SDNN)"
        case .wristTempC: return "Wrist Temperature"
        }
    }

    private func featureUnit(for feature: ScoreEngine.Feature) -> String {
        switch feature {
        case .hrMeanBPM: return "bpm"
        case .hrvSDNNms: return "ms"
        case .wristTempC: return "C"
        }
    }

    private func missingFeatures(from result: ScoreEngine.ScoreResult) -> [ScoreEngine.Feature] {
        let used = Set(result.drivers.map { $0.feature })
        return ScoreEngine.Feature.allCases.filter { !used.contains($0) }
    }
}

private struct SummaryRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
            Spacer()
            Text(value)
                .monospacedDigit()
        }
    }
}

private struct DebugTuningView: View {
    @Binding var useDebugOverrides: Bool
    @Binding var debugHeartRate: Double
    @Binding var debugHRV: Double
    @Binding var debugWristTemp: Double
    let applyCurrentVitals: () -> Void
    let applyRelaxedPreset: () -> Void
    let applyStressedPreset: () -> Void

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Toggle("Use debug values for scoring", isOn: $useDebugOverrides)
                    Text("Overrides only affect the score calculation. HealthKit data is unchanged.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Presets") {
                    Button("Use current vitals") { applyCurrentVitals() }
                    Button("Relaxed preset") { applyRelaxedPreset() }
                    Button("Stressed preset") { applyStressedPreset() }
                }

                Section("Overrides") {
                    HStack {
                        Text("Heart Rate")
                        Spacer()
                        Text("\(Int(debugHeartRate)) bpm")
                            .monospacedDigit()
                    }
                    Slider(value: $debugHeartRate, in: 40...140, step: 1)

                    HStack {
                        Text("HRV (SDNN)")
                        Spacer()
                        Text("\(Int(debugHRV)) ms")
                            .monospacedDigit()
                    }
                    Slider(value: $debugHRV, in: 10...150, step: 1)

                    HStack {
                        Text("Wrist Temperature")
                        Spacer()
                        Text(String(format: "%.1f C", debugWristTemp))
                            .monospacedDigit()
                    }
                    Slider(value: $debugWristTemp, in: 30.0...36.5, step: 0.1)
                }
                .disabled(!useDebugOverrides)
            }
            .navigationTitle("Debug Tuning")
        }
    }
}

/// VitalsSheetView lists core and optional vitals sourced from HealthKit.
/// Displays pre-formatted strings from VitalsViewModel with a manual reload action.
private struct VitalsSheetView: View {
    @ObservedObject var vm: VitalsViewModel

    var body: some View {
        // Embedded in a NavigationStack for title/toolbar
        NavigationStack {
            ScrollView {
                // Cards for each metric; values come pre-formatted from the view model
                VStack(spacing: 10) {
                    MetricCard(title: "Sleep (last night)", value: vm.sleep)
                    MetricCard(title: "Heart Rate (latest)", value: vm.heartRate)
                    MetricCard(title: "Resting HR", value: vm.restingHR)
                    MetricCard(title: "HRV (SDNN)", value: vm.hrvSDNN)
                    MetricCard(title: "Blood Pressure", value: vm.bloodPressure)
                    MetricCard(title: "Wrist Temperature", value: vm.wristTemperature)
                    MetricCard(title: "Respiration Rate", value: vm.respirationRate)
                    MetricCard(title: "Blood Oxygen", value: vm.bloodOxygen)
                    MetricCard(title: "Steps (today)", value: vm.steps)
                    MetricCard(title: "Exercise Minutes", value: vm.exerciseMinutes)
                }
                .padding(.horizontal)
                .padding(.top)
            }
            .navigationTitle("Vitals")
            // Toolbar provides a reload button for manual refresh
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await vm.reloadAll() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .accessibilityLabel("Reload")
                }
            }
            // Request authorization on first display; otherwise refresh values
            .onAppear {
                if !vm.authorized {
                    vm.requestAuthorizationAndLoad()
                } else {
                    Task { await vm.reloadAll() }
                }
            }
        }
    }
}

extension iOSRootView {
    /// Equatable snapshot of the vitals we care about for scoring.
    private struct VitalsSnapshot: Equatable {
        let sleep: String
        let restingHR: String
        let heartRate: String
        let hrvSDNN: String
        let bloodPressure: String
        let wristTemperature: String
        let steps: String
        let exerciseMinutes: String
        let respirationRate: String
        let bloodOxygen: String
    }

    private var vitalsSnapshot: VitalsSnapshot {
        VitalsSnapshot(
            sleep: vitals.sleep,
            restingHR: vitals.restingHR,
            heartRate: vitals.heartRate,
            hrvSDNN: vitals.hrvSDNN,
            bloodPressure: vitals.bloodPressure,
            wristTemperature: vitals.wristTemperature,
            steps: vitals.steps,
            exerciseMinutes: vitals.exerciseMinutes,
            respirationRate: vitals.respirationRate,
            bloodOxygen: vitals.bloodOxygen
        )
    }

    private struct DebugSnapshot: Equatable {
        let useDebugOverrides: Bool
        let heartRate: Double
        let hrv: Double
        let wristTemp: Double
    }

    private var debugSnapshot: DebugSnapshot {
        DebugSnapshot(
            useDebugOverrides: useDebugOverrides,
            heartRate: debugHeartRate,
            hrv: debugHRV,
            wristTemp: debugWristTemp
        )
    }

    private struct ScheduleSnapshot: Equatable {
        let enabled: Bool
        let slot1Enabled: Bool
        let slot2Enabled: Bool
        let slot3Enabled: Bool
        let slot1Minutes: Int
        let slot2Minutes: Int
        let slot3Minutes: Int
        let skipSleep: Bool
        let skipExercise: Bool
    }

    private var scheduleSnapshot: ScheduleSnapshot {
        ScheduleSnapshot(
            enabled: logScheduleEnabled,
            slot1Enabled: logScheduleSlot1Enabled,
            slot2Enabled: logScheduleSlot2Enabled,
            slot3Enabled: logScheduleSlot3Enabled,
            slot1Minutes: logScheduleSlot1Minutes,
            slot2Minutes: logScheduleSlot2Minutes,
            slot3Minutes: logScheduleSlot3Minutes,
            skipSleep: logScheduleSkipSleep,
            skipExercise: logScheduleSkipExercise
        )
    }

    /// Builds a FeatureSample from current vitals and updates the computed score using ScoreEngine.
    @MainActor private func computeScoreFromVitals(animate: Bool) {
        let sample: ScoreEngine.FeatureSample
        if useDebugOverrides {
            sample = ScoreEngine.sample(
                hrMeanBPM: debugHeartRate,
                hrvSDNNms: debugHRV,
                wristTempC: debugWristTemp
            )
        } else {
            sample = ScoreEngine.sampleFromFormattedStrings(
                heartRate: vitals.heartRate,
                hrvSDNN: vitals.hrvSDNN,
                wristTemperature: vitals.wristTemperature
            )
        }
        guard let scoreEngine else {
            scoreDetails = nil
            return
        }
        let result = scoreEngine.computeScore(sample: sample)
        if !useDebugOverrides {
            scoreEngine.updateUserBaselinesIfNeeded(with: sample)
        }

        if animate {
            withAnimation(.easeInOut(duration: 0.6)) {
                scoreDetails = result
            }
        } else {
            scoreDetails = result
        }
    }

    @MainActor private func triggerScoreUpdate() {
        guard scoreEngine != nil else {
            scoreDetails = nil
            return
        }
        computeScoreFromVitals(animate: true)
        startSmoothingLoop()
    }

    @MainActor private func startSmoothingLoop() {
        smoothingTask?.cancel()
        smoothingTask = Task { @MainActor in
            var lastScore = scoreDetails?.score
            for _ in 0..<30 {
                try? await Task.sleep(nanoseconds: 200_000_000)
                if Task.isCancelled { break }
                computeScoreFromVitals(animate: false)
                let currentScore = scoreDetails?.score
                if let lastScore, let currentScore, abs(currentScore - lastScore) < 0.3 {
                    break
                }
                lastScore = currentScore
            }
        }
    }

    @MainActor private func startScheduleLoop() {
        scheduleTask?.cancel()
        guard logScheduleEnabled else { return }
        scheduleTask = Task {
            await checkScheduledLogging()
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 60_000_000_000)
                if Task.isCancelled { break }
                await checkScheduledLogging()
            }
        }
    }

    private func checkScheduledLogging() async {
        let snapshot = await MainActor.run { scheduleSnapshot }
        guard snapshot.enabled else { return }

        let now = Date()
        let minutesNow = minutesSinceMidnight(for: now)
        let todayKey = dayKey(for: now)
        let slots: [(index: Int, enabled: Bool, minutes: Int)] = [
            (1, snapshot.slot1Enabled, snapshot.slot1Minutes),
            (2, snapshot.slot2Enabled, snapshot.slot2Minutes),
            (3, snapshot.slot3Enabled, snapshot.slot3Minutes)
        ]

        for slot in slots {
            guard slot.enabled else { continue }
            guard isWithinWindow(nowMinutes: minutesNow, targetMinutes: slot.minutes, windowMinutes: 5) else { continue }
            guard !hasLoggedToday(slotIndex: slot.index, dayKey: todayKey) else { continue }

            if snapshot.skipSleep || snapshot.skipExercise {
                let skip = await shouldSkipScheduledLog(skipSleep: snapshot.skipSleep, skipExercise: snapshot.skipExercise)
                if skip { continue }
            }

            let error = await saveLogEntry(note: "Scheduled log slot \(slot.index)")
            if let error {
                await MainActor.run {
                    logAlert = LogAlert(title: "Scheduled Log Failed", message: error)
                }
            } else {
                markLoggedToday(slotIndex: slot.index, dayKey: todayKey)
            }
        }
    }

    private func shouldSkipScheduledLog(skipSleep: Bool, skipExercise: Bool) async -> Bool {
        if skipSleep, await isSleepingNow() {
            return true
        }
        if skipExercise, await isExercisingNow() {
            return true
        }
        return false
    }

    private func isSleepingNow() async -> Bool {
        let authorized = await MainActor.run { vitals.authorized }
        guard authorized else { return false }
        guard let sleepType = HKObjectType.categoryType(forIdentifier: .sleepAnalysis) else { return false }

        let now = Date()
        let start = Calendar.current.date(byAdding: .hour, value: -12, to: now) ?? now
        let samples = await HealthKitManager.shared.categorySamples(for: sleepType, start: start, end: now)

        for sample in samples {
            guard now >= sample.startDate && now <= sample.endDate else { continue }
            if #available(iOS 16.0, *) {
                if let state = HKCategoryValueSleepAnalysis(rawValue: sample.value) {
                    switch state {
                    case .asleepCore, .asleepDeep, .asleepREM:
                        return true
                    default:
                        break
                    }
                }
            } else {
                if sample.value == HKCategoryValueSleepAnalysis.asleep.rawValue {
                    return true
                }
            }
        }
        return false
    }

    private func isExercisingNow() async -> Bool {
        let authorized = await MainActor.run { vitals.authorized }
        guard authorized else { return false }
        guard let type = HKObjectType.quantityType(forIdentifier: .appleExerciseTime) else { return false }

        let end = Date()
        let start = Calendar.current.date(byAdding: .minute, value: -20, to: end) ?? end
        let samples = await HealthKitManager.shared.quantitySamples(for: type, start: start, end: end)
        let totalMinutes = samples.reduce(0.0) { sum, sample in
            sum + sample.quantity.doubleValue(for: HKUnit.minute())
        }
        return totalMinutes > 0.1
    }

    private func saveLogEntry(note: String?) async -> String? {
        let entry = await MainActor.run {
            computeScoreFromVitals(animate: false)
            return buildLogEntry(note: note)
        }
        do {
            try await StressLogStore.shared.append(entry: entry)
            return nil
        } catch {
            return error.localizedDescription
        }
    }

    private func deleteAllLogs() async -> String? {
        do {
            try await StressLogStore.shared.deleteAll()
            return nil
        } catch {
            return error.localizedDescription
        }
    }

    @MainActor private func buildLogEntry(note: String?) -> StressLogEntry {
        let (sys, dia) = parseBloodPressure(from: vitals.bloodPressure)
        var finalNote = note
        if useDebugOverrides {
            if let note = finalNote {
                finalNote = "\(note) (debug overrides)"
            } else {
                finalNote = "Debug overrides enabled"
            }
        }

        let hr = useDebugOverrides ? debugHeartRate : numeric(from: vitals.heartRate)
        let hrv = useDebugOverrides ? debugHRV : numeric(from: vitals.hrvSDNN)
        let temp = useDebugOverrides ? debugWristTemp : numeric(from: vitals.wristTemperature)
        let confidence = scoreDetails?.confidence.rawValue ?? "Unknown"

        return StressLogEntry(
            id: UUID(),
            timestamp: Date(),
            sleepHours: parseSleepHours(from: vitals.sleep),
            heartRate: hr,
            restingHR: numeric(from: vitals.restingHR),
            hrvSDNN: hrv,
            bloodPressureSystolic: sys,
            bloodPressureDiastolic: dia,
            wristTemperature: temp,
            respirationRate: numeric(from: vitals.respirationRate),
            bloodOxygen: numeric(from: vitals.bloodOxygen),
            steps: numeric(from: vitals.steps),
            exerciseMinutes: numeric(from: vitals.exerciseMinutes),
            score: scoreDetails?.score,
            confidence: confidence,
            selfReport: nil,
            note: finalNote
        )
    }

    private func parseSleepHours(from string: String) -> Double? {
        let numbers = string.split(whereSeparator: { !"0123456789.".contains($0) })
        guard !numbers.isEmpty else { return nil }
        let hours = Double(numbers[0]) ?? 0
        if numbers.count >= 2 {
            let minutes = Double(numbers[1]) ?? 0
            return hours + (minutes / 60.0)
        }
        return hours
    }

    private func parseBloodPressure(from string: String) -> (Double?, Double?) {
        let parts = string.split(separator: "/")
        guard parts.count >= 2 else { return (nil, nil) }
        let sys = numeric(from: String(parts[0]))
        let dia = numeric(from: String(parts[1]))
        return (sys, dia)
    }

    private func isWithinWindow(nowMinutes: Int, targetMinutes: Int, windowMinutes: Int) -> Bool {
        abs(nowMinutes - targetMinutes) <= windowMinutes
    }

    private func dayKey(for date: Date) -> String {
        let comps = Calendar.current.dateComponents([.year, .month, .day], from: date)
        let y = comps.year ?? 0
        let m = comps.month ?? 0
        let d = comps.day ?? 0
        return "\(y)-\(m)-\(d)"
    }

    private func minutesSinceMidnight(for date: Date) -> Int {
        let comps = Calendar.current.dateComponents([.hour, .minute], from: date)
        let h = comps.hour ?? 0
        let m = comps.minute ?? 0
        return max(0, min(1439, (h * 60) + m))
    }

    private func lastLogKey(for slotIndex: Int) -> String {
        "StressLog.lastLog.slot\(slotIndex)"
    }

    private func hasLoggedToday(slotIndex: Int, dayKey: String) -> Bool {
        let defaults = UserDefaults.standard
        return defaults.string(forKey: lastLogKey(for: slotIndex)) == dayKey
    }

    private func markLoggedToday(slotIndex: Int, dayKey: String) {
        let defaults = UserDefaults.standard
        defaults.set(dayKey, forKey: lastLogKey(for: slotIndex))
    }

    private func applyCurrentVitalsToDebug() {
        if let hr = numeric(from: vitals.heartRate) {
            debugHeartRate = hr
        }
        if let hrv = numeric(from: vitals.hrvSDNN) {
            debugHRV = hrv
        }
        if let temp = numeric(from: vitals.wristTemperature) {
            debugWristTemp = temp
        }
    }

    private func applyRelaxedPreset() {
        debugHeartRate = 60
        debugHRV = 70
        debugWristTemp = 34.0
    }

    private func applyStressedPreset() {
        debugHeartRate = 90
        debugHRV = 30
        debugWristTemp = 32.5
    }

    private func numeric(from string: String) -> Double? {
        let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
        let allowed = "0123456789.-"
        let filtered = trimmed.filter { allowed.contains($0) }
        guard !filtered.isEmpty else { return nil }
        return Double(filtered)
    }
}
