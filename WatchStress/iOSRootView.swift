import SwiftUI
import Foundation

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

    // Scoring engine and state
    private let scoreEngine = ScoreEngine()
    @State private var computedScore: Double? = nil
    @State private var confidence: ScoreEngine.Confidence = .low
    @State private var driverReasons: [ScoreEngine.DriverContribution] = []

    var body: some View {
        ZStack {
            // Lightweight background to match brand tone
            Color.yellow.opacity(0.18).ignoresSafeArea()

            VStack(spacing: 16) {
                // Show computed score if available; otherwise allow manual slider simulation
                if let s = computedScore {
                    StressRingView(score: s)
                        .animation(.easeInOut(duration: 0.6), value: s)
                    Text("Confidence: \(confidence.rawValue)")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .accessibilityLabel("Confidence \(confidence.rawValue)")
                    // Debug: show computed score under the ring
                    Text("Computed: \(Int(s.rounded()))")
                        .font(.caption)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                        .accessibilityLabel("Computed score \(Int(s.rounded()))")
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

                Button {
                    withAnimation {
                        computeScoreFromVitals()
                    }
                } label: {
                    Label("Compute Score", systemImage: "function")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .padding(.horizontal)

                if computedScore == nil {
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
                    await MainActor.run { computeScoreFromVitals() }
                }
            }
        }
        // Recompute score whenever any of the core strings change
        .onChange(of: vitalsSnapshot, initial: true) { _, _ in
            computeScoreFromVitals()
        }
        // Present the vitals sheet
        .sheet(isPresented: $showVitalsSheet) {
            VitalsSheetView(vm: vitals)
                .presentationDragIndicator(.visible)
        }
        // Present the AI assistant
        .sheet(isPresented: $showAssistant) {
            AIAssistantView(vm: vitals, score: computedScore ?? score)
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
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
        let hrvSDNN: String
        let steps: String
        let exerciseMinutes: String
    }

    private var vitalsSnapshot: VitalsSnapshot {
        VitalsSnapshot(
            sleep: vitals.sleep,
            restingHR: vitals.restingHR,
            hrvSDNN: vitals.hrvSDNN,
            steps: vitals.steps,
            exerciseMinutes: vitals.exerciseMinutes
        )
    }

    /// Builds a FeatureSample from current vitals and updates the computed score using ScoreEngine.
    @MainActor private func computeScoreFromVitals() {
        // Build a sample from formatted strings
        let sample = ScoreEngine.sampleFromFormattedStrings(
            sleep: vitals.sleep,
            restingHR: vitals.restingHR,
            hrvSDNN: vitals.hrvSDNN,
            steps: vitals.steps,
            exerciseMinutes: vitals.exerciseMinutes
        )
        // Update baselines once per day (idempotent)
        scoreEngine.updateBaselinesIfNeeded(with: sample)
        // Compute score
        let result = scoreEngine.computeScore(sample: sample)
        withAnimation(.easeInOut(duration: 0.6)) {
            computedScore = result.score
            confidence = result.confidence
            driverReasons = result.drivers
        }
    }
}

