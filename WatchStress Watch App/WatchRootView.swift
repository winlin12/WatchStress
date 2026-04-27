import SwiftUI

/// WatchRootView is the main entry for the watchOS app.
/// Shows the current stress score and a thumbs-up / thumbs-down feedback row.
/// Feedback triggers a Huber SGD step in ScoreEngine, keeping weights in sync
/// with the iOS app via the shared UserDefaults app group.
struct WatchRootView: View {

    // Score produced by the most recent HealthKit sample, in −100…+100.
    // Set externally (e.g. via WatchConnectivity) or computed locally if available.
    @State private var score: Double = 0
    @State private var pendingFeedbackSample: ScoreEngine.FeatureSample? = nil
    @State private var feedbackGiven: Bool = false
    @State private var feedbackCount: Int = 0

    private let scoreEngine = ScoreEngine()

    var body: some View {
        ScrollView {
            VStack(spacing: 10) {

                // ── Score display ──────────────────────────────────────
                ZStack {
                    Circle()
                        .stroke(scoreColor.opacity(0.25), lineWidth: 6)
                        .frame(width: 80, height: 80)
                    Circle()
                        .trim(from: 0, to: fillFraction)
                        .stroke(scoreColor, style: StrokeStyle(lineWidth: 6, lineCap: .round))
                        .frame(width: 80, height: 80)
                        .rotationEffect(.degrees(-90))
                        .animation(.easeInOut(duration: 0.5), value: fillFraction)
                    VStack(spacing: 1) {
                        Text(scoreLabel)
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(scoreColor)
                        Text(scoreEngine.isPersonalized ? "You" : "Pop.")
                            .font(.system(size: 9))
                            .foregroundStyle(.secondary)
                    }
                }

                // ── Feedback prompt ────────────────────────────────────
                if pendingFeedbackSample != nil {
                    if feedbackGiven {
                        Label("Updated", systemImage: "checkmark.circle.fill")
                            .font(.system(size: 11))
                            .foregroundStyle(.green)
                    } else {
                        VStack(spacing: 6) {
                            Text("Accurate?")
                                .font(.system(size: 11))
                                .foregroundStyle(.secondary)
                            HStack(spacing: 10) {
                                Button {
                                    submitFeedback(wasStressed: false)
                                } label: {
                                    Image(systemName: "hand.thumbsup.fill")
                                        .foregroundStyle(.green)
                                }
                                .accessibilityLabel("I was calm")

                                Button {
                                    submitFeedback(wasStressed: true)
                                } label: {
                                    Image(systemName: "hand.thumbsdown.fill")
                                        .foregroundStyle(.orange)
                                }
                                .accessibilityLabel("I was stressed")
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                } else {
                    // Placeholder so layout doesn't jump on first score
                    Button("Check In") {
                        // Trigger a score fetch — placeholder for WatchConnectivity
                        withAnimation { feedbackGiven = false }
                    }
                    .buttonStyle(.bordered)
                    .font(.system(size: 12))
                }

                if feedbackCount > 0 {
                    Text("\(feedbackCount) correction\(feedbackCount == 1 ? "" : "s")")
                        .font(.system(size: 9))
                        .foregroundStyle(.tertiary)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
        }
        .onAppear {
            feedbackCount = scoreEngine.feedbackCount
        }
    }

    // MARK: - Helpers

    private var fillFraction: Double {
        // score is in -3…+3 (raw) or -100…+100 (display). Accept either.
        let normalised = abs(score) <= 3.0 ? score / 3.0 : score / 100.0
        return max(0.05, min(1.0, (normalised + 1.0) / 2.0))
    }

    private var scoreLabel: String {
        let display = abs(score) <= 3.0 ? Int((score * 100.0 / 3.0).rounded()) : Int(score.rounded())
        return (display >= 0 ? "+" : "") + "\(display)"
    }

    private var scoreColor: Color {
        let normalised = abs(score) <= 3.0 ? score / 3.0 : score / 100.0
        if normalised > 0.2  { return .orange }
        if normalised < -0.2 { return .green  }
        return .yellow
    }

    private func submitFeedback(wasStressed: Bool) {
        guard let sample = pendingFeedbackSample else { return }
        scoreEngine.recordFeedback(sample: sample, wasStressed: wasStressed)
        feedbackCount = scoreEngine.feedbackCount
        withAnimation { feedbackGiven = true }
    }
}

