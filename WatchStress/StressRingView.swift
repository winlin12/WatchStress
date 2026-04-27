import SwiftUI

/// StressRingView renders a circular ring for a score in the range −100…+100.
///   +100 = fully filled red   (maximally stressed)
///      0 = empty ring         (neutral — at your personal baseline)
///   −100 = fully filled green (maximally calm)
/// Fill amount tracks abs(score)/100; color tracks sign.
struct StressRingView: View {
    let score: Double   // expected −100…+100

    private var clamped: Double { min(100, max(-100, score)) }
    private var progress: Double { abs(clamped) / 100.0 }
    private var ringColor: Color { clamped >= 0 ? .red : .green }

    var body: some View {
        ZStack {
            // Background track
            Circle()
                .stroke(.primary.opacity(0.10), lineWidth: 22)

            // Progress stroke — fills clockwise from top proportional to abs(score)
            Circle()
                .trim(from: 0, to: progress)
                .stroke(ringColor, style: StrokeStyle(lineWidth: 22, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(.easeInOut(duration: 0.4), value: progress)
                .animation(.easeInOut(duration: 0.4), value: ringColor == .red)

            // Numeric score and label in the center
            VStack(spacing: 6) {
                Text(clamped == 0 ? "0" : String(format: "%+d", Int(clamped.rounded())))
                    .font(.system(size: 56, weight: .bold))
                Text("Score")
                    .font(.headline)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(24)
        .accessibilityLabel("Score \(Int(clamped.rounded())), \(clamped > 0 ? "stressed" : clamped < 0 ? "calm" : "neutral")")
    }
}

