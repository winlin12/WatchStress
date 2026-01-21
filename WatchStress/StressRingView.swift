import SwiftUI

/// StressRingView renders a circular progress ring for a score between 0 and 100.
/// The ring color transitions from red (low) to green (high) and the numeric score
/// is displayed in the center for quick readability.
struct StressRingView: View {
    let score: Double   // expected 0...100

    // Clamp the input score into the valid 0...100 range
    private var clamped: Double { min(100, max(0, score)) }
    // Convert to 0.0...1.0 for drawing the ring
    private var progress: Double { clamped / 100.0 }

    // Simple red -> green gradient by interpolating RGB components
    private var ringColor: Color {
        let t = progress
        return Color(red: 1.0 - t, green: t, blue: 0.0)
    }

    var body: some View {
        ZStack {
            // Background track
            Circle()
                .stroke(.primary.opacity(0.10), lineWidth: 22)

            // Progress stroke, rotated so 0 starts at the top
            Circle()
                .trim(from: 0, to: progress)
                .stroke(ringColor, style: StrokeStyle(lineWidth: 22, lineCap: .round))
                .rotationEffect(.degrees(-90))

            // Numeric score and label in the center
            VStack(spacing: 6) {
                Text("\(Int(clamped.rounded()))")
                    .font(.system(size: 56, weight: .bold))
                Text("Score")
                    .font(.headline)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(24)
        .accessibilityLabel("Score \(Int(clamped)) out of 100")
    }
}

