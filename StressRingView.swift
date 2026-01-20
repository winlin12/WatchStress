import SwiftUI

struct StressRingView: View {
    let score: Double   // expected 0...100

    private var clamped: Double { min(100, max(0, score)) }
    private var progress: Double { clamped / 100.0 }

    // Simple red -> green gradient by interpolation (no fancy color spaces)
    private var ringColor: Color {
        let t = progress
        return Color(red: 1.0 - t, green: t, blue: 0.0)
    }

    var body: some View {
        ZStack {
            Circle()
                .stroke(.primary.opacity(0.10), lineWidth: 22)

            Circle()
                .trim(from: 0, to: progress)
                .stroke(ringColor, style: StrokeStyle(lineWidth: 22, lineCap: .round))
                .rotationEffect(.degrees(-90))

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
