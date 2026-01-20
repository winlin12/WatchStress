import SwiftUI

struct iOSRootView: View {
    @State private var score: Double = 72

    var body: some View {
        ZStack {
            Color.yellow.opacity(0.18).ignoresSafeArea()

            VStack(spacing: 16) {
                StressRingView(score: score)

                // Placeholder cards — you will replace with HealthKit values later
                VStack(spacing: 10) {
                    MetricCard(title: "Sleep", value: "—")
                    MetricCard(title: "Resting HR", value: "—")
                    MetricCard(title: "HRV (SDNN)", value: "—")
                }
                .padding(.horizontal)

                Slider(value: $score, in: 0...100, step: 1)
                    .padding(.horizontal)

                Spacer(minLength: 8)
            }
            .padding(.top, 8)
        }
    }
}

private struct MetricCard: View {
    let title: String
    let value: String

    var body: some View {
        HStack {
            Text(title)
            Spacer()
            Text(value).monospacedDigit()
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }
}
