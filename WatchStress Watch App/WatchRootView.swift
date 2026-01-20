import SwiftUI

struct WatchRootView: View {
    @State private var score: Double = 72

    var body: some View {
        VStack(spacing: 10) {
            StressRingView(score: score)
                .padding(.top, 6)

            // Placeholder for quick check-in later
            Button("Log Check-in") {
                // TODO: your action later (e.g., open check-in screen)
            }
            .buttonStyle(.bordered)
        }
        .padding(.horizontal, 8)
    }
}
