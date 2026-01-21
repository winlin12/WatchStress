import SwiftUI

/// WatchRootView is the main entry for the watchOS app.
/// It will host quick actions like logging a check-in and, in the future,
/// glanceable metrics or complications.
struct WatchRootView: View {
    @State private var score: Double = 72

    var body: some View {
        VStack(spacing: 10) {
            // Placeholder action for a future quick check-in flow
            Button("Log Check-in") {
                // TODO: your action later (e.g., open check-in screen)
            }
            // Bordered style for better affordance on watchOS
            .buttonStyle(.bordered)
        }
        .padding(.horizontal, 8)
    }
}

