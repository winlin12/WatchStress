import SwiftUI

struct DisclaimerView: View {
    @Environment(\.dismiss) private var dismiss
    @Binding var hasSeenDisclaimer: Bool

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Important Notice")
                        .font(.title3).bold()
                    Text("WatchStress is not a medical device and does not provide medical advice. It presents general wellness information derived from Apple Health data and optional self check-ins. Consult a qualified professional for medical concerns.")
                    Text("Data & Privacy")
                        .font(.headline)
                        .padding(.top, 8)
                    Text("Your data stays on-device. The app reads Apple Health data you permit and stores derived values locally. You can revoke Health permissions at any time in Settings → Health.")
                    Text("Limitations")
                        .font(.headline)
                        .padding(.top, 8)
                    Text("Some metrics (like HRV) may be unavailable or intermittent. The app shows a confidence indicator to reflect data quality.")
                }
                .padding()
            }
            .navigationTitle("Disclaimer")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Don’t show again") {
                        hasSeenDisclaimer = true
                        dismiss()
                    }
                }
            }
        }
    }
}

#Preview {
    DisclaimerView(hasSeenDisclaimer: .constant(false))
}
