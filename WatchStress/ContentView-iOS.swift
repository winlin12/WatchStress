import SwiftUI

struct ContentView: View {
    @AppStorage("didAcceptDisclaimer") private var didAcceptDisclaimer = false

    var body: some View {
        RootDashboardView()
            .sheet(isPresented: .constant(!didAcceptDisclaimer)) {
                DisclaimerView(didAcceptDisclaimer: $didAcceptDisclaimer)
                    .interactiveDismissDisabled(true)
            }
    }
}

struct DisclaimerView: View {
    @Binding var didAcceptDisclaimer: Bool
    @State private var dontShowAgain = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Disclaimer")
                .font(.title2).bold()

            Text("WatchStress is for educational/research purposes and wellness reflection only. It is not a medical device and does not provide diagnosis or treatment.")
                .font(.body)

            Text("If you feel unsafe or in crisis, please contact local emergency services or a trusted professional.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            Toggle("Don’t show this again", isOn: $dontShowAgain)
                .padding(.top, 8)

            Button {
                didAcceptDisclaimer = true
            } label: {
                Text("I understand")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)

            Spacer()
        }
        .padding()
    }
}
