import SwiftUI

struct CheckInControl: View {
    @State private var value: Double = 5
    var onSubmit: (Int) -> Void

    var body: some View {
        VStack(spacing: 6) {
            HStack {
                Text("Check-in")
                    .font(.headline)
                Spacer()
                Text("\(Int(value)) / 10")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            Slider(value: $value, in: 0...10, step: 1) { _ in
                #if os(watchOS)
                WKInterfaceDevice.current().play(.click)
                #endif
            }
            Button {
                onSubmit(Int(value))
            } label: {
                Label("Send", systemImage: "paperplane.fill")
            }
            .buttonStyle(.borderedProminent)
        }
    }
}

#Preview {
    CheckInControl { _ in }
}
