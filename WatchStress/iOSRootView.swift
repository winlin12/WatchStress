import SwiftUI

struct iOSRootView: View {
    @State private var score: Double = 72
    @StateObject private var vitals = VitalsViewModel()
    @State private var showVitalsSheet: Bool = false

    var body: some View {
        ZStack {
            Color.yellow.opacity(0.18).ignoresSafeArea()

            VStack(spacing: 16) {
                StressRingView(score: score)

                Button {
                    showVitalsSheet = true
                } label: {
                    Label("Show Vitals", systemImage: "heart.text.square")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .padding(.horizontal)

                Slider(value: $score, in: 0...100, step: 1)
                    .padding(.horizontal)

                Spacer(minLength: 8)
            }
            .padding(.top, 8)
        }
        .sheet(isPresented: $showVitalsSheet) {
            VitalsSheetView(vm: vitals)
                .presentationDragIndicator(.visible)
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
private struct VitalsSheetView: View {
    @ObservedObject var vm: VitalsViewModel

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 10) {
                    MetricCard(title: "Sleep (last night)", value: vm.sleep)
                    MetricCard(title: "Heart Rate (latest)", value: vm.heartRate)
                    MetricCard(title: "Resting HR", value: vm.restingHR)
                    MetricCard(title: "HRV (SDNN)", value: vm.hrvSDNN)
                    MetricCard(title: "Blood Pressure", value: vm.bloodPressure)
                    MetricCard(title: "Wrist Temperature", value: vm.wristTemperature)
                    MetricCard(title: "Respiration Rate", value: vm.respirationRate)
                    MetricCard(title: "Blood Oxygen", value: vm.bloodOxygen)
                    MetricCard(title: "Steps (today)", value: vm.steps)
                    MetricCard(title: "Exercise Minutes", value: vm.exerciseMinutes)
                }
                .padding(.horizontal)
                .padding(.top)
            }
            .navigationTitle("Vitals")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await vm.reloadAll() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .accessibilityLabel("Reload")
                }
            }
            .onAppear {
                if !vm.authorized {
                    vm.requestAuthorizationAndLoad()
                } else {
                    Task { await vm.reloadAll() }
                }
            }
        }
    }
}

