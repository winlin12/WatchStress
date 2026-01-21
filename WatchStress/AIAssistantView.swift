import SwiftUI
import FoundationModels

/// A SwiftUI view that integrates with Foundation Models to perform independent text generation.
struct AIAssistantView: View {
    @State private var prompt: String = ""
    @State private var responseText: String = ""
    @State private var isGenerating: Bool = false
    @State private var modelAvailable: Bool = false
    @State private var errorMessage: String?
    
    /// Health metrics source and current score passed in from the host view
    @ObservedObject var vm: VitalsViewModel
    var score: Double
    @State private var useContext: Bool = true

    /// Reference to the system language model to check availability.
    private let model = SystemLanguageModel.default

    var body: some View {
        VStack(spacing: 16) {
            Text("AI Assistant")
                .font(.title)
                .bold()
                .accessibilityAddTraits(.isHeader)

            TextEditor(text: $prompt)
                .border(Color.gray, width: 1)
                .frame(height: 150)
                .accessibilityLabel("Prompt input")
            
            Toggle(isOn: $useContext) {
                Label("Use vitals context", systemImage: "heart.text.square")
            }
            .toggleStyle(.switch)
            .accessibilityLabel("Use vitals context toggle")

            if let errorMessage = errorMessage {
                Text(errorMessage)
                    .foregroundColor(.red)
                    .accessibilityLabel("Error message")
            }

            Button(action: generateText) {
                if isGenerating {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle())
                        .accessibilityLabel("Generating text")
                } else {
                    Text("Generate")
                        .bold()
                        .accessibilityLabel("Generate text button")
                }
            }
            .disabled(isGenerating || !modelAvailable || prompt.isEmpty)
            .buttonStyle(.borderedProminent)

            ScrollView {
                Text(responseText)
                    .padding()
                    .accessibilityLabel("Generated response")
            }
            .frame(maxHeight: 300)
            .border(Color.gray, width: 1)

            Spacer()
        }
        .padding()
        .task {
            checkModelAvailability()
        }
        .onAppear {
            if !vm.authorized {
                vm.requestAuthorizationAndLoad()
            }
        }
    }

    /// Checks if the on-device language model is available and updates UI state.
    @MainActor
    private func checkModelAvailability() {
        switch model.availability {
        case .available:
            modelAvailable = true
            errorMessage = nil
        case .unavailable(.deviceNotEligible):
            modelAvailable = false
            errorMessage = "Device not eligible for Apple Intelligence."
        case .unavailable(.appleIntelligenceNotEnabled):
            modelAvailable = false
            errorMessage = "Please enable Apple Intelligence in Settings."
        case .unavailable(.modelNotReady):
            modelAvailable = false
            errorMessage = "Model is downloading or not ready."
        case .unavailable(let other):
            modelAvailable = false
            errorMessage = "Model unavailable: \(other)"
        }
    }

    /// Generates text based on the current prompt asynchronously.
    private func generateText() {
        guard !prompt.isEmpty else { return }
        isGenerating = true
        responseText = ""
        errorMessage = nil

        // Build optional instructions that summarize current vitals and score
        let instructions: String? = {
            guard useContext else { return nil }
            let summary = """
            You are a wellness assistant. Base your answer primarily on the provided context.
            If a value is unavailable (shown as "—"), acknowledge it and avoid guessing.
            Be concise and actionable.
            
            Context:
            - Stress score: \(Int(score.rounded()))
            - Sleep (last night): \(vm.sleep)
            - Heart Rate (latest): \(vm.heartRate)
            - Resting HR: \(vm.restingHR)
            - HRV (SDNN): \(vm.hrvSDNN)
            - Respiration Rate: \(vm.respirationRate)
            - Blood Oxygen: \(vm.bloodOxygen)
            - Steps (today): \(vm.steps)
            - Exercise Minutes: \(vm.exerciseMinutes)
            - Wrist Temperature: \(vm.wristTemperature)
            - Blood Pressure: \(vm.bloodPressure)
            """
            return summary
        }()

        Task {
            do {
                // Create a session per request; include instructions when using vitals context
                let session = instructions != nil ? LanguageModelSession(instructions: instructions!) : LanguageModelSession()
                let response = try await session.respond(to: prompt)
                await MainActor.run {
                    responseText = response.content
                    isGenerating = false
                }
            } catch {
                await MainActor.run {
                    errorMessage = "Failed to generate text: \(error.localizedDescription)"
                    isGenerating = false
                }
            }
        }
    }
}

