import SwiftUI
import Foundation

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss

    let useDebugOverrides: Bool
    @Binding var logScheduleEnabled: Bool
    @Binding var logScheduleSlot1Enabled: Bool
    @Binding var logScheduleSlot2Enabled: Bool
    @Binding var logScheduleSlot3Enabled: Bool
    @Binding var logScheduleSlot1Minutes: Int
    @Binding var logScheduleSlot2Minutes: Int
    @Binding var logScheduleSlot3Minutes: Int
    @Binding var logScheduleSkipSleep: Bool
    @Binding var logScheduleSkipExercise: Bool

    let computeScore: () -> Void
    let openScoreDetails: () -> Void
    let openDebugTuning: () -> Void
    let saveLogEntry: () async -> String?
    let deleteAllLogs: () async -> String?

    @State private var statusAlert: StatusAlert? = nil
    @State private var showDeleteConfirm: Bool = false
    @State private var showDataView: Bool = false
    @State private var exportURL: URL? = nil

    var body: some View {
        NavigationStack {
            Form {
                Section("Scoring") {
                    Button {
                        computeScore()
                    } label: {
                        Label("Compute Score", systemImage: "function")
                    }

                    Button {
                        dismiss()
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                            openScoreDetails()
                        }
                    } label: {
                        Label("Score Details", systemImage: "list.bullet.rectangle")
                    }

                    Button {
                        dismiss()
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                            openDebugTuning()
                        }
                    } label: {
                        Label("Debug Tuning", systemImage: "slider.horizontal.3")
                    }
                }

                Section("Logging") {
                    Button {
                        Task {
                            let error = await saveLogEntry()
                            statusAlert = StatusAlert(
                                title: error == nil ? "Log Saved" : "Log Failed",
                                message: error ?? "Saved a stress log snapshot."
                            )
                            await refreshExportURL(showAlertOnError: false)
                        }
                    } label: {
                        Label("Save Log Entry", systemImage: "square.and.arrow.down")
                    }

                    if useDebugOverrides {
                        Text("Debug overrides are on; logged HR/HRV/temperature will use the debug values.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }

                    Button(role: .destructive) {
                        showDeleteConfirm = true
                    } label: {
                        Label("Delete All Data", systemImage: "trash")
                    }
                }

                Section("Data") {
                    Button {
                        showDataView = true
                    } label: {
                        Label("View Data", systemImage: "list.bullet")
                    }

                    if let exportURL {
                        ShareLink(item: exportURL) {
                            Label("Export Data", systemImage: "square.and.arrow.up")
                        }
                    } else {
                        Button {
                            Task { await refreshExportURL(showAlertOnError: true) }
                        } label: {
                            Label("Export Data", systemImage: "square.and.arrow.up")
                        }
                    }
                }

                Section("Scheduled Logging") {
                    Toggle("Enable scheduled logging", isOn: $logScheduleEnabled)

                    if logScheduleEnabled {
                        scheduleRow(title: "Slot 1", isOn: $logScheduleSlot1Enabled, minutes: $logScheduleSlot1Minutes)
                        scheduleRow(title: "Slot 2", isOn: $logScheduleSlot2Enabled, minutes: $logScheduleSlot2Minutes)
                        scheduleRow(title: "Slot 3", isOn: $logScheduleSlot3Enabled, minutes: $logScheduleSlot3Minutes)

                        Toggle("Skip if sleeping", isOn: $logScheduleSkipSleep)
                        Toggle("Skip if exercising", isOn: $logScheduleSkipExercise)

                        Text("Scheduled logs run only while the app is open. Pick times you are usually awake and not working out.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Settings")
            .alert(item: $statusAlert) { alert in
                Alert(title: Text(alert.title), message: Text(alert.message), dismissButton: .default(Text("OK")))
            }
            .confirmationDialog("Delete all logged data?", isPresented: $showDeleteConfirm, titleVisibility: .visible) {
                Button("Delete All Data", role: .destructive) {
                    Task {
                        let error = await deleteAllLogs()
                        statusAlert = StatusAlert(
                            title: error == nil ? "Logs Deleted" : "Delete Failed",
                            message: error ?? "All logged entries were removed."
                        )
                        await refreshExportURL(showAlertOnError: false)
                    }
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This removes the local stress log file. HealthKit data is not affected.")
            }
            .sheet(isPresented: $showDataView) {
                LogDataView()
                    .presentationDetents([.medium, .large])
                    .presentationDragIndicator(.visible)
            }
            .task {
                await refreshExportURL(showAlertOnError: false)
            }
        }
    }

    private struct StatusAlert: Identifiable {
        let id = UUID()
        let title: String
        let message: String
    }

    @MainActor private func refreshExportURL(showAlertOnError: Bool) async {
        do {
            exportURL = try await StressLogStore.shared.exportCSV()
        } catch {
            exportURL = nil
            if showAlertOnError {
                statusAlert = StatusAlert(title: "Export Failed", message: error.localizedDescription)
            }
        }
    }

    private func scheduleRow(title: String, isOn: Binding<Bool>, minutes: Binding<Int>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Toggle(title, isOn: isOn)
            DatePicker(
                "Time",
                selection: timeBinding(minutes),
                displayedComponents: .hourAndMinute
            )
            .labelsHidden()
            .disabled(!isOn.wrappedValue)
        }
    }

    private func timeBinding(_ minutes: Binding<Int>) -> Binding<Date> {
        Binding<Date>(
            get: {
                date(for: minutes.wrappedValue)
            },
            set: { newValue in
                minutes.wrappedValue = minutesSinceMidnight(for: newValue)
            }
        )
    }

    private func date(for minutes: Int) -> Date {
        let cal = Calendar.current
        let now = Date()
        let comps = cal.dateComponents([.year, .month, .day], from: now)
        var dateComps = DateComponents()
        dateComps.year = comps.year
        dateComps.month = comps.month
        dateComps.day = comps.day
        dateComps.hour = minutes / 60
        dateComps.minute = minutes % 60
        return cal.date(from: dateComps) ?? now
    }

    private func minutesSinceMidnight(for date: Date) -> Int {
        let comps = Calendar.current.dateComponents([.hour, .minute], from: date)
        let h = comps.hour ?? 0
        let m = comps.minute ?? 0
        return max(0, min(1439, (h * 60) + m))
    }
}

struct SettingsButtonRow: View {
    @Binding var showSettings: Bool

    var body: some View {
        HStack {
            Spacer()
            Button {
                showSettings = true
            } label: {
                Image(systemName: "gearshape")
                    .imageScale(.large)
                    .padding(8)
            }
            .accessibilityLabel("Settings")
        }
        .padding(.horizontal)
        .padding(.top, 8)
    }
}

private struct LogDataView: View {
    @State private var entries: [StressLogEntry] = []
    @State private var errorAlert: ErrorAlert? = nil

    var body: some View {
        NavigationStack {
            List {
                if entries.isEmpty {
                    Text("No logs yet.")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(entries) { entry in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(Self.dateFormatter.string(from: entry.timestamp))
                                .font(.headline)
                            Text("Score: \(format(entry.score, decimals: 1)) (\(entry.confidence))")
                                .font(.subheadline)
                            Text("HR: \(format(entry.heartRate)) bpm, Resting: \(format(entry.restingHR)) bpm")
                            Text("HRV: \(format(entry.hrvSDNN)) ms, Temp: \(format(entry.wristTemperature, decimals: 1)) C")
                            Text("Sleep: \(format(entry.sleepHours, decimals: 2)) h, Resp: \(format(entry.respirationRate)) bpm, SpO2: \(format(entry.bloodOxygen)) %")
                            Text("BP: \(format(entry.bloodPressureSystolic))/\(format(entry.bloodPressureDiastolic)) mmHg, Steps: \(format(entry.steps)), Exercise: \(format(entry.exerciseMinutes)) min")
                            if let note = entry.note, !note.isEmpty {
                                Text("Note: \(note)")
                            }
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
            .navigationTitle("Logged Data")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await loadEntries() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .accessibilityLabel("Reload")
                }
            }
            .task {
                await loadEntries()
            }
            .alert(item: $errorAlert) { alert in
                Alert(title: Text("Load Failed"), message: Text(alert.message), dismissButton: .default(Text("OK")))
            }
        }
    }

    private struct ErrorAlert: Identifiable {
        let id = UUID()
        let message: String
    }

    private static let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter
    }()

    @MainActor private func loadEntries() async {
        do {
            let loaded = try await StressLogStore.shared.loadAll()
            entries = loaded.sorted(by: { $0.timestamp > $1.timestamp })
        } catch {
            errorAlert = ErrorAlert(message: error.localizedDescription)
        }
    }

    private func format(_ value: Double?, decimals: Int = 0) -> String {
        guard let value else { return "N/A" }
        return String(format: "%.\(decimals)f", value)
    }
}
