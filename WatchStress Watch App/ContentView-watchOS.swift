import SwiftUI
import WatchKit

struct ContentView: View {
    var body: some View {
        NavigationStack {
            RootDashboardView()
                .navigationBarTitleDisplayMode(.inline)
        }
    }
}
