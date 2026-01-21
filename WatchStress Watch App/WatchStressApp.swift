//
//  WatchStressApp.swift
//  WatchStress Watch App
//
//  Created by Winston Lin on 1/10/26.
//

import SwiftUI

/// Main entry point for the watchOS app.
/// Creates the initial window group and shows WatchRootView.
@main
struct WatchStress_Watch_AppApp: App {
    var body: some Scene {
        WindowGroup {
            WatchRootView() // Root content view for the watch app
        }
    }
}
