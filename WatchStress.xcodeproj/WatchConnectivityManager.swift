import Foundation
#if canImport(WatchConnectivity)
import WatchConnectivity
#endif

final class WatchConnectivityManager: NSObject, ObservableObject {
    #if canImport(WatchConnectivity)
    private var session: WCSession? { WCSession.isSupported() ? WCSession.default : nil }
    #endif

    override init() {
        super.init()
        #if canImport(WatchConnectivity)
        if let session = session {
            session.delegate = self
            session.activate()
        }
        #endif
    }

    func sendCheckIn(value: Int, tags: [String]) {
        #if canImport(WatchConnectivity)
        guard let session = session, session.isPaired, session.isReachable else { return }
        let payload: [String: Any] = [
            "type": "checkin",
            "value": value,
            "tags": tags,
            "timestamp": Date().timeIntervalSince1970
        ]
        session.sendMessage(payload, replyHandler: nil, errorHandler: nil)
        #endif
    }
}

#if canImport(WatchConnectivity)
extension WatchConnectivityManager: WCSessionDelegate {
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {}
    #if os(iOS)
    func sessionDidBecomeInactive(_ session: WCSession) {}
    func sessionDidDeactivate(_ session: WCSession) { session.activate() }
    #endif

    func session(_ session: WCSession, didReceiveMessage message: [String : Any]) {
        // Handle incoming messages if needed later
    }
}
#endif
