import Foundation

enum AppConfig {
    private static let defaultHost = "127.0.0.1"
    private static let defaultPort = 9123

    static var backendHost: String {
        let value = UserDefaults.standard.string(forKey: "backend_host")?.trimmingCharacters(in: .whitespacesAndNewlines)
        return (value?.isEmpty == false) ? value! : defaultHost
    }

    static var backendPort: Int {
        let value = UserDefaults.standard.integer(forKey: "backend_port")
        return (1...65535).contains(value) ? value : defaultPort
    }

    static var apiBaseURL: URL {
        URL(string: "http://\(backendHost):\(backendPort)")!
    }

    static var websocketURL: URL {
        URL(string: "ws://\(backendHost):\(backendPort)/ws/transcribe")!
    }

    static var healthURL: URL {
        URL(string: "http://\(backendHost):\(backendPort)/health")!
    }
}
