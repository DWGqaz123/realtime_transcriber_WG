//
//  RealtimeTranscriberMacApp.swift
//  RealtimeTranscriberMac
//
//  Created by 董文光 on 2026/1/12.
//


import SwiftUI

/// 负责在 App 退出时关掉后端子进程。
/// 没有这一步，backend 会留在后台占着端口，下次启动就连不上。
final class AppDelegate: NSObject, NSApplicationDelegate {
    weak var backendManager: BackendManager?

    func applicationWillTerminate(_ notification: Notification) {
        backendManager?.stopBackend()
    }
}

@main
struct RealtimeTranscriberMacApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var backendManager = BackendManager()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(backendManager)
                .onAppear {
                    appDelegate.backendManager = backendManager
                    // 🔧 设置通知监听器
                    setupNotificationObservers()
                    

                }
        }
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About Realtime Transcriber") {
                    let statusText = backendManager.isRunning ? "✅ Running" : "❌ Stopped"
                    let errorText = backendManager.error ?? "No errors"
                    
                    NSApplication.shared.orderFrontStandardAboutPanel(
                        options: [
                            .credits: NSAttributedString(
                                string: "Backend Status: \(statusText)\n\(errorText)",
                                attributes: [
                                    .font: NSFont.systemFont(ofSize: 10),
                                    .foregroundColor: NSColor.secondaryLabelColor
                                ]
                            ),
                            .applicationVersion: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? ""
                        ]
                    )
                }
            }
        }
        
        // 添加设置窗口
        Settings {
            SettingsView()
        }
    }
    
    // MARK: - Private Methods
    
    /// 设置通知监听器
    private func setupNotificationObservers() {
        // 🔧 监听重启后端通知
        NotificationCenter.default.addObserver(
            forName: .restartBackend,
            object: nil,
            queue: .main
        ) { [backendManager] _ in
            // 停止后端（stopBackend 会等它真正退出）
            backendManager.stopBackend()

            // 重启时必须新起进程，不能复用可能还在跑的旧实例（否则新 key 不生效）
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                backendManager.startBackend(reuseExisting: false)
            }
        }
    }
}
